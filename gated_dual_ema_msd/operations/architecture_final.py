"""Inference-only final comparison, preserving immutable screening artifacts."""
from __future__ import annotations

import json
import math
import os
import pathlib
import tempfile

import pandas as pd
import torch
from huggingface_hub import CommitOperationAdd, hf_hub_download

from gated_dual_ema_msd.cli import evaluate
from gated_dual_ema_msd.cli.matrix import MatrixJob
from gated_dual_ema_msd.config.contracts import DATASET_REVISIONS, MODEL_REVISION
from gated_dual_ema_msd.data.loading import read_jsonl
from gated_dual_ema_msd.evaluation.predictions import validate_predictions
from gated_dual_ema_msd.operations.architecture_search import (
    _linked_manifests, confirmation_decision, environment_contract, verified_result,
)
from gated_dual_ema_msd.operations.notebook_batch import bind_manifest, prepare_data, write_json
from gated_dual_ema_msd.tracking.hf import _api, sha256_file
from gated_dual_ema_msd.training.direct import compute_metrics
from gated_dual_ema_msd.training.r2_runtime import environment_metadata

FINAL_LEDGER = "final_evaluation/evaluation.json"
FINAL_PREDICTIONS = "final_evaluation/test_predictions.csv"


def publish_final_evaluation(repo_id: str, prediction_path: pathlib.Path, ledger_path: pathlib.Path) -> str:
    """Add two files in one commit; never replace screening metadata or weights."""
    ledger = json.loads(pathlib.Path(ledger_path).read_text(encoding="utf-8"))
    if ledger.get("state") != "complete" or ledger.get("test_evaluations") != 1:
        raise ValueError("Cannot publish incomplete final evaluation")
    api = _api()
    parent = api.repo_info(repo_id=repo_id, repo_type="model").sha
    existing = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=parent)
    if FINAL_LEDGER in existing or FINAL_PREDICTIONS in existing:
        raise ValueError("Final evaluation already exists on HF; resume its recorded artifacts")
    commit = api.create_commit(repo_id=repo_id, repo_type="model", parent_commit=parent,
                               commit_message="Single-model final test inference; preserve screening artifacts",
                               operations=[CommitOperationAdd(path_in_repo=FINAL_LEDGER, path_or_fileobj=str(ledger_path)),
                                           CommitOperationAdd(path_in_repo=FINAL_PREDICTIONS, path_or_fileobj=str(prediction_path))])
    revision = commit.oid
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError("HF final commit is not an immutable revision")
    with tempfile.TemporaryDirectory(prefix="nli-final-readback-") as cache:
        for name, source in ((FINAL_LEDGER, ledger_path), (FINAL_PREDICTIONS, prediction_path)):
            remote = pathlib.Path(hf_hub_download(repo_id=repo_id, repo_type="model", filename=name,
                                                 revision=revision, token=os.environ.get("HF_TOKEN"), local_dir=cache))
            if sha256_file(remote) != sha256_file(source):
                raise RuntimeError(f"HF final read-back hash mismatch: {name}")
    return revision


def _validate_final(ledger: dict, predictions: pathlib.Path, source: dict, rows: list[dict], test_hash: str) -> None:
    expected = {"state": "complete", "dataset": "vinli", "experiment_id": source["experiment_id"],
                "seed": source["seed"], "test_evaluations": 1, "eval_precision": "fp32",
                "checkpoint_hf_revision": source["hf_revision"], "test_split_sha256": test_hash,
                "checkpoint_sha256": source["hf_checkpoint_sha256"], "split": "test", "max_length": 512,
                "model_revision": MODEL_REVISION, "data_revision": DATASET_REVISIONS["vinli"],
                "checkpoint_hf_repo_id": source["hf_repo_id"], "source_git_sha": source["git_sha"],
                "selection_policy": "dev_macro_f1"}
    if any(ledger.get(k) != v for k, v in expected.items()):
        raise ValueError("Existing final ledger differs from the frozen model/data")
    if ledger.get("prediction_sha256") != sha256_file(predictions):
        raise ValueError("Final prediction hash mismatch")
    frame = pd.read_csv(predictions, dtype={"sample_id": str})
    validate_predictions(frame, [str(r["id"]) for r in rows])
    if frame.gold_label.tolist() != [r["label"] for r in rows]:
        raise ValueError("Final prediction gold labels differ from pinned test")
    if len(frame) != ledger.get("row_count"):
        raise ValueError("Final prediction row count mismatch")
    mapping = {"E": 0, "C": 1, "N": 2}
    recomputed = compute_metrics(frame.pred_label.map(mapping).to_numpy(), frame.gold_label.map(mapping).to_numpy())
    for key in ("macro_f1", "accuracy"):
        if not math.isclose(ledger.get("metrics", {}).get(key, float("nan")), recomputed[key], abs_tol=1e-10):
            raise ValueError("Final ledger metrics differ from predictions")


def _recover_complete_ledger(ledger_path: pathlib.Path, prediction_path: pathlib.Path,
                             source: dict, rows: list[dict], test_hash: str) -> None:
    """Finish wrapper provenance after a completed CLI; never perform inference."""
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    additions = dict(seed=source["seed"], checkpoint_hf_repo_id=source["hf_repo_id"],
                     checkpoint_hf_revision=source["hf_revision"], source_git_sha=source["git_sha"],
                     selection_policy="dev_macro_f1", test_split_sha256=test_hash, prior_test_exposure=True)
    if any(k in ledger and ledger[k] != v for k, v in additions.items()):
        raise ValueError("Completed CLI ledger provenance differs from its frozen source")
    recovered = {**ledger, **additions}
    _validate_final(recovered, prediction_path, source, rows, test_hash)
    write_json(ledger_path, recovered)


def evaluate_final_batch(screening_root: pathlib.Path, confirmation_root: pathlib.Path, candidate: str,
                         repo_dir: pathlib.Path, output_root: pathlib.Path, hf_private: bool = False) -> dict:
    """Evaluate six existing selected checkpoints, once each; no training calls."""
    decision = confirmation_decision(screening_root, confirmation_root, candidate)
    if not decision.get("passes_confirmation"):
        raise ValueError("Final evaluation requires the paired three-seed dev confirmation gate")
    pilot = _linked_manifests(screening_root, confirmation_root, candidate)
    if environment_contract(environment_metadata()) != pilot["environment_contract"]:
        raise ValueError("Final runtime differs from the frozen study environment")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA runtime for final inference")
    fingerprints = prepare_data(pathlib.Path(repo_dir), ["vinli"], frozen_final=True)["vinli"]
    test_hash = fingerprints["splits"]["test"]["sha256"]
    rows = read_jsonl(pathlib.Path(repo_dir) / "data/external/vinli/test.jsonl")
    sources = []
    for seed in (42, 2024, 3407):
        root = screening_root if seed == 42 else confirmation_root
        for method in ("M3_FULL", candidate):
            source = verified_result(root, method, seed)
            if {k: source.get("data_fingerprints", {}).get("splits", {}).get(k) for k in ("train", "dev")} != {k: fingerprints["splits"][k] for k in ("train", "dev")}:
                raise ValueError("Final pinned data differs from screening data")
            sources.append(source)
    output_root = pathlib.Path(output_root)
    bind_manifest(output_root, dict(candidate=candidate, decision=decision, data_fingerprints=fingerprints,
                                   models=[dict(experiment_id=s["experiment_id"],seed=s["seed"],hf_repo_id=s["hf_repo_id"],hf_revision=s["hf_revision"]) for s in sources],
                                   prior_test_exposure=True, protocol="single_model_architecture_final"))
    api = _api()
    final = []
    for source in sources:
        job = MatrixJob("vinli", source["experiment_id"], source["seed"])
        folder = job.output_dir(output_root); folder.mkdir(parents=True, exist_ok=True)
        ledger_path = folder / "evaluation.json"
        prediction_path = folder / "test_predictions.csv"
        marker_path = folder / "verified_final.json"
        binding_path = folder / "final_source.json"
        binding = dict(experiment_id=source["experiment_id"], seed=source["seed"],
                       hf_repo_id=source["hf_repo_id"], hf_revision=source["hf_revision"],
                       checkpoint_sha256=source["hf_checkpoint_sha256"], git_sha=source["git_sha"],
                       test_split_sha256=test_hash, environment_contract=pilot["environment_contract"])
        if binding_path.exists():
            if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                raise ValueError("Final source binding differs; refusing resume")
        elif ledger_path.exists() and not marker_path.exists():
            raise ValueError("Unbound existing final ledger; no automatic test rerun")
        else:
            write_json(binding_path, binding)
        if marker_path.exists():
            ledger = json.loads(ledger_path.read_text())
            _validate_final(ledger, prediction_path, source, rows, test_hash)
            marker = json.loads(marker_path.read_text())
            revision = marker.get("hf_final_revision", "")
            if (not marker.get("artifact_readback_verified") or len(revision) != 40
                    or any(c not in "0123456789abcdef" for c in revision)
                    or any(marker.get(k) != v for k, v in ledger.items())):
                raise ValueError("Invalid final resume marker")
            final.append(marker)
            continue
        info = api.repo_info(repo_id=source["hf_repo_id"], repo_type="model")
        if bool(info.private) != bool(hf_private):
            raise ValueError("Final HF repository visibility differs from the configured study")
        files = api.list_repo_files(repo_id=source["hf_repo_id"], repo_type="model", revision=info.sha)
        remote_exists = FINAL_LEDGER in files or FINAL_PREDICTIONS in files
        if remote_exists:
            if FINAL_LEDGER not in files or FINAL_PREDICTIONS not in files:
                raise ValueError("Incomplete remote final artifacts; no automatic test rerun")
            with tempfile.TemporaryDirectory(prefix="nli-final-resume-") as cache:
                for name, destination in ((FINAL_LEDGER, ledger_path), (FINAL_PREDICTIONS, prediction_path)):
                    downloaded = pathlib.Path(hf_hub_download(repo_id=source["hf_repo_id"], filename=name,
                                                             revision=info.sha, token=os.environ.get("HF_TOKEN"), local_dir=cache))
                    destination.write_bytes(downloaded.read_bytes())
            revision = info.sha
        else:
            existing = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else None
            if existing is None or (existing.get("state") == "initializing" and existing.get("test_evaluations") == 0):
                with tempfile.TemporaryDirectory(prefix="nli-final-model-") as cache:
                    meta_path = hf_hub_download(repo_id=source["hf_repo_id"], filename="run_metadata.json",
                                                revision=source["hf_revision"], token=os.environ.get("HF_TOKEN"), local_dir=cache)
                    metadata = json.loads(pathlib.Path(meta_path).read_text())
                    identity_keys = ("dataset", "experiment_id", "seed", "git_sha", "batch_manifest_sha256",
                                     "data_fingerprints", "model_revision", "selected_weight_source")
                    if any(metadata.get(k) != source.get(k) for k in identity_keys):
                        raise ValueError("Immutable HF metadata differs from the verified screening source")
                    checkpoint = hf_hub_download(repo_id=source["hf_repo_id"], filename=source["hf_checkpoint_path"],
                                                  revision=source["hf_revision"], token=os.environ.get("HF_TOKEN"), local_dir=cache)
                    digest = sha256_file(pathlib.Path(checkpoint))
                    if digest != source["hf_checkpoint_sha256"] or digest != metadata["checkpoint_manifest"]["pytorch_model.bin"]["sha256"]:
                        raise ValueError("Selected HF checkpoint hash mismatch")
                    args = ["--checkpoint",str(checkpoint),"--dataset","vinli","--split","test",
                            "--experiment_id",job.experiment_id,"--frozen_final","--require_cuda",
                            "--output_csv",str(prediction_path),"--output_json",str(ledger_path)]
                    if existing is not None:
                        args.append("--resume_unstarted")
                    evaluate.main(args)
                    torch.cuda.empty_cache()
            _recover_complete_ledger(ledger_path, prediction_path, source, rows, test_hash)
            revision = publish_final_evaluation(source["hf_repo_id"], prediction_path, ledger_path)
        ledger = json.loads(ledger_path.read_text())
        _validate_final(ledger, prediction_path, source, rows, test_hash)
        marker = {**ledger,"hf_repo_id":source["hf_repo_id"],"hf_final_revision":revision,
                  "artifact_readback_verified":True}
        write_json(marker_path,marker); final.append(marker)
    table = pd.DataFrame([dict(experiment_id=r["experiment_id"],seed=r["seed"],
                               test_macro_f1=r["metrics"]["macro_f1"],test_accuracy=r["metrics"]["accuracy"],
                               hf_final_revision=r["hf_final_revision"]) for r in final])
    table.to_csv(output_root/"single_model_final_runs.csv",index=False)
    table.groupby("experiment_id").agg(seeds=("seed","nunique"),test_macro_f1_mean=("test_macro_f1","mean"),
                                         test_macro_f1_std=("test_macro_f1","std")).to_csv(output_root/"single_model_final_summary.csv")
    return dict(state="complete",candidate=candidate,verified_evaluations=len(final),output_root=str(output_root))
