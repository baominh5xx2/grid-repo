"""Migrate the reviewed ViNLI Flat CafeBERT checkpoint to canonical E/C/N order.

The source checkpoint was trained with the obsolete raw-label mapping 0=E, 1=N,
2=C while the classifier tensor is named in canonical E/C/N order.  Therefore rows
1 and 2 of both classifier weight and bias must be swapped exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
from collections.abc import Mapping

import torch
from huggingface_hub import hf_hub_download

SOURCE_REPO = "trinhtrantran122/hier-nli-e-first-flat-cafebert-vinli"
SOURCE_REVISION = "3ae7df0b009d14ece648dbdf4cfd88d7ad37f570"
SOURCE_CHECKPOINT_PATH = "checkpoint/pytorch_model.bin"
TOKENIZER_FILES = ("checkpoint/tokenizer.json", "checkpoint/tokenizer_config.json")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classifier_keys(state_dict: Mapping[str, torch.Tensor]) -> tuple[str, str]:
    weight = [k for k, v in state_dict.items() if k.endswith("classifier.weight") and v.ndim == 2 and v.shape[0] == 3]
    bias = [k for k, v in state_dict.items() if k.endswith("classifier.bias") and v.ndim == 1 and v.shape[0] == 3]
    if len(weight) != 1 or len(bias) != 1:
        raise ValueError(f"expected exactly one 3-way classifier weight/bias, found weight={weight}, bias={bias}")
    return weight[0], bias[0]


def migrate_state_dict(state_dict: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    """Clone a state dict and swap classifier rows 1/2, leaving all else unchanged."""
    migrated = {k: v.detach().clone() for k, v in state_dict.items()}
    weight_key, bias_key = classifier_keys(migrated)
    for key in (weight_key, bias_key):
        original = migrated[key].clone()
        migrated[key][1] = original[2]
        migrated[key][2] = original[1]
    return migrated, {
        "operation": "swap_classifier_rows_1_and_2",
        "source_order_as_trained": ["E", "N", "C"],
        "canonical_output_order": ["E", "C", "N"],
        "weight_key": weight_key,
        "bias_key": bias_key,
    }


def migrate_local_checkpoint(source_path: pathlib.Path, output_dir: pathlib.Path,
                             tokenizer_files: list[pathlib.Path] | None = None,
                             source_metadata: dict | None = None) -> dict:
    output_dir = pathlib.Path(output_dir)
    source_path = pathlib.Path(source_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"migration output must be fresh and empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(source_path, map_location="cpu", weights_only=True)
    migrated, operation = migrate_state_dict(state)
    output_path = output_dir / "pytorch_model.bin"
    torch.save(migrated, output_path)

    copied_tokenizer_files = []
    for src in tokenizer_files or []:
        src = pathlib.Path(src)
        dst = output_dir / src.name
        shutil.copyfile(src, dst)
        copied_tokenizer_files.append(dst.name)

    metadata = {
        "source_checkpoint_sha256": sha256_file(source_path),
        "migrated_checkpoint_sha256": sha256_file(output_path),
        "tokenizer_files": copied_tokenizer_files,
        **(source_metadata or {}),
        **operation,
    }
    (output_dir / "migration_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def migrate_checkpoint(output_dir: pathlib.Path, repo_id: str = SOURCE_REPO,
                       revision: str = SOURCE_REVISION, token: str | None = None) -> dict:
    source_path = pathlib.Path(hf_hub_download(
        repo_id=repo_id, revision=revision, filename=SOURCE_CHECKPOINT_PATH, token=token,
    ))
    tokenizer_files = [pathlib.Path(hf_hub_download(
        repo_id=repo_id, revision=revision, filename=remote_name, token=token,
    )) for remote_name in TOKENIZER_FILES]
    return migrate_local_checkpoint(
        source_path=source_path,
        output_dir=output_dir,
        tokenizer_files=tokenizer_files,
        source_metadata={
            "source_repo_id": repo_id,
            "source_revision": revision,
            "source_checkpoint_path": SOURCE_CHECKPOINT_PATH,
        },
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default=SOURCE_REPO)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    args = parser.parse_args(argv)
    metadata = migrate_checkpoint(
        pathlib.Path(args.output_dir), repo_id=args.repo_id, revision=args.revision,
        token=os.getenv("HF_TOKEN"),
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
