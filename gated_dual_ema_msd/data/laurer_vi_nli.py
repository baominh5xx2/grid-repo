"""Pinned Vietnamese MT-NLI data and deterministic MACRO-BI batching for M25.

Only the five Vietnamese parquet shards from the immutable Laurer dataset are
accepted.  Training reads only the translated ``premise``/``hypothesis`` and
``label`` columns; the original English columns are schema-checked but never
loaded into the training row sequences.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, overload

REPO_ID = "MoritzLaurer/multilingual-NLI-26lang-2mil7"
REVISION = "510a233972a0d7ff0f767d82f46e046832c10538"
DOMAIN = "laurer_mtvi"
SPLIT_ORDER = ("vi_anli", "vi_fever", "vi_ling", "vi_mnli", "vi_wanli")
EXPECTED_SPLITS = {
    "vi_anli": {
        "remote_path": "data/vi_anli-00000-of-00001-7e787be6ac1fe1d9.parquet",
        "row_count": 25_000,
        "sha256": "894552c3b936e9d354efd941d54ea731a217df11f61dcca0bb6006eb1c47061e",
    },
    "vi_fever": {
        "remote_path": "data/vi_fever-00000-of-00001-f716544ff727fe83.parquet",
        "row_count": 25_000,
        "sha256": "b6991c2f28a2641b714b81e9dfc34cc381e6a9a1ff12991899c4b1e26c42dbae",
    },
    "vi_ling": {
        "remote_path": "data/vi_ling-00000-of-00001-9fffc9d6c215b4ab.parquet",
        "row_count": 5_000,
        "sha256": "7bc54b29d100b073887f55491460f49d349b38e06bf7e9f30fbce53b76ecc854",
    },
    "vi_mnli": {
        "remote_path": "data/vi_mnli-00000-of-00001-7932e0643171dfc4.parquet",
        "row_count": 25_000,
        "sha256": "7dca9bc3b65d7109eafa4b66716a377b06f8f4c5410d0463bb6a1005234a6027",
    },
    "vi_wanli": {
        "remote_path": "data/vi_wanli-00000-of-00001-6bcb1563253b7063.parquet",
        "row_count": 25_000,
        "sha256": "141e42e4a4dfeb7e28b5645c3d558baa81405b139b23910de010400b2fe369b4",
    },
}
EXPECTED_TOTAL_ROWS = 105_000
SOURCE_FIELDS = (
    "premise_original",
    "hypothesis_original",
    "label",
    "premise",
    "hypothesis",
)
TRAINING_SOURCE_FIELDS = ("premise", "hypothesis", "label")
EXCLUDED_SOURCE_FIELDS = ("premise_original", "hypothesis_original")
SOURCE_LABEL_TO_CANONICAL_ID = {0: 0, 1: 2, 2: 1}
SOURCE_LABEL_TO_CANONICAL_NAME = {0: "E", 1: "N", 2: "C"}
CANONICAL_LABEL_ORDER = ("E", "C", "N")
LICENSE_BINDING = {
    "status": "not_declared_in_pinned_repository",
    "spdx": None,
    "evidence": (
        "README.md front matter omits license and dataset_infos.json has an empty "
        f"license field at revision {REVISION}"
    ),
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_config_binding() -> dict[str, Any]:
    """Return the exact config-side provenance contract for M25."""
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "manifest": "data/external/laurer_vi_nli/manifest.json",
        "license": dict(LICENSE_BINDING),
        "source_fields": list(TRAINING_SOURCE_FIELDS),
        "excluded_source_fields": list(EXCLUDED_SOURCE_FIELDS),
        "label_mapping": {
            "source": {0: "E", 1: "N", 2: "C"},
            "source_to_canonical_id": dict(SOURCE_LABEL_TO_CANONICAL_ID),
            "canonical": {0: "E", 1: "C", 2: "N"},
        },
        "splits": {
            split: {
                "remote_path": values["remote_path"],
                "row_count": values["row_count"],
                "sha256": values["sha256"],
            }
            for split, values in EXPECTED_SPLITS.items()
        },
        "total_rows": EXPECTED_TOTAL_ROWS,
    }


def validate_config_binding(binding: Mapping[str, Any]) -> None:
    """Fail closed unless an M25 config pins every data/provenance field."""
    expected = expected_config_binding()
    if dict(binding) != expected:
        for key, value in expected.items():
            if binding.get(key) != value:
                raise ValueError(f"M25 auxiliary dataset binding mismatch for {key}")
        raise ValueError("M25 auxiliary dataset binding has unknown fields")


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only in a broken environment
        raise RuntimeError(
            "pyarrow is required to validate and load the pinned M25 parquet files"
        ) from exc
    return pa, pq


def _inspect_parquet(path: pathlib.Path, split: str) -> dict[str, Any]:
    pa, pq = _require_pyarrow()
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if tuple(schema.names) != SOURCE_FIELDS:
        raise RuntimeError(
            f"{split} field mismatch: {tuple(schema.names)} != {SOURCE_FIELDS}"
        )
    expected_types = {
        "premise_original": pa.string(),
        "hypothesis_original": pa.string(),
        "label": pa.int64(),
        "premise": pa.string(),
        "hypothesis": pa.string(),
    }
    for field, expected_type in expected_types.items():
        if schema.field(field).type != expected_type:
            raise RuntimeError(
                f"{split} field {field} has type {schema.field(field).type}, "
                f"expected {expected_type}"
            )

    row_count = parquet.metadata.num_rows
    expected_count = EXPECTED_SPLITS[split]["row_count"]
    if row_count != expected_count:
        raise RuntimeError(f"{split} row count {row_count} != expected {expected_count}")

    label_counts: Counter[int] = Counter()
    empty_translated_field_counts: Counter[str] = Counter()
    checked_rows = 0
    # Intentionally read only translated text plus label values.  The English
    # source columns are never materialized by this project.
    for batch in parquet.iter_batches(
        batch_size=8_192,
        columns=list(TRAINING_SOURCE_FIELDS),
        use_threads=False,
    ):
        premises = batch.column(0).to_pylist()
        hypotheses = batch.column(1).to_pylist()
        labels = batch.column(2).to_pylist()
        for offset, (premise, hypothesis, label) in enumerate(
            zip(premises, hypotheses, labels, strict=True)
        ):
            row_number = checked_rows + offset
            if not isinstance(premise, str):
                raise TypeError(f"{split} translated premise invalid at row {row_number}")
            if not isinstance(hypothesis, str):
                raise TypeError(f"{split} translated hypothesis invalid at row {row_number}")
            # Preserve the pinned release byte-for-semantics. The source has one
            # known empty translated premise (vi_mnli row 23618); never replace it
            # with the excluded English original or silently alter split counts.
            if not premise.strip():
                empty_translated_field_counts["premise"] += 1
            if not hypothesis.strip():
                empty_translated_field_counts["hypothesis"] += 1
            if label not in SOURCE_LABEL_TO_CANONICAL_ID:
                raise RuntimeError(f"{split} label {label!r} invalid at row {row_number}")
            label_counts[int(label)] += 1
        checked_rows += batch.num_rows
    if checked_rows != expected_count or sum(label_counts.values()) != expected_count:
        raise RuntimeError(f"{split} full parquet scan did not cover every row")
    if set(label_counts) != set(SOURCE_LABEL_TO_CANONICAL_ID):
        raise RuntimeError(f"{split} does not contain all source labels 0, 1, 2")
    canonical_counts = {
        SOURCE_LABEL_TO_CANONICAL_NAME[source_label]: count
        for source_label, count in label_counts.items()
    }
    return {
        "row_count": row_count,
        "source_label_distribution": {
            str(label): label_counts[label] for label in sorted(label_counts)
        },
        "canonical_label_distribution": {
            label: canonical_counts[label] for label in CANONICAL_LABEL_ORDER
        },
        "empty_translated_field_counts": {
            field: empty_translated_field_counts[field]
            for field in ("premise", "hypothesis")
        },
    }


def _manifest_static() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "multilingual-NLI-26lang-2mil7 Vietnamese MT subsets",
        "source": {
            "repo_id": REPO_ID,
            "repo_type": "dataset",
            "revision": REVISION,
            "license": dict(LICENSE_BINDING),
            "provenance": {
                "language": "vi",
                "translation": "pretranslated by the dataset authors; no runtime translation",
                "english_derived_sources": ["ANLI", "FEVER-NLI", "LingNLI", "MultiNLI", "WANLI"],
                "dataset_card": f"https://huggingface.co/datasets/{REPO_ID}/tree/{REVISION}",
            },
        },
        "usage_contract": {
            "source_fields": list(SOURCE_FIELDS),
            "training_source_fields": list(TRAINING_SOURCE_FIELDS),
            "excluded_source_fields": list(EXCLUDED_SOURCE_FIELDS),
            "label_mapping": {
                "source": {"0": "E", "1": "N", "2": "C"},
                "source_to_canonical_id": {"0": 0, "1": 2, "2": 1},
                "canonical": {"0": "E", "1": "C", "2": "N"},
            },
        },
        "allowed_remote_files": [EXPECTED_SPLITS[split]["remote_path"] for split in SPLIT_ORDER],
        "total_rows": EXPECTED_TOTAL_ROWS,
    }


def materialize_dataset(out_dir: pathlib.Path, force_download: bool = False) -> dict[str, Any]:
    """Download exactly five pinned ``vi_*`` parquet files and write a manifest."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is mandatory in real runs
        raise RuntimeError("huggingface_hub is required to materialize M25 data") from exc

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        expected = EXPECTED_SPLITS[split]
        downloaded = pathlib.Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=REVISION,
                filename=expected["remote_path"],
                local_dir=out_dir,
                force_download=force_download,
            )
        )
        expected_local = out_dir / expected["remote_path"]
        if downloaded.resolve() != expected_local.resolve():
            raise RuntimeError(f"unexpected local path for {split}: {downloaded}")
        digest = sha256_file(downloaded)
        if digest != expected["sha256"]:
            raise RuntimeError(f"{split} SHA256 {digest} != pinned {expected['sha256']}")
        inspected = _inspect_parquet(downloaded, split)
        split_manifest[split] = {
            "remote_path": expected["remote_path"],
            "local_path": expected["remote_path"],
            "sha256": digest,
            **inspected,
        }
        print(
            f"[prepare_laurer_vi_nli] {split}: {inspected['row_count']} rows "
            f"sha256={digest}",
            flush=True,
        )

    manifest = {**_manifest_static(), "splits": split_manifest}
    if sum(item["row_count"] for item in split_manifest.values()) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError("M25 auxiliary total row count mismatch")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verify_materialized_dataset(manifest_path, load_rows=False)
    return manifest


def _validate_manifest_static(manifest: Mapping[str, Any]) -> None:
    expected = _manifest_static()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"M25 auxiliary manifest mismatch for {key}")
    if set(manifest) != {*expected, "splits"}:
        raise RuntimeError("M25 auxiliary manifest contains unknown or missing top-level fields")
    if set(manifest.get("splits", {})) != set(SPLIT_ORDER):
        raise RuntimeError("M25 auxiliary manifest split set mismatch")


class ParquetTranslatedRows(Sequence[dict[str, Any]]):
    """Memory-efficient row view over one pinned parquet shard.

    The view always keeps the full five-column schema in memory.  Rows are
    served through :meth:`row` with the ``include_originals`` flag, so the
    M25/M26 invariant (no English columns in training rows) and the M27/M28
    variants (original fields required) are enforced at the row boundary
    rather than by loading separate tables.
    """

    def __init__(self, path: pathlib.Path, split: str):
        _, pq = _require_pyarrow()
        self.path = pathlib.Path(path)
        self.split = split
        self._table = pq.read_table(
            self.path,
            columns=list(SOURCE_FIELDS),
            memory_map=True,
            use_threads=False,
        )

    def __len__(self) -> int:
        return self._table.num_rows

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[dict[str, Any]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, Any] | Sequence[dict[str, Any]]:
        if isinstance(index, slice):
            raise TypeError("ParquetTranslatedRows supports integer indexing only")
        return self.row(index, include_originals=False)

    def get_raw(self, index: int) -> dict[str, Any]:
        """Every stored field for one row (English originals included)."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_label = int(self._table.column("label")[index].as_py())
        return {
            "id": f"laurer-{self.split}-{index:06d}",
            "premise": self._table.column("premise")[index].as_py(),
            "hypothesis": self._table.column("hypothesis")[index].as_py(),
            "premise_original": self._table.column("premise_original")[index].as_py(),
            "hypothesis_original": self._table.column("hypothesis_original")[index].as_py(),
            "label": SOURCE_LABEL_TO_CANONICAL_NAME[source_label],
            "domain": DOMAIN,
            "aux_split": self.split,
        }

    def row(self, index: int, *, include_originals: bool) -> dict[str, Any]:
        """One training row; English originals are only added when required."""
        raw = self.get_raw(index)
        row = {
            "id": raw["id"],
            "premise": raw["premise"],
            "hypothesis": raw["hypothesis"],
            "label": raw["label"],
            "domain": raw["domain"],
            "aux_split": raw["aux_split"],
        }
        if include_originals:
            row["premise_original"] = raw["premise_original"]
            row["hypothesis_original"] = raw["hypothesis_original"]
        return row


def verify_materialized_dataset(
    manifest_path: pathlib.Path,
    config_binding: Mapping[str, Any] | None = None,
    *,
    load_rows: bool = True,
) -> tuple[dict[str, Sequence[dict[str, Any]]], dict[str, Any]]:
    """Verify manifest, hashes, schema, fields, labels and counts before loading."""
    manifest_path = pathlib.Path(manifest_path)
    if config_binding is not None:
        validate_config_binding(config_binding)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"M25 auxiliary manifest missing at {manifest_path}; run "
            "scripts/prepare_laurer_vi_nli.py"
        ) from exc
    _validate_manifest_static(manifest)

    rows: dict[str, Sequence[dict[str, Any]]] = {}
    total = 0
    root = manifest_path.parent.resolve()
    for split in SPLIT_ORDER:
        expected = EXPECTED_SPLITS[split]
        entry = manifest["splits"][split]
        expected_entry_keys = {
            "remote_path",
            "local_path",
            "sha256",
            "row_count",
            "source_label_distribution",
            "canonical_label_distribution",
            "empty_translated_field_counts",
        }
        if set(entry) != expected_entry_keys:
            raise RuntimeError(f"{split} manifest fields mismatch")
        if entry["remote_path"] != expected["remote_path"]:
            raise RuntimeError(f"{split} remote path mismatch")
        if entry["local_path"] != expected["remote_path"]:
            raise RuntimeError(f"{split} local path mismatch")
        if entry["sha256"] != expected["sha256"]:
            raise RuntimeError(f"{split} manifest SHA256 mismatch")
        if entry["row_count"] != expected["row_count"]:
            raise RuntimeError(f"{split} manifest row count mismatch")
        path = (manifest_path.parent / entry["local_path"]).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"{split} local path escapes the materialized data directory")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise RuntimeError(f"{split} on-disk SHA256 mismatch")
        inspected = _inspect_parquet(path, split)
        for key, value in inspected.items():
            if entry.get(key) != value:
                raise RuntimeError(f"{split} manifest {key} mismatch")
        if load_rows:
            rows[split] = ParquetTranslatedRows(path, split)
        total += inspected["row_count"]
    if total != EXPECTED_TOTAL_ROWS or manifest["total_rows"] != EXPECTED_TOTAL_ROWS:
        raise RuntimeError("M25 auxiliary total row count mismatch")
    return rows, dict(manifest)


class _IndexCycler:
    """Seeded, shuffled index stream that cycles without copying source rows."""

    def __init__(self, length: int, rng: random.Random):
        if length <= 0:
            raise ValueError("cannot cycle an empty source")
        self.length = length
        self.rng = rng
        self.queue: list[int] = []

    def next_indices(self, count: int) -> list[int]:
        indices: list[int] = []
        while len(indices) < count:
            if not self.queue:
                self.queue = list(range(self.length))
                self.rng.shuffle(self.queue)
            take = min(count - len(indices), len(self.queue))
            indices.extend(self.queue[:take])
            del self.queue[:take]
        return indices


class MtViMacroBiBatchMixer:
    """Exactly 50/50 native ViNLI and auxiliary NLI rows per physical batch.

    Auxiliary row sources are the pinned Laurer shards.  ``aux_splits`` limits
    which shards participate (M26/M28 use only ``vi_anli``); the default full
    set follows the published 25k/25k/5k/25k/25k composition.  ``text_mode``
    selects which fields become ``premise``/``hypothesis``:

    * ``mt_vi`` — translated Vietnamese only (never exposes English fields);
    * ``en`` — the original English fields of the same pinned rows;
    * ``code_switch_10`` — translated rows, with exactly 10% of auxiliary
      draws converted to cross-lingual pairs (alternating VI premise + EN
      hypothesis and EN premise + VI hypothesis), a controlled mapping of the
      dataset card's documented mix.

    The split schedule is row-proportional without replacement, so one epoch
    is a full pass over the selected auxiliary corpus; native ViNLI rows
    cycle alongside.  Deterministic under ``seed``.
    """

    def __init__(
        self,
        native_rows: Sequence[dict[str, Any]],
        auxiliary_rows: Mapping[str, Sequence[dict[str, Any]]],
        *,
        native_per_batch: int,
        auxiliary_per_batch: int,
        seed: int,
        text_mode: str = "mt_vi",
        aux_splits: Sequence[str] = SPLIT_ORDER,
        code_switch_fraction: float = 0.0,
        epoch_anchor: str = "aux",
    ):
        if native_per_batch <= 0 or auxiliary_per_batch <= 0:
            raise ValueError("per-batch source counts must be positive")
        if native_per_batch + auxiliary_per_batch != 4:
            raise ValueError("physical batch size must be 4")
        self.epoch_anchor = epoch_anchor
        if epoch_anchor not in ("aux", "native_full_pass"):
            raise ValueError(f"unsupported epoch anchor {epoch_anchor!r}")
        self.text_mode = text_mode
        if text_mode not in ("mt_vi", "en", "code_switch_10"):
            raise ValueError(f"unsupported M25 text mode {text_mode!r}")
        self.aux_splits = tuple(aux_splits)
        if set(self.aux_splits) != set(self.aux_splits) & set(SPLIT_ORDER):
            raise ValueError("aux_splits must be a subset of the pinned vi_* splits")
        if not self.aux_splits:
            raise ValueError("at least one auxiliary split is required")
        self.native_rows = native_rows
        self.auxiliary_rows = dict(auxiliary_rows)
        if set(self.auxiliary_rows) != set(self.aux_splits):
            raise ValueError(
                f"M25 auxiliary split set mismatch: {set(self.auxiliary_rows)} != {set(self.aux_splits)}"
            )
        if text_mode == "en" and self.aux_splits != ("vi_anli",):
            raise ValueError("M28 raw-English mode is registered for vi_anli only")
        self.native_per_batch = native_per_batch
        self.auxiliary_per_batch = auxiliary_per_batch
        self.code_switch_fraction = code_switch_fraction
        if not (0.0 <= code_switch_fraction <= 1.0):
            raise ValueError("code_switch_fraction must be in [0, 1]")

        seed_rng = random.Random(seed)
        self._native_cycler = _IndexCycler(
            len(native_rows), random.Random(seed_rng.randrange(2**31))
        )
        self._aux_cyclers = {
            split: _IndexCycler(
                len(self.auxiliary_rows[split]),
                random.Random(seed_rng.randrange(2**31)),
            )
            for split in self.aux_splits
        }
        # Row-proportional schedule over the selected splits.
        self._split_cycle = [
            split
            for split in self.aux_splits
            for _ in range(len(self.auxiliary_rows[split]))
        ]
        random.Random(seed).shuffle(self._split_cycle)
        self._split_position = 0
        if self.text_mode == "code_switch_10":
            # Exactly 10% of the cycle positions are code-switched; the
            # per-epoch direction alternates deterministically.
            n_aux = len(self._split_cycle)
            n_switched = n_aux // 10
            positions = list(range(n_aux))
            random.Random(seed).shuffle(positions)
            self._code_switch_positions: set[int] = set(positions[:n_switched])
        else:
            self._code_switch_positions = set()

    @property
    def batch_size(self) -> int:
        return self.native_per_batch + self.auxiliary_per_batch

    def epoch_steps(self) -> int:
        if self.epoch_anchor == "aux":
            return math.ceil(
                sum(len(self.auxiliary_rows[split]) for split in self.aux_splits)
                / self.auxiliary_per_batch
            )
        return math.ceil(len(self.native_rows) / self.native_per_batch)

    def full_epoch_aux_counts(self) -> dict[str, int]:
        return {
            split: len(self.auxiliary_rows[split]) for split in self.aux_splits
        }

    def _next_aux_split(self) -> str:
        split = self._split_cycle[self._split_position]
        self._split_position = (self._split_position + 1) % len(self._split_cycle)
        return split

    @staticmethod
    def _row_for_mode(raw: dict[str, Any], index: int, mode: str) -> dict[str, Any]:
        """Build the training row dict for one auxiliary raw row."""
        if mode == "en":
            return {
                "id": raw["id"],
                "premise": raw["premise_original"],
                "hypothesis": raw["hypothesis_original"],
                "label": raw["label"],
                "domain": raw["domain"],
                "aux_split": raw["aux_split"],
            }
        base = {
            "id": raw["id"],
            "premise": raw["premise"],
            "hypothesis": raw["hypothesis"],
            "label": raw["label"],
            "domain": raw["domain"],
            "aux_split": raw["aux_split"],
        }
        if mode == "mt_vi":
            return base
        # code_switch_10: alternating cross-lingual direction on marked draws.
        if raw.get("_code_switch_position") is None:
            return base
        position = int(raw["_code_switch_position"])
        if position not in raw["_code_switch_positions"]:
            return base
        direction = (position // 10) % 2
        if direction == 0:
            base["hypothesis"] = raw["hypothesis_original"]
        else:
            base["premise"] = raw["premise_original"]
        return base

    def _aux_row(self, split: str, index: int, position: int) -> dict[str, Any]:
        source = self.auxiliary_rows[split]
        if hasattr(source, "get_raw"):
            raw = source.get_raw(index)
            raw["_code_switch_position"] = position
            raw["_code_switch_positions"] = self._code_switch_positions
            row = self._row_for_mode(raw, position, self.text_mode)
            row.pop("_code_switch_position", None)
            row.pop("_code_switch_positions", None)
            return row
        return dict(source[index])

    def next_batch_rows(self) -> list[dict[str, Any]]:
        native_indices = self._native_cycler.next_indices(self.native_per_batch)
        batch = [self.native_rows[index] for index in native_indices]
        for _ in range(self.auxiliary_per_batch):
            split = self._next_aux_split()
            position = self._split_position - 1
            if position < 0:
                position += len(self._split_cycle)
            index = self._aux_cyclers[split].next_indices(1)[0]
            batch.append(self._aux_row(split, index, position))
        domains = Counter(row.get("domain") for row in batch)
        expected_domains = {DOMAIN: self.auxiliary_per_batch}
        native_domain = self.native_rows[0].get("domain") or "vianli"
        expected_domains[native_domain] = self.native_per_batch
        if domains != expected_domains:
            raise RuntimeError(
                f"auxiliary batch balance invariant failed: {dict(domains)} != {expected_domains}"
            )
        return batch

    def batches(self, n_steps: int | None = None):
        if n_steps is None:
            n_steps = self.epoch_steps()
        for _ in range(n_steps):
            yield self.next_batch_rows()
