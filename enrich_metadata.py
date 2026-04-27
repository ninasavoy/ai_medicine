from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

INPUT_METADATA = Path("processed_audio/metadata.csv")
INPUT_DIAGNOSIS_CANDIDATES = (
    Path("data/diagnosis.txt"),
    Path("diagnosis.txt"),
)
OUTPUT_ENRICHED = Path("processed_audio/metadata.csv")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


class MetadataEnrichmentError(RuntimeError):
    """Raised when input validation fails for metadata enrichment."""


def _load_metadata_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise MetadataEnrichmentError("Metadata CSV is empty.")

    required_cols = {
        "patient_id",
        "original_file",
        "cycle_index",
        "spectrogram_path",
        "start_time",
        "end_time",
        "crackles",
        "wheezes",
    }
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        raise MetadataEnrichmentError(
            f"Metadata is missing required columns: {sorted(missing_cols)}"
        )

    for row in rows:
        row["patient_id"] = str(row["patient_id"]).strip()
        row["original_file"] = str(row["original_file"]).strip()
        row["spectrogram_path"] = str(row["spectrogram_path"]).strip()

    return rows


def _load_simple_mapping(path: Path, *, name: str) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise MetadataEnrichmentError(
                    f"Malformed {name} line {line_no}: '{line}'. Expected exactly 2 fields."
                )

            key, value = parts
            if key in mapping and mapping[key] != value:
                raise MetadataEnrichmentError(
                    f"Conflicting {name} mapping for key '{key}': "
                    f"'{mapping[key]}' vs '{value}'."
                )
            mapping[key] = value

    if not mapping:
        raise MetadataEnrichmentError(
            f"{name} file is empty. Add lines like '101 URTI' in {path}."
        )

    return mapping


def _resolve_diagnosis_file() -> Path:
    for candidate in INPUT_DIAGNOSIS_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in INPUT_DIAGNOSIS_CANDIDATES)
    raise FileNotFoundError(f"Diagnosis file not found. Looked for: {searched}")


def _validate_set_match(
    metadata_patients: set[str],
    diagnosis_patients: set[str],
    *,
    context: str,
) -> None:
    missing_in_secondary = sorted(metadata_patients - diagnosis_patients)
    extra_in_secondary = sorted(diagnosis_patients - metadata_patients)

    if missing_in_secondary or extra_in_secondary:
        preview_missing = missing_in_secondary[:10]
        preview_extra = extra_in_secondary[:10]
        raise MetadataEnrichmentError(
            f"Patient ID mismatch ({context}). "
            f"Missing in secondary: {preview_missing} "
            f"(total={len(missing_in_secondary)}), "
            f"Extra in secondary: {preview_extra} "
            f"(total={len(extra_in_secondary)})."
        )


def _assign_patient_level_splits(
    patient_ids: Iterable[str],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise MetadataEnrichmentError(
            "Split ratios must sum to 1.0. "
            f"Received train={train_ratio}, validation={validation_ratio}, test={test_ratio}."
        )

    shuffled = sorted(set(patient_ids))
    if not shuffled:
        raise MetadataEnrichmentError("No patient IDs found in metadata.")

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)
    test_count = total - train_count - validation_count

    if train_count <= 0 or validation_count <= 0 or test_count <= 0:
        raise MetadataEnrichmentError(
            "Split counts must all be > 0. "
            f"Got train={train_count}, validation={validation_count}, test={test_count}, total={total}."
        )

    train_ids = set(shuffled[:train_count])
    validation_ids = set(shuffled[train_count : train_count + validation_count])
    test_ids = set(shuffled[train_count + validation_count :])

    result: dict[str, str] = {}
    for pid in train_ids:
        result[pid] = "train"
    for pid in validation_ids:
        result[pid] = "validation"
    for pid in test_ids:
        result[pid] = "test"

    return result


def _as_float(value: str, *, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise MetadataEnrichmentError(
            f"Invalid numeric value in field '{field_name}': {value}"
        ) from exc


def _validate_spectrogram_paths(rows: list[dict[str, str]]) -> None:
    missing_paths: list[str] = []
    for row in rows:
        path = Path(row["spectrogram_path"])
        if not path.exists():
            missing_paths.append(path.as_posix())

    if missing_paths:
        raise MetadataEnrichmentError(
            "Found missing spectrogram files referenced in metadata. "
            f"Examples: {missing_paths[:10]} (total={len(missing_paths)})."
        )


def enrich_metadata() -> None:
    rows = _load_metadata_rows(INPUT_METADATA)
    diagnosis_file = _resolve_diagnosis_file()
    diagnosis_map = _load_simple_mapping(diagnosis_file, name="diagnosis")

    metadata_patients = {row["patient_id"] for row in rows}
    _validate_set_match(
        metadata_patients,
        set(diagnosis_map.keys()),
        context="metadata vs diagnosis",
    )

    split_map = _assign_patient_level_splits(
        metadata_patients,
        seed=RANDOM_SEED,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
    )

    for row in rows:
        start = _as_float(row["start_time"], field_name="start_time")
        end = _as_float(row["end_time"], field_name="end_time")
        duration = end - start
        if duration <= 0:
            raise MetadataEnrichmentError(
                f"Invalid cycle duration (<= 0) for {row['original_file']} cycle {row['cycle_index']}."
            )

        row["duration"] = f"{duration:.6f}"
        row["diagnosis"] = diagnosis_map[row["patient_id"]]
        row["split"] = split_map[row["patient_id"]]

    # Final checks: no missing diagnosis/split values
    missing_diag = sum(1 for row in rows if not str(row.get("diagnosis", "")).strip())
    missing_split = sum(1 for row in rows if not str(row.get("split", "")).strip())
    missing_duration = sum(1 for row in rows if not str(row.get("duration", "")).strip())
    if missing_diag > 0 or missing_split > 0 or missing_duration > 0:
        raise MetadataEnrichmentError(
            "Missing values after enrichment: "
            f"diagnosis={missing_diag}, split={missing_split}, duration={missing_duration}."
        )

    _validate_spectrogram_paths(rows)

    # Final checks: single diagnosis and split per patient
    patient_to_diag: dict[str, set[str]] = {}
    patient_to_split: dict[str, set[str]] = {}
    for row in rows:
        pid = row["patient_id"]
        patient_to_diag.setdefault(pid, set()).add(row["diagnosis"])
        patient_to_split.setdefault(pid, set()).add(row["split"])

    bad_diag = {pid: sorted(vals) for pid, vals in patient_to_diag.items() if len(vals) > 1}
    bad_split = {pid: sorted(vals) for pid, vals in patient_to_split.items() if len(vals) > 1}

    if bad_diag:
        raise MetadataEnrichmentError(
            "Diagnosis inconsistency per patient detected: "
            f"{list(bad_diag.items())[:10]}"
        )
    if bad_split:
        raise MetadataEnrichmentError(
            "Split inconsistency per patient detected: "
            f"{list(bad_split.items())[:10]}"
        )

    output_columns = [
        "patient_id",
        "original_file",
        "cycle_index",
        "spectrogram_path",
        "start_time",
        "end_time",
        "duration",
        "crackles",
        "wheezes",
        "diagnosis",
        "split",
    ]

    OUTPUT_ENRICHED.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_ENRICHED.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in output_columns})

    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    patient_split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    patient_to_split: dict[str, str] = {}
    for row in rows:
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        patient_to_split.setdefault(row["patient_id"], split)

    for split in patient_to_split.values():
        patient_split_counts[split] = patient_split_counts.get(split, 0) + 1

    print(f"Saved enriched metadata: {OUTPUT_ENRICHED.resolve()}")
    print(f"Rows: {len(rows)}")
    print(f"Patients: {len(metadata_patients)}")
    print(
        "Cycle split counts: "
        f"train={split_counts.get('train', 0)}, "
        f"validation={split_counts.get('validation', 0)}, "
        f"test={split_counts.get('test', 0)}"
    )
    print(
        "Patient split counts: "
        f"train={patient_split_counts.get('train', 0)}, "
        f"validation={patient_split_counts.get('validation', 0)}, "
        f"test={patient_split_counts.get('test', 0)}"
    )


if __name__ == "__main__":
    enrich_metadata()
