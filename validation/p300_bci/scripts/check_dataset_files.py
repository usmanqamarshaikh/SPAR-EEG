from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "smoke_tests" / "dataset_file_audit.csv"


EXPECTED_ROOT_FILES = [
    "README",
    "CHANGES",
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "task-RSVPandP300speller_events.json",
]

EXPECTED_RUNS = [
    ("eyesopenrestingstate", 1),
    ("eyesclosedrestingstate", 2),
    ("RSVPtask", 3),
    ("eyesopenrestingstate", 4),
    ("eyesclosedrestingstate", 5),
    ("P300trainrun1", 6),
    ("P300trainrun2", 7),
    ("P300testrun1", 8),
    ("P300testrun2", 9),
    ("P300testrun3", 10),
    ("P300testrun4", 11),
    ("eyesopenrestingstate", 12),
    ("eyesclosedrestingstate", 13),
]

EXPECTED_SUFFIXES = [
    "_eeg.set",
    "_eeg.fdt",
    "_eeg.json",
    "_channels.tsv",
    "_electrodes.tsv",
    "_coordsystem.json",
]


@dataclass(frozen=True)
class AuditRow:
    subject: str
    task: str
    run: int
    suffix: str
    path: str
    present: bool
    size_bytes: int


def expected_file(subject: str, task: str, run: int, suffix: str) -> str:
    return f"{subject}_task-{task}_run-{run}{suffix}"


def audit_dataset(dataset_root: Path) -> tuple[list[str], list[AuditRow]]:
    errors: list[str] = []
    rows: list[AuditRow] = []

    if not dataset_root.exists():
        return [f"Dataset root does not exist: {dataset_root}"], rows

    for name in EXPECTED_ROOT_FILES:
        path = dataset_root / name
        if not path.is_file():
            errors.append(f"Missing root file: {name}")

    for idx in range(1, 56):
        subject = f"sub-{idx:03d}"
        eeg_dir = dataset_root / subject / "eeg"
        if not eeg_dir.is_dir():
            errors.append(f"Missing EEG directory: {subject}/eeg")

        for task, run in EXPECTED_RUNS:
            for suffix in EXPECTED_SUFFIXES:
                fname = expected_file(subject, task, run, suffix)
                path = eeg_dir / fname
                present = path.is_file()
                size = path.stat().st_size if present else 0
                rows.append(
                    AuditRow(
                        subject=subject,
                        task=task,
                        run=run,
                        suffix=suffix,
                        path=str(path),
                        present=present,
                        size_bytes=size,
                    )
                )
                if not present:
                    errors.append(f"Missing file: {subject}/eeg/{fname}")
                elif size == 0:
                    errors.append(f"Empty file: {subject}/eeg/{fname}")

    return errors, rows


def write_csv(rows: list[AuditRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "task",
                "run",
                "suffix",
                "path",
                "present",
                "size_bytes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Won2022_BIDS file completeness.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    errors, rows = audit_dataset(args.dataset_root)
    write_csv(rows, args.out_csv)

    n_expected = len(rows)
    n_present = sum(row.present for row in rows)
    n_empty = sum(row.present and row.size_bytes == 0 for row in rows)

    print(f"Dataset root: {args.dataset_root}")
    print(f"Expected run-sidecar files: {n_expected}")
    print(f"Present files: {n_present}")
    print(f"Empty files: {n_empty}")
    print(f"Audit CSV: {args.out_csv}")

    if errors:
        print("\nFAIL")
        for err in errors[:50]:
            print(f"- {err}")
        if len(errors) > 50:
            print(f"- ... {len(errors) - 50} more errors")
        return 1

    print("\nPASS: all expected dataset files are present and non-empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
