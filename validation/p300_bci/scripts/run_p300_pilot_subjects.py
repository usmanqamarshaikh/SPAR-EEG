from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
DEFAULT_PREPROCESS_DIR = PROJECT_ROOT / "outputs" / "preprocessed_fp12"
DEFAULT_MANIFEST_DIR = DEFAULT_PREPROCESS_DIR / "manifests"
DEFAULT_DECODER_DIR = PROJECT_ROOT / "outputs" / "swlda_decoder"
DEFAULT_RUNLOG = PROJECT_ROOT / "outputs" / "pilot_runs" / "p300_pilot10_runlog.csv"

PREPROCESS_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess_p300_subject.py"
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_p300_subject_preprocessing.py"
DECODE_SCRIPT = PROJECT_ROOT / "scripts" / "decode_p300_subject_swlda.py"


@dataclass(frozen=True)
class StepResult:
    status: str
    elapsed_sec: float
    return_code: int


def natural_subject_key(name: str) -> tuple[int, str]:
    match = re.search(r"sub-(\d+)$", name)
    if not match:
        return (10**9, name)
    return (int(match.group(1)), name)


def discover_subjects(dataset_root: Path) -> list[str]:
    return sorted(
        [
            item.name
            for item in dataset_root.iterdir()
            if item.is_dir() and re.match(r"^sub-\d+$", item.name)
        ],
        key=natural_subject_key,
    )


def run_step(label: str, cmd: list[str], dry_run: bool) -> StepResult:
    print(f"\n--- {label} ---")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    if dry_run:
        return StepResult(status="dry_run", elapsed_sec=0.0, return_code=0)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else "failed"
    print(f"--- {label} finished: {status} in {elapsed:.1f} s ---")
    return StepResult(status=status, elapsed_sec=elapsed, return_code=proc.returncode)


def append_runlog(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "step",
        "status",
        "elapsed_sec",
        "return_code",
        "manifest",
        "decoder_summary",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def subject_manifest(manifest_dir: Path, subject: str) -> Path:
    return manifest_dir / f"{subject}_p300_preprocessing_manifest.csv"


def subject_decoder_summary(decoder_dir: Path, subject: str) -> Path:
    return decoder_dir / f"{subject}_swlda_summary.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an FP1/FP2 P300 pilot over multiple subjects: preprocessing, audit, "
            "and SWLDA decoding with live console output."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subjects", nargs="*", default=None, help="Explicit subject IDs, e.g. sub-001 sub-002.")
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--all-subjects", action="store_true", help="Process every discovered sub-XXX folder.")
    parser.add_argument("--start-subject", type=int, default=1, help="First subject number when --subjects is omitted.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--preprocess-dir", type=Path, default=DEFAULT_PREPROCESS_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--decoder-dir", type=Path, default=DEFAULT_DECODER_DIR)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--branches", nargs="+", default=["baseline", "eog", "emg_eog", "full"])
    parser.add_argument("--notch-freqs", type=float, nargs="*", default=[50.0])
    parser.add_argument("--notch-widths", type=float, default=2.0)
    parser.add_argument("--overwrite-preprocess", action="store_true")
    parser.add_argument("--skip-existing-preprocess", action="store_true", help="Reuse existing wide/branch files and continue.")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    elif args.all_subjects:
        subjects = discover_subjects(args.dataset_root)
    else:
        all_subjects = discover_subjects(args.dataset_root)
        wanted = {f"sub-{idx:03d}" for idx in range(args.start_subject, args.start_subject + args.n_subjects)}
        subjects = [sub for sub in all_subjects if sub in wanted]

    if not subjects:
        raise ValueError("No subjects selected.")

    print("P300 pilot subjects:")
    for subject in subjects:
        print(f"  {subject}")
    print(f"Runlog: {args.runlog}")

    all_ok = True
    for si, subject in enumerate(subjects, start=1):
        print(f"\n==================== SUBJECT {si}/{len(subjects)}: {subject} ====================")
        manifest = subject_manifest(args.manifest_dir, subject)
        summary = subject_decoder_summary(args.decoder_dir, subject)
        rows: list[dict[str, object]] = []

        if not args.skip_preprocess:
            preprocess_cmd = [
                str(args.python),
                str(PREPROCESS_SCRIPT),
                "--dataset-root",
                str(args.dataset_root),
                "--subject",
                subject,
                "--out-dir",
                str(args.preprocess_dir),
                "--manifest-dir",
                str(args.manifest_dir),
                "--matlab",
                args.matlab,
                "--notch-widths",
                str(args.notch_widths),
                "--stream-output",
            ]
            if args.notch_freqs:
                preprocess_cmd.append("--notch-freqs")
                preprocess_cmd.extend(str(freq) for freq in args.notch_freqs)
            else:
                preprocess_cmd.append("--notch-freqs")
            if args.overwrite_preprocess:
                preprocess_cmd.append("--overwrite")
            if args.skip_existing_preprocess:
                preprocess_cmd.append("--skip-existing")

            result = run_step(f"{subject} preprocess", preprocess_cmd, args.dry_run)
            rows.append(
                {
                    "subject": subject,
                    "step": "preprocess",
                    "status": result.status,
                    "elapsed_sec": f"{result.elapsed_sec:.3f}",
                    "return_code": result.return_code,
                    "manifest": str(manifest),
                    "decoder_summary": str(summary),
                }
            )
            append_runlog(args.runlog, rows[-1:])
            if result.return_code != 0:
                all_ok = False
                if not args.keep_going:
                    return result.return_code

        if not args.skip_audit:
            audit_cmd = [
                str(args.python),
                str(AUDIT_SCRIPT),
                "--manifest",
                str(manifest),
                "--out-csv",
                str(args.manifest_dir / f"{subject}_p300_preprocessing_audit.csv"),
            ]
            result = run_step(f"{subject} audit", audit_cmd, args.dry_run)
            rows.append(
                {
                    "subject": subject,
                    "step": "audit",
                    "status": result.status,
                    "elapsed_sec": f"{result.elapsed_sec:.3f}",
                    "return_code": result.return_code,
                    "manifest": str(manifest),
                    "decoder_summary": str(summary),
                }
            )
            append_runlog(args.runlog, rows[-1:])
            if result.return_code != 0:
                all_ok = False
                if not args.keep_going:
                    return result.return_code

        if not args.skip_decode:
            decode_cmd = [
                str(args.python),
                str(DECODE_SCRIPT),
                "--manifest",
                str(manifest),
                "--out-dir",
                str(args.decoder_dir),
                "--branches",
                *args.branches,
            ]
            result = run_step(f"{subject} decode", decode_cmd, args.dry_run)
            rows.append(
                {
                    "subject": subject,
                    "step": "decode",
                    "status": result.status,
                    "elapsed_sec": f"{result.elapsed_sec:.3f}",
                    "return_code": result.return_code,
                    "manifest": str(manifest),
                    "decoder_summary": str(summary),
                }
            )
            append_runlog(args.runlog, rows[-1:])
            if result.return_code != 0:
                all_ok = False
                if not args.keep_going:
                    return result.return_code

    print("\nPilot runner finished.")
    print(f"Runlog: {args.runlog}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
