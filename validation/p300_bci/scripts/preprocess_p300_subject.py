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
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed_fp12"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "outputs" / "preprocessed_fp12" / "manifests"
ONE_RUN_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess_fp12_one_run.py"


@dataclass(frozen=True)
class P300Run:
    subject: str
    task: str
    run: int
    set_path: Path

    @property
    def stem(self) -> str:
        return f"{self.subject}_task-{self.task}_run-{self.run}_fp12"


def discover_p300_runs(dataset_root: Path, subject: str) -> list[P300Run]:
    eeg_dir = dataset_root / subject / "eeg"
    if not eeg_dir.is_dir():
        raise FileNotFoundError(eeg_dir)

    pattern = re.compile(rf"^{re.escape(subject)}_task-(P300(?:train|test)run\d+)_run-(\d+)_eeg\.set$")
    runs: list[P300Run] = []
    for set_path in sorted(eeg_dir.glob("*P300*_eeg.set")):
        match = pattern.match(set_path.name)
        if not match:
            continue
        task = match.group(1)
        run = int(match.group(2))
        runs.append(P300Run(subject=subject, task=task, run=run, set_path=set_path))

    return sorted(runs, key=lambda item: item.run)


def output_paths(out_dir: Path, run: P300Run, max_seconds: float | None) -> tuple[Path, Path]:
    stem = run.stem
    if max_seconds is not None:
        stem += f"_first{max_seconds:g}s"
    return out_dir / f"{stem}_wide.h5", out_dir / f"{stem}_branches.h5"


def run_one(
    py_exe: Path,
    dataset_root: Path,
    out_dir: Path,
    run: P300Run,
    overwrite: bool,
    skip_matlab: bool,
    max_seconds: float | None,
    notch_freqs: list[float],
    notch_widths: float,
    matlab: str,
    stream_output: bool,
) -> tuple[int, str, float]:
    cmd = [
        str(py_exe),
        str(ONE_RUN_SCRIPT),
        "--dataset-root",
        str(dataset_root),
        "--subject",
        run.subject,
        "--task",
        run.task,
        "--run",
        str(run.run),
        "--out-dir",
        str(out_dir),
        "--notch-widths",
        str(notch_widths),
        "--matlab",
        matlab,
    ]
    if notch_freqs:
        cmd.append("--notch-freqs")
        cmd.extend(str(freq) for freq in notch_freqs)
    else:
        cmd.append("--notch-freqs")
    if overwrite:
        cmd.append("--overwrite")
    if skip_matlab:
        cmd.append("--skip-matlab")
    if max_seconds is not None:
        cmd.extend(["--max-seconds", str(max_seconds)])

    t0 = time.perf_counter()
    if stream_output:
        proc = subprocess.run(cmd, text=True)
        message = "output streamed to console"
    else:
        proc = subprocess.run(cmd, text=True, capture_output=True)
        message = (proc.stdout + "\n" + proc.stderr).strip()
    elapsed = time.perf_counter() - t0
    return proc.returncode, message, elapsed


def write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "task",
        "run",
        "set_path",
        "wide_h5",
        "branches_h5",
        "status",
        "elapsed_sec",
        "message_tail",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess all P300 FP1/FP2 runs for one subject.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--notch-freqs", type=float, nargs="*", default=[50.0])
    parser.add_argument("--notch-widths", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-matlab", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing wide/branch files and still write manifest rows.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--stream-output", action="store_true", help="Show per-run Python/MATLAB output live.")
    args = parser.parse_args()

    runs = discover_p300_runs(args.dataset_root, args.subject)
    if args.limit is not None:
        runs = runs[: args.limit]

    print(f"Subject: {args.subject}")
    print(f"P300 runs discovered: {len(runs)}")
    for item in runs:
        wide_h5, branches_h5 = output_paths(args.out_dir, item, args.max_seconds)
        print(f"  run-{item.run:02d} | {item.task:14s} | {item.set_path.name}")
        print(f"      wide:     {wide_h5}")
        print(f"      branches: {branches_h5}")

    if args.dry_run:
        print("\nDry run only. No preprocessing launched.")
        return 0

    rows: list[dict[str, object]] = []
    for idx, item in enumerate(runs, start=1):
        wide_h5, branches_h5 = output_paths(args.out_dir, item, args.max_seconds)
        print(f"\n[{idx}/{len(runs)}] Preprocessing {item.subject} {item.task} run-{item.run}")
        existing_ok = wide_h5.is_file() and (args.skip_matlab or branches_h5.is_file())
        if args.skip_existing and existing_ok and not args.overwrite:
            code = 0
            message = "existing outputs reused"
            elapsed = 0.0
        else:
            code, message, elapsed = run_one(
                py_exe=args.python,
                dataset_root=args.dataset_root,
                out_dir=args.out_dir,
                run=item,
                overwrite=args.overwrite,
                skip_matlab=args.skip_matlab,
                max_seconds=args.max_seconds,
                notch_freqs=list(args.notch_freqs),
                notch_widths=args.notch_widths,
                matlab=args.matlab,
                stream_output=args.stream_output,
            )
        status = "ok" if code == 0 else "failed"
        print(f"  status={status} elapsed={elapsed:.1f} s")
        if message:
            print("\n".join("  " + line for line in message.splitlines()[-8:]))

        rows.append(
            {
                "subject": item.subject,
                "task": item.task,
                "run": item.run,
                "set_path": str(item.set_path),
                "wide_h5": str(wide_h5),
                "branches_h5": str(branches_h5),
                "status": status,
                "elapsed_sec": f"{elapsed:.3f}",
                "message_tail": "\n".join(message.splitlines()[-6:]),
            }
        )

        if code != 0 and not args.keep_going:
            manifest_path = args.manifest_dir / f"{args.subject}_p300_preprocessing_manifest.csv"
            write_manifest(manifest_path, rows)
            print(f"\nManifest written: {manifest_path}")
            return code

    manifest_path = args.manifest_dir / f"{args.subject}_p300_preprocessing_manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"\nManifest written: {manifest_path}")
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        print(f"Completed with failures: {len(failed)}")
        return 1
    print("All requested P300 runs preprocessed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
