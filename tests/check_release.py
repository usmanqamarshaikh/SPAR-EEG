from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".m", ".md", ".txt", ".yml", ".yaml", ".toml", ".cff"}
FORBIDDEN = (
    "C:\\Users\\" + "zrf1847",
    "OneDrive - " + "AUT University",
    "J4_" + "codex",
)


def main() -> None:
    findings: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or any(part in {".venv", ".idea", ".git"} for part in path.parts):
            continue
        if path.suffix.lower() in {".set", ".fdt", ".xdf", ".mat", ".h5", ".hdf5"}:
            findings.append(f"Dataset-like file is tracked: {path.relative_to(REPO_ROOT)}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in FORBIDDEN:
                if token in text:
                    findings.append(f"Private path token in {path.relative_to(REPO_ROOT)}: {token}")

    if findings:
        raise RuntimeError("\n".join(findings))
    print("Release audit passed: no datasets or private workstation paths found.")


if __name__ == "__main__":
    main()
