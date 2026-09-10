"""Fail when public repository policy or curated evidence is violated."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_FILE_BYTES = 10_000_000
PROHIBITED_PREFIXES = (
    ".plans/",
    ".scratch/",
    ".runtime/",
    ".venv",
    "artifacts/",
    "runs/",
    "logs/",
    "checkpoints/",
    "src/humanoid_rl/",
)
PROHIBITED_SUFFIXES = (".mjb", ".onnx", ".pt", ".pth", ".pyc")
PRIVATE_PATTERNS = (
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/Users/[^/]+/"),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/root/"),
)
TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode() for value in result.stdout.split(b"\0") if value]


def check_git_history() -> None:
    """Reject prohibited names and oversized blobs anywhere in reachable history."""
    result = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    for line in result.stdout.splitlines():
        object_id, separator, name = line.partition(" ")
        if not separator:
            continue
        normalized = name.replace("\\", "/")
        if normalized.startswith(PROHIBITED_PREFIXES) or normalized.endswith(PROHIBITED_SUFFIXES):
            raise SystemExit(f"prohibited historical path: {normalized}")
        size = subprocess.run(
            ["git", "cat-file", "-s", object_id],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="ascii",
        )
        if int(size.stdout) > MAX_GIT_FILE_BYTES:
            raise SystemExit(f"historical blob exceeds {MAX_GIT_FILE_BYTES} bytes: {normalized}")


def check_media() -> None:
    media_root = ROOT / "docs" / "assets" / "standing-v1"
    manifest = json.loads((media_root / "media-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = media_root / name
        if path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"media size mismatch: {name}")
        if sha256(path) != expected["sha256"]:
            raise SystemExit(f"media hash mismatch: {name}")


def check_markdown_links() -> None:
    pattern = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
    for document in ROOT.rglob("*.md"):
        if any(
            part.startswith(".") and part != ".github" for part in document.relative_to(ROOT).parts
        ):
            continue
        for destination in pattern.findall(document.read_text(encoding="utf-8")):
            clean = destination.strip("<>").split("#", 1)[0]
            if not clean or "://" in clean or clean.startswith("mailto:"):
                continue
            target = (document.parent / clean).resolve()
            try:
                target.relative_to(ROOT)
            except ValueError as error:
                raise SystemExit(f"link escapes repository: {document}: {destination}") from error
            if not target.exists():
                raise SystemExit(f"broken local link: {document}: {destination}")


def main() -> int:
    files = tracked_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(PROHIBITED_PREFIXES) or relative.endswith(PROHIBITED_SUFFIXES):
            raise SystemExit(f"prohibited tracked path: {relative}")
        if path.stat().st_size > MAX_GIT_FILE_BYTES:
            raise SystemExit(f"tracked file exceeds {MAX_GIT_FILE_BYTES} bytes: {relative}")
        # This policy implementation necessarily contains the patterns it enforces.
        if relative != "scripts/check_repository.py" and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in PRIVATE_PATTERNS:
                if pattern.search(text):
                    raise SystemExit(f"private absolute path in {relative}: {pattern.pattern}")
    check_media()
    check_markdown_links()
    check_git_history()
    print(f"repository policy passed for {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
