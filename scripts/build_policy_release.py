"""Build deterministic standing-v1 inference and resume artifacts for a GitHub release."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from g1_mjlab.artifacts import sha256_file
from g1_mjlab.policy_distribution import validate_policy

ARCHIVE_NAME = "standing-v1-inference.zip"
CHECKPOINT_NAME = "model_999.pt"
FIXED_TIMESTAMP = (2026, 9, 9, 0, 0, 0)


def _write_archive(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def build(run: Path, output_dir: Path, project: Path) -> dict[str, object]:
    """Assemble only the files required for deterministic inference and trusted resume."""
    run = run.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME
    checkpoint_output = output_dir / CHECKPOINT_NAME
    checksums_path = output_dir / "SHA256SUMS.txt"
    for output in (archive_path, checkpoint_output, checksums_path):
        if output.exists():
            raise FileExistsError(f"release output already exists: {output}")

    bundle = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    if bundle.get("final_test_qualified") is not True:
        raise ValueError("refusing to publish a policy that did not pass final qualification")
    checkpoint = run / "checkpoints" / bundle["checkpoint"]
    if checkpoint.name != CHECKPOINT_NAME:
        raise ValueError(f"expected selected checkpoint {CHECKPOINT_NAME}, got {checkpoint.name}")
    if sha256_file(checkpoint) != bundle["checkpoint_sha256"]:
        raise ValueError("selected checkpoint integrity mismatch")

    with tempfile.TemporaryDirectory(prefix="standing-v1-release-") as temporary:
        staging = Path(temporary)
        (staging / "checkpoints").mkdir()
        (staging / "LICENSES").mkdir()
        shutil.copy2(run / "contract.json", staging / "contract.json")
        shutil.copy2(run / "policy-bundle.json", staging / "policy-bundle.json")
        shutil.copy2(run / "model.mjb", staging / "model.mjb")
        shutil.copy2(run / "checkpoints" / "policy.onnx", staging / "checkpoints/policy.onnx")
        shutil.copy2(run / "evaluation/final-100x60/summary.json", staging / "scenarios.json")
        shutil.copy2(project / "THIRD_PARTY_NOTICES.md", staging / "THIRD_PARTY_NOTICES.md")
        shutil.copy2(project / "LICENSE", staging / "LICENSE")
        shutil.copy2(
            project / "release/UNITREE_G1_LICENSE.txt",
            staging / "LICENSES/UNITREE_G1_LICENSE.txt",
        )
        (staging / "README.md").write_text(
            """# G1 standing-v1 inference bundle

This is the qualified deterministic Unitree G1 standing policy for
https://github.com/Elaybarnoam/g1-mjlab-locomotion.

Install the repository at tag `v0.2.0`, then run:

```console
uv sync --extra native
uv run g1-mjlab play-policy --policy /path/to/this/directory
```

The ONNX graph includes the actor observation normalizer and deterministic actor mean. This bundle
is for simulation research only. It is not approved or safe for deployment on physical hardware.
See `THIRD_PARTY_NOTICES.md` and `LICENSES/UNITREE_G1_LICENSE.txt` before redistribution.
""",
            encoding="utf-8",
        )
        validate_policy(staging)
        _write_archive(staging, archive_path)

    shutil.copy2(checkpoint, checkpoint_output)
    checksums = {
        ARCHIVE_NAME: sha256_file(archive_path),
        CHECKPOINT_NAME: sha256_file(checkpoint_output),
    }
    checksums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="ascii"
    )
    return {
        "archive": str(archive_path),
        "checkpoint": str(checkpoint_output),
        "checksums": checksums,
        "checksums_file": str(checksums_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    print(json.dumps(build(args.run, args.output, project), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
