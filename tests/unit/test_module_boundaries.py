from pathlib import Path

import pytest

from g1_mjlab.cli import parser
from g1_mjlab.config import load_config
from g1_mjlab.deployment import load_bundle
from g1_mjlab.deployment import play_native as legacy_play_native
from g1_mjlab.deployment import record_native_video as legacy_record_native_video
from g1_mjlab.diagnostics import diagnose_standing
from g1_mjlab.native_media import play_native, record_native_video
from g1_mjlab.runtime import diagnose_checkpoints as legacy_diagnose_checkpoints
from g1_mjlab.runtime import diagnose_standing as legacy_diagnose_standing


def test_compatibility_entry_points_remain_importable() -> None:
    assert callable(legacy_play_native)
    assert callable(legacy_record_native_video)
    assert callable(legacy_diagnose_checkpoints)
    assert callable(legacy_diagnose_standing)
    assert callable(play_native)


def test_cli_routes_to_new_module_commands() -> None:
    assert parser().parse_args(["install-policy", "standing-v1"]).command == "install-policy"
    assert (
        parser().parse_args(["play-policy", "--policy", "policies/standing-v1"]).command
        == "play-policy"
    )
    assert (
        parser()
        .parse_args(
            ["record-native", "--run", "run", "--scenarios", "scenarios", "--output", "out.mp4"]
        )
        .command
        == "record-native"
    )
    assert (
        parser()
        .parse_args(
            ["diagnose-standing", "--config", "config", "--checkpoint", "model", "--output", "out"]
        )
        .command
        == "diagnose-standing"
    )


def test_native_recording_rejects_invalid_dimensions_before_file_access() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        record_native_video(Path("missing"), Path("missing"), Path("output.mp4"), width=0)


def test_diagnostics_reject_unsafe_bounds_before_simulator_start() -> None:
    config = load_config(Path("configs/standing-v1/smoke.json"))
    with pytest.raises(ValueError, match="1..4 trials"):
        diagnose_standing(config, Path("model.pt"), Path("output"), trials=5)


@pytest.mark.parametrize(
    ("contract_version", "bundle_version", "message"),
    [(3, 1, "contract"), (2, 2, "bundle")],
)
def test_bundle_loader_rejects_unknown_schema_before_artifact_access(
    tmp_path: Path, contract_version: int, bundle_version: int, message: str
) -> None:
    (tmp_path / "contract.json").write_text(
        f'{{"schema_version": {contract_version}}}', encoding="utf-8"
    )
    (tmp_path / "policy-bundle.json").write_text(
        f'{{"schema_version": {bundle_version}}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        load_bundle(tmp_path)
