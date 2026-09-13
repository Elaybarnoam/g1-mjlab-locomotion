from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.walking_v2_campaign import load_walking_v2_campaign


def test_checked_in_walking_v2_campaign_is_hash_bound() -> None:
    root = Path(__file__).parents[2]
    campaign = load_walking_v2_campaign(root / "configs/walking-v2/campaign-v2.json")

    assert campaign.training_seeds == (42, 43, 44)
    assert campaign.environment_candidates == (64, 128, 256)
    assert campaign.rollout_steps == 24
    assert campaign.maximum_acquisition_updates == 4000


def test_campaign_rejects_artifact_drift(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = root / "configs/walking-v2/campaign-v2.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["ppo"]["sha256"] = "0" * 64
    target = tmp_path / "campaign.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError), match="ppo"):
        load_walking_v2_campaign(target)
