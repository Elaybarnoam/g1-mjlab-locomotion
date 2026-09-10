from pathlib import Path
from types import SimpleNamespace


def test_native_crash_is_retried_and_preserved(tmp_path: Path, monkeypatch) -> None:
    from g1_mjlab.campaign import evaluate_checkpoints

    checkpoint = tmp_path / "model_10.pt"
    checkpoint.touch()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        if len(calls) == 1:
            return SimpleNamespace(returncode=-11)
        (output / "summary.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", run)
    result = evaluate_checkpoints(
        config=tmp_path / "config.json",
        checkpoints=[checkpoint],
        run=tmp_path / "run",
        trials=20,
        horizon_s=10.0,
        retries=2,
    )
    assert len(calls) == 2
    assert result["jobs"][0]["attempts"][0]["return_code"] == -11
    assert result["jobs"][0]["attempts"][1]["return_code"] == 0
