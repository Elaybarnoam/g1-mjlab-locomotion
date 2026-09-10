"""Command-line entry point for supported G1 mjlab workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, load_ppo_profile, load_reward_profile


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="g1-mjlab")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", required=True, type=Path)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--output", required=True, type=Path)
    qualify = commands.add_parser("qualify-controller")
    qualify.add_argument("--config", required=True, type=Path)
    qualify.add_argument("--output", required=True, type=Path)
    native = commands.add_parser("evaluate-native")
    native.add_argument("--run", required=True, type=Path)
    native.add_argument("--scenarios", required=True, type=Path)
    native.add_argument("--output", required=True, type=Path)
    record = commands.add_parser("record-native")
    record.add_argument("--run", required=True, type=Path)
    record.add_argument("--scenarios", required=True, type=Path)
    record.add_argument("--output", required=True, type=Path)
    record.add_argument("--trial-id", type=int, default=0)
    record.add_argument("--duration-seconds", type=float, default=15.0)
    record.add_argument("--fps", type=int, default=30)
    record.add_argument("--width", type=int, default=960)
    record.add_argument("--height", type=int, default=720)
    play = commands.add_parser("play-native")
    play.add_argument("--run", required=True, type=Path)
    play.add_argument("--scenarios", required=True, type=Path)
    play.add_argument("--trial-id", type=int, default=0)
    parity = commands.add_parser("check-transfer-parity")
    parity.add_argument("--config", required=True, type=Path)
    parity.add_argument("--run", required=True, type=Path)
    parity.add_argument("--output", required=True, type=Path)
    select = commands.add_parser("select-checkpoint")
    select.add_argument("--run", required=True, type=Path)
    freeze = commands.add_parser("freeze-selected")
    freeze.add_argument("--config", required=True, type=Path)
    freeze.add_argument("--run", required=True, type=Path)
    campaign = commands.add_parser("evaluate-checkpoints")
    campaign.add_argument("--config", required=True, type=Path)
    campaign.add_argument("--checkpoints", required=True, type=Path, nargs="+")
    campaign.add_argument("--run", required=True, type=Path)
    campaign.add_argument("--trials", type=int, default=20)
    campaign.add_argument("--horizon-seconds", type=float, default=10.0)
    campaign.add_argument("--attempts", type=int, default=3)
    qualify_final = commands.add_parser("qualify-final")
    qualify_final.add_argument("--run", required=True, type=Path)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--reward-profile", type=Path)
    train.add_argument("--ppo-profile", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--trials", type=int, default=20)
    evaluate.add_argument("--horizon-seconds", type=float, default=60.0)
    evaluate.add_argument("--seed", type=int, default=10042)
    evaluate.add_argument("--min-height-m", type=float, default=0.60)
    evaluate.add_argument("--phase", choices=("development", "final"), default="development")
    diagnose = commands.add_parser("diagnose-standing")
    diagnose.add_argument("--config", required=True, type=Path)
    diagnose.add_argument("--checkpoint", required=True, type=Path)
    diagnose.add_argument("--onnx", type=Path)
    diagnose.add_argument("--output", required=True, type=Path)
    diagnose.add_argument("--trials", type=int, default=4)
    diagnose.add_argument("--horizon-seconds", type=float, default=3.0)
    checkpoints = commands.add_parser("diagnose-checkpoints")
    checkpoints.add_argument("--config", required=True, type=Path)
    checkpoints.add_argument("--checkpoints", required=True, type=Path, nargs="+")
    checkpoints.add_argument("--output", required=True, type=Path)
    checkpoints.add_argument("--horizon-seconds", type=float, default=3.0)
    report = commands.add_parser("report")
    report.add_argument("--run", required=True, type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--final", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "doctor":
        from .runtime import doctor

        print(json.dumps(doctor(args.output), indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        from .runtime import git_source_metadata, train

        config = load_config(args.config)
        reward_profile = load_reward_profile(args.reward_profile) if args.reward_profile else None
        ppo_profile = load_ppo_profile(args.ppo_profile) if args.ppo_profile else None
        project_root = Path(__file__).resolve().parents[2]
        train(
            config,
            args.output,
            git_source_metadata(project_root),
            reward_profile=reward_profile,
            ppo_profile=ppo_profile,
            resume=args.resume,
        )
        return 0
    if args.command == "qualify-controller":
        from .qualification import qualify_controller

        print(json.dumps(qualify_controller(load_config(args.config), args.output), indent=2))
        return 0
    if args.command == "evaluate-native":
        from .deployment import evaluate_native

        print(json.dumps(evaluate_native(args.run, args.scenarios, args.output), indent=2))
        return 0
    if args.command == "record-native":
        from .native_media import record_native_video

        print(
            json.dumps(
                record_native_video(
                    args.run,
                    args.scenarios,
                    args.output,
                    trial_id=args.trial_id,
                    duration_s=args.duration_seconds,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "play-native":
        from .native_media import play_native

        play_native(args.run, args.scenarios, trial_id=args.trial_id)
        return 0
    if args.command == "check-transfer-parity":
        from .deployment import check_transfer_parity

        print(
            json.dumps(
                check_transfer_parity(load_config(args.config), args.run, args.output), indent=2
            )
        )
        return 0
    if args.command == "select-checkpoint":
        from .selection import select_checkpoint

        print(json.dumps(select_checkpoint(args.run), indent=2))
        return 0
    if args.command == "freeze-selected":
        from .deployment import freeze_selected_policy

        print(json.dumps(freeze_selected_policy(load_config(args.config), args.run), indent=2))
        return 0
    if args.command == "evaluate-checkpoints":
        from .campaign import evaluate_checkpoints

        result = evaluate_checkpoints(
            config=args.config,
            checkpoints=args.checkpoints,
            run=args.run,
            trials=args.trials,
            horizon_s=args.horizon_seconds,
            retries=args.attempts,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "qualify-final":
        from .deployment import qualify_final_bundle

        print(json.dumps(qualify_final_bundle(args.run), indent=2))
        return 0
    if args.command == "evaluate":
        from .evaluation import StandingCriteria
        from .runtime import evaluate

        config = load_config(args.config)
        result = evaluate(
            config,
            args.checkpoint,
            args.output,
            trials=args.trials,
            horizon_s=args.horizon_seconds,
            seed=args.seed,
            criteria=StandingCriteria(min_height_m=args.min_height_m),
            phase=args.phase,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "diagnose-standing":
        from .diagnostics import diagnose_standing

        config = load_config(args.config)
        result = diagnose_standing(
            config,
            args.checkpoint,
            args.output,
            onnx_path=args.onnx,
            trials=args.trials,
            horizon_s=args.horizon_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "diagnose-checkpoints":
        from .diagnostics import diagnose_checkpoints

        config = load_config(args.config)
        result = diagnose_checkpoints(
            config,
            args.checkpoints,
            args.output,
            horizon_s=args.horizon_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        from .reporting import render_report

        print(render_report(args.run, args.output, final=args.final))
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
