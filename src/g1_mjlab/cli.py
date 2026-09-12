"""Command-line entry point for supported G1 mjlab workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    load_config,
    load_ppo_profile,
    load_reward_profile,
    load_walking_training_profile,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="g1-mjlab")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", required=True, type=Path)
    prepare_motion = commands.add_parser("prepare-motion")
    prepare_motion.add_argument("--source-csv", required=True, type=Path)
    prepare_motion.add_argument("--model-xml", required=True, type=Path)
    prepare_motion.add_argument("--output", required=True, type=Path)
    prepare_motion.add_argument("--audit", required=True, type=Path)
    prepare_motion.add_argument("--source-fps", type=float, default=120.0)
    prepare_motion.add_argument("--output-fps", type=float, default=50.0)
    prepare_motion.add_argument("--first-frame", type=int, default=409)
    prepare_motion.add_argument("--last-frame", type=int, default=537)
    preview_motion = commands.add_parser("preview-motion")
    preview_motion.add_argument("--reference", required=True, type=Path)
    preview_motion.add_argument("--model-xml", required=True, type=Path)
    preview_motion.add_argument("--output", required=True, type=Path)
    probe_walking = commands.add_parser("probe-walking")
    probe_walking.add_argument("--config", required=True, type=Path)
    probe_walking.add_argument("--output", required=True, type=Path)
    probe_walking.add_argument("--steps", type=int, default=100)
    probe_walking.add_argument("--num-envs", type=int)
    probe_walking.add_argument("--zero-actions", action="store_true")
    select_parallel = commands.add_parser("select-walking-parallelism")
    select_parallel.add_argument("--probes", required=True, type=Path, nargs="+")
    select_parallel.add_argument("--repeats", required=True, type=int)
    select_parallel.add_argument("--minimum-free-ratio", type=float, default=0.2)
    select_parallel.add_argument("--output", required=True, type=Path)
    install = commands.add_parser("install-policy")
    install.add_argument("name", choices=("standing-v1",))
    install.add_argument("--output", type=Path)
    installed_play = commands.add_parser("play-policy")
    installed_play.add_argument("--policy", required=True, type=Path)
    installed_play.add_argument("--trial-id", type=int, default=0)
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
    train.add_argument("--walking-profile", type=Path)
    continuation = train.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path)
    continuation.add_argument("--initialize-actor", type=Path)
    continuation.add_argument("--fine-tune", type=Path)
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
    evaluate_walking = commands.add_parser("evaluate-walking")
    evaluate_walking.add_argument("--config", required=True, type=Path)
    evaluate_walking.add_argument("--checkpoint", required=True, type=Path)
    evaluate_walking.add_argument("--schedule", type=Path)
    evaluate_walking.add_argument("--scenarios", type=Path)
    evaluate_walking.add_argument("--criteria", type=Path)
    evaluate_walking.add_argument("--metric-schema", type=int, choices=(1, 2), default=1)
    evaluate_walking.add_argument("--output", required=True, type=Path)
    evaluate_walking.add_argument("--trials", type=int, default=16)
    evaluate_walking.add_argument("--seed", type=int, default=10042)
    evaluate_walking.add_argument("--video", action="store_true")
    evaluate_walking.add_argument(
        "--initialization",
        choices=("standing", "reference", "reference-fixed"),
        default="standing",
    )
    diagnose_walking = commands.add_parser("diagnose-walking")
    diagnose_walking.add_argument("--config", required=True, type=Path)
    diagnose_walking.add_argument("--checkpoint", required=True, type=Path)
    diagnose_walking.add_argument("--scenarios", required=True, type=Path)
    diagnose_walking.add_argument("--output", required=True, type=Path)
    diagnose_walking.add_argument(
        "--criteria", type=Path, default=Path("configs/walking-v1/evaluation-v2.json")
    )
    diagnose_walking.add_argument("--physics-trace", action="store_true")
    diagnose_walking.add_argument("--video", action="store_true")
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
    if args.command == "prepare-motion":
        from .motion.preparation import prepare_reference

        result = prepare_reference(
            source_csv=args.source_csv,
            model_xml=args.model_xml,
            output_npz=args.output,
            output_audit=args.audit,
            source_fps=args.source_fps,
            output_fps=args.output_fps,
            first_frame=args.first_frame,
            last_frame=args.last_frame,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "preview-motion":
        from .motion.preparation import render_reference_preview

        print(
            json.dumps(
                render_reference_preview(
                    reference_npz=args.reference,
                    model_xml=args.model_xml,
                    output_mp4=args.output,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "probe-walking":
        from .walking_diagnostics import run_walking_probe

        overrides = {"num_envs": args.num_envs} if args.num_envs is not None else None
        probe_config = load_config(args.config, overrides)
        print(
            json.dumps(
                run_walking_probe(
                    probe_config,
                    args.output,
                    steps=args.steps,
                    reference_actions=not args.zero_actions,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "select-walking-parallelism":
        from .walking_diagnostics import select_parallelism

        probes = [json.loads(path.read_text(encoding="utf-8")) for path in args.probes]
        result = select_parallelism(
            probes,
            repeats=args.repeats,
            minimum_free_ratio=args.minimum_free_ratio,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "install-policy":
        from .policy_distribution import install_policy

        output = args.output or Path("policies") / args.name
        print(json.dumps(install_policy(args.name, output), indent=2, sort_keys=True))
        return 0
    if args.command == "play-policy":
        from .native_media import play_native

        play_native(args.policy, args.policy / "scenarios.json", trial_id=args.trial_id)
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
        walking_profile = (
            load_walking_training_profile(args.walking_profile) if args.walking_profile else None
        )
        project_root = Path(__file__).resolve().parents[2]
        train(
            config,
            args.output,
            git_source_metadata(project_root),
            reward_profile=reward_profile,
            ppo_profile=ppo_profile,
            resume=args.resume,
            initialize_actor=args.initialize_actor,
            fine_tune=args.fine_tune,
            walking_profile=walking_profile,
        )
        return 0
    if args.command == "qualify-controller":
        from .qualification import qualify_controller

        print(json.dumps(qualify_controller(load_config(args.config), args.output), indent=2))
        return 0
    if args.command == "evaluate-walking":
        from .walking_runtime import diagnose_walking, evaluate_walking

        if args.metric_schema == 2:
            if args.scenarios is None or args.criteria is None:
                raise ValueError("metric schema 2 requires --scenarios and --criteria")
            print(
                json.dumps(
                    diagnose_walking(
                        load_config(args.config),
                        args.checkpoint,
                        args.scenarios,
                        args.output,
                        criteria_path=args.criteria,
                        physics_trace=True,
                        video=args.video,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.schedule is None:
            raise ValueError("metric schema 1 requires --schedule")

        print(
            json.dumps(
                evaluate_walking(
                    load_config(args.config),
                    args.checkpoint,
                    args.schedule,
                    args.output,
                    trials=args.trials,
                    seed=args.seed,
                    video=args.video,
                    initialization=args.initialization,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "diagnose-walking":
        from .walking_runtime import diagnose_walking

        print(
            json.dumps(
                diagnose_walking(
                    load_config(args.config),
                    args.checkpoint,
                    args.scenarios,
                    args.output,
                    criteria_path=args.criteria,
                    physics_trace=args.physics_trace,
                    video=args.video,
                ),
                indent=2,
                sort_keys=True,
            )
        )
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
