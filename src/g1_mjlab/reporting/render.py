"""Render one run as a portable, offline HTML research report."""

# Embedded HTML/CSS/JavaScript is intentionally readable as one offline template.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import read_jsonl


def _safe(value: object) -> str:
    return html.escape(str(value))


def _json_for_script(value: object) -> str:
    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _series(records: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        metric = record.get("metric")
        value = record.get("value")
        step = record.get("update", record.get("step", 0))
        if (
            isinstance(metric, str)
            and isinstance(value, (int, float))
            and isinstance(step, (int, float))
        ):
            result[metric].append({"x": float(step), "y": float(value)})
    return {name: sorted(points, key=lambda p: p["x"]) for name, points in result.items()}


def _rows(values: dict[str, Any]) -> str:
    return "".join(
        f"<tr><td>{_safe(key)}</td><td><code>{_safe(value)}</code></td></tr>"
        for key, value in sorted(values.items())
    )


def render_report(run_dir: Path, output_dir: Path | None = None, *, final: bool = False) -> Path:
    """Render a self-contained report from non-executable run artifacts."""
    root = run_dir.resolve()
    manifest = _read_object(root / "manifest.json")
    config = _read_object(root / "config.json")
    reward_profile = _read_object(root / "reward-profile.json")
    ppo_profile = _read_object(root / "ppo-profile.json")
    contract = _read_object(root / "contract.json")
    algorithm = _read_object(root / "algorithm.json")
    mdp = _read_object(root / "mdp.json")
    memory = _read_object(root / "memory.json")
    bundle = _read_object(root / "policy-bundle.json")
    development_evaluation = _read_object(root / "evaluation" / "summary.json")
    final_evaluation = _read_object(root / "evaluation" / "final-100x60" / "summary.json")
    evaluation = final_evaluation or development_evaluation
    native_final = _read_object(root / "native-final-100x60" / "summary.json")
    records, truncated = read_jsonl(root / "metrics" / "metrics.jsonl")
    target = (output_dir or root / "report").resolve()
    target.relative_to(root)
    target.mkdir(parents=True, exist_ok=True)
    index = target / "index.html"
    if final and index.exists():
        raise FileExistsError(f"final report already exists: {index}")
    videos = sorted((root / "videos").glob("*.mp4"))
    if videos:
        video_cards = "".join(
            f'<article class="card"><p class="eyebrow">Deterministic playback</p>'
            f'<video controls preload="metadata" src="{_safe(Path(os.path.relpath(video, target)).as_posix())}"></video>'
            f"<p><code>{_safe(video.name)}</code></p></article>"
            for video in videos
        )
        video_html = f'<h2>Policy playback</h2><section class="grid">{video_cards}</section><p class="note">Playback is qualitative evidence. The frozen numerical evaluations remain authoritative.</p>'
        video_note = "Deterministic playback was captured with separate hash metadata."
    else:
        video_html = ""
        video_note = "Video was not captured."

    title = _safe(manifest.get("run_id", root.name))
    status = _safe(manifest.get("status", "unknown"))
    passed = int(evaluation.get("passed", 0))
    planned = int(evaluation.get("planned", 0))
    evaluation_phase = _safe(
        evaluation.get("phase", "legacy development (not an independent final test)")
    )
    outcome = (
        "The training process completed its declared update budget."
        if manifest.get("status") == "completed"
        else f"The recorded training state is {status}; budget completion is not established."
    )
    eval_html = (
        f"<p class='hero-number'>{passed}/{planned}</p><p>{evaluation_phase} trials passed</p>"
        if evaluation
        else "<p class='empty'>Evaluation has not been recorded.</p>"
    )
    trial_rows = (
        "".join(
            "<tr><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{}</td></tr>".format(
                item.get("trial_id", ""),
                "pass" if item.get("passed") else "fail",
                float(item.get("survived_seconds", 0)),
                float(item.get("max_drift_m", 0)),
                _safe(item.get("failure_reason") or "—"),
            )
            for item in evaluation.get("trials", [])
        )
        or "<tr><td colspan='5'>No evaluation trials recorded</td></tr>"
    )
    contract_rows = (
        "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                _safe(item.get("name", "")),
                item.get("offset", ""),
                item.get("size", ""),
                _safe(item.get("unit", "")),
                _safe(item.get("source", item.get("resolved_term", {}).get("func", ""))),
            )
            for item in contract.get("actor_fields", [])
        )
        or "<tr><td colspan='5'>Contract unavailable</td></tr>"
    )
    actuator_rows = (
        "".join(
            "<tr><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.1f}</td><td>{:.6f}</td></tr>".format(
                _safe(", ".join(item.get("target_names_expr", []))),
                float(item.get("stiffness", 0)),
                float(item.get("damping", 0)),
                float(item.get("effort_limit", 0)),
                float(item.get("armature", 0)),
            )
            for item in contract.get("actuator_groups", [])
        )
        or "<tr><td colspan='5'>Actuator metadata unavailable</td></tr>"
    )
    if contract.get("schema_version") == 2:
        actuator_rows = "".join(
            f"<tr><td>{_safe(name)}</td><td>{gain[0]:.6g}</td><td>{-bias[2]:.6g}</td>"
            f"<td>{max(abs(v) for v in limits):.6g}</td><td>{armature:.6g}</td></tr>"
            for name, gain, bias, limits, armature in zip(
                contract["action_names"],
                contract["actuator_gain_parameters"],
                contract["actuator_bias_parameters"],
                contract["actuator_force_limits"],
                contract["armature"],
                strict=True,
            )
        )
    reward_rows = (
        "".join(
            f"<tr><td>{_safe(name)}</td><td>{_safe(term.get('weight'))}</td>"
            f"<td><code>{_safe(term.get('func'))}</code></td></tr>"
            for name, term in mdp.get("rewards", {}).items()
        )
        or "<tr><td colspan='3'>Resolved reward configuration unavailable for this legacy run.</td></tr>"
    )
    actor = algorithm.get("actor", {})
    critic = algorithm.get("critic", {})
    ppo = algorithm.get("algorithm", {})
    actor_input_dim = sum(int(item.get("size", 0)) for item in contract.get("actor_fields", []))
    critic_input_dim = sum(int(item.get("size", 0)) for item in contract.get("critic_fields", []))
    action_dim = len(contract.get("action_names", []))
    actor_arch = (
        f"{actor_input_dim} → {' → '.join(map(str, actor.get('hidden_dims', [])))} → {action_dim}"
        if actor
        else "unavailable"
    )
    critic_arch = (
        f"{critic_input_dim} → {' → '.join(map(str, critic.get('hidden_dims', [])))} → 1"
        if critic
        else "unavailable"
    )
    integrity = (
        "The final metric line was incomplete after interruption."
        if truncated
        else "All readable metric records are structurally complete."
    )
    resolved_config = dict(config)
    resolved_config.update(
        {f"reward_profile.{key}": value for key, value in reward_profile.items()}
    )
    resolved_config.update({f"ppo_profile.{key}": value for key, value in ppo_profile.items()})

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 research run — {title}</title>
<style>:root{{--ink:#111;--muted:#666;--line:#ddd;--paper:#f7f7f2;--accent:#e54b2a;--ok:#18794e}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1180px;margin:auto;padding:48px 28px 100px}}header{{border-top:6px solid var(--ink);padding:28px 0 52px}}h1{{font:700 clamp(38px,7vw,86px)/.95 Georgia,serif;letter-spacing:-.05em;margin:.15em 0;overflow-wrap:anywhere}}h2{{font:600 30px/1.1 Georgia,serif;margin:64px 0 20px}}.eyebrow,.meta{{text-transform:uppercase;letter-spacing:.11em;font-size:12px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.card{{background:#fff;border:1px solid var(--line);padding:24px;min-height:150px}}.hero-number{{font:700 clamp(25px,3vw,42px)/1 Georgia,serif;margin:12px 0;overflow-wrap:anywhere}}.status{{color:var(--ok)}}canvas{{width:100%;height:260px;border:1px solid var(--line);background:#fff}}video{{display:block;width:100%;background:#000}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);font-size:14px}}code{{font:13px ui-monospace,monospace}}.empty,.note{{color:var(--muted)}}.table-scroll{{overflow-x:auto}}@media(max-width:760px){{main{{padding:24px 16px 70px}}.grid{{grid-template-columns:1fr}}h2{{margin-top:42px}}}}</style></head>
<body><main><header><p class="eyebrow">Humanoid locomotion research · run report</p><h1>{title}</h1><p class="meta">Status: <span class="status">{status}</span> · schema {_safe(manifest.get("schema_version", "?"))}</p></header>
<section class="grid"><article class="card"><p class="eyebrow">Training outcome</p><p class="hero-number">{status}</p><p>{_safe(manifest.get("max_iterations", "?"))} planned PPO updates</p></article><article class="card"><p class="eyebrow">Evaluation</p>{eval_html}</article><article class="card"><p class="eyebrow">Reproducibility</p><p><code>{_safe(manifest.get("config_sha256", "unavailable"))[:16]}</code></p><p>configuration hash</p></article></section>
<h2>What happened</h2><p>{outcome} Behavioral evaluation is separate: {passed} of {planned} recorded trials passed. Completion is not a standing-success claim.</p><p>Evaluation phase: {evaluation_phase}. Criteria and provenance: <code>{_safe(evaluation.get("criteria", "Legacy survival-only evaluation"))}</code>.</p>
<h2>Learning curves</h2><p class="note">Select any recorded reward, loss, policy, episode, or performance metric.</p><select id="metric" aria-label="Metric"></select><canvas id="chart" width="1080" height="260"></canvas>
<h2>Standing versus survival</h2><p>Strict standing passes: {passed}/{planned}. Survival-only passes: {_safe(evaluation.get("survival_passed", "not separately recorded"))}. Standing success Wilson 95% interval: {_safe(evaluation.get("standing_success_wilson_95", "unavailable"))}. Phase: {evaluation_phase}. Development selection and untouched final testing are stored separately.</p>
<h2>Sim-to-sim transfer</h2><section class="grid"><article class="card"><p class="eyebrow">mjlab / MuJoCo Warp</p><p class="hero-number">{passed}/{planned}</p><p>strict {_safe(evaluation.get("horizon_seconds", "?"))}-second final trials</p></article><article class="card"><p class="eyebrow">Native MuJoCo / ONNX</p><p class="hero-number">{_safe(native_final.get("passed", 0))}/{_safe(native_final.get("planned", 0))}</p><p>same frozen final initial states</p></article><article class="card"><p class="eyebrow">Qualification</p><p class="hero-number">{_safe(bundle.get("final_test_qualified", False))}</p><p>requires at least 95/100 in both backends</p></article></section>
{video_html}
<h2>Reward definition</h2><p>Each weighted reward rate is multiplied by the control timestep. The termination weight is −5/control_dt, producing a −5 event penalty. Episode_Reward logs are accumulated reward divided by the configured episode horizon, not raw per-step reward.</p><div class="table-scroll"><table><thead><tr><th>Term</th><th>Weight</th><th>Implementation</th></tr></thead><tbody>{reward_rows}</tbody></table></div><details><summary>Full resolved reward, termination, reset and command parameters</summary><pre>{_safe(json.dumps(mdp, indent=2))}</pre></details>
<h2>Evaluation trials</h2><div class="table-scroll"><table><thead><tr><th>Trial</th><th>Outcome</th><th>Survival (s)</th><th>Max drift (m)</th><th>Reason</th></tr></thead><tbody>{trial_rows}</tbody></table></div>
<h2>Neural policy</h2><section class="grid"><article class="card"><p class="eyebrow">Actor</p><p class="hero-number">{actor_arch}</p><p>{_safe(actor.get("activation", "?"))} · normalized inputs · Gaussian actions</p></article><article class="card"><p class="eyebrow">Critic</p><p class="hero-number">{critic_arch}</p><p>Privileged foot/contact information is used only during training.</p></article><article class="card"><p class="eyebrow">Rollout</p><p class="hero-number">{_safe(algorithm.get("num_steps_per_env", "?"))}</p><p>steps per robot before each update</p></article></section>
<h2>Policy input vector</h2><div class="table-scroll"><table><thead><tr><th>Field</th><th>Offset</th><th>Size</th><th>Unit</th><th>Source</th></tr></thead><tbody>{contract_rows}</tbody></table></div>
<h2>PD actuator contract</h2><p><code>qtarget = qnominal + scale × action</code>. Built-in position actuators apply joint-specific stiffness, damping, armature, and effort limits every physics step.</p><div class="table-scroll"><table><thead><tr><th>Joint selector</th><th>Kp</th><th>Kd</th><th>Limit (N·m)</th><th>Armature</th></tr></thead><tbody>{actuator_rows}</tbody></table></div>
<h2>Algorithm</h2><div class="card"><p><strong>PPO with generalized advantage estimation.</strong></p><p>Reference equations: <code>δₜ = rₜ + γbₜV(sₜ₊₁) − V(sₜ)</code>; <code>Aₜ = δₜ + γλcₜAₜ₊₁</code>. Bootstrap mask b is zero at true termination; recursion mask c is zero at any episode boundary.</p><p><code>Lclip = E[min(ρₜAₜ, clip(ρₜ,1−ε,1+ε)Aₜ)]</code>; ρ is the new/old action probability ratio, not reward r.</p><p>Pinned RSL-RL 5.5.0 instead implements timeout compensation by adding γ times the stored transition value to reward, then using done-masked GAE. This is not a claim of final-observation bootstrapping.</p><p>Parallel worlds supply {_safe(algorithm.get("num_steps_per_env", "?"))}-step rollouts. PPO reuses each rollout for shuffled minibatch epochs.</p></div><div class="table-scroll"><table><thead><tr><th>PPO setting</th><th>Value</th></tr></thead><tbody>{_rows(ppo)}</tbody></table></div>
<h2>Resolved configuration</h2><div class="table-scroll"><table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>{_rows(resolved_config)}</tbody></table></div>
<h2>Checkpoint and resource provenance</h2><div class="table-scroll"><table><tbody>{_rows(bundle)}{_rows(memory)}</tbody></table></div>
<h2>Data integrity and limitations</h2><p>{integrity}</p><p>This report covers the selected training run and its mjlab/native-MuJoCo qualification. It is not hardware-readiness evidence. Diagnostic joint, target, actuator-force, body-state, and action traces are stored separately from the training scalars. {video_note}</p>
<script>const SERIES={_json_for_script(_series(records))};const sel=document.querySelector('#metric'),c=document.querySelector('#chart'),ctx=c.getContext('2d');for(const k of Object.keys(SERIES).sort()){{const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o)}}function draw(){{ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle='#ddd';ctx.strokeRect(0,0,c.width,c.height);const a=SERIES[sel.value]||[];if(!a.length)return;const ys=a.map(p=>p.y),lo=Math.min(...ys),hi=Math.max(...ys),span=hi-lo||1;ctx.strokeStyle='#e54b2a';ctx.lineWidth=3;ctx.beginPath();a.forEach((p,i)=>{{const x=30+(p.x-a[0].x)/(a[a.length-1].x-a[0].x||1)*(c.width-60),y=c.height-25-(p.y-lo)/span*(c.height-50);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke();ctx.fillStyle='#111';ctx.font='13px system-ui';ctx.fillText(sel.value+'  min '+lo.toPrecision(4)+'  max '+hi.toPrecision(4)+'  updates '+a[0].x+'–'+a[a.length-1].x,35,20)}}sel.addEventListener('change',draw);draw();</script></main></body></html>"""
    temporary = index.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(index)
    return index
