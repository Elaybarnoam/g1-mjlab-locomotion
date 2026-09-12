"""Offline walking experiment research page generated from immutable evidence."""

# Embedded template is kept dependency-free for portable offline reports.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..artifacts import sha256_file


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("walking trace rows must be objects")
        rows.append(value)
    return rows


def _script_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def render_walking_research_report(
    run: Path,
    experiment_table: Path,
    policy_spec: Path,
    native_summary: Path,
    native_trace: Path,
    parity_summary: Path,
    output: Path,
) -> Path:
    """Render available telemetry and label every unavailable channel without invention."""
    if output.exists():
        raise FileExistsError(f"walking research report output already exists: {output}")
    run = run.resolve(strict=True)
    sources = {
        "manifest": run / "manifest.json",
        "config": run / "config.json",
        "algorithm": run / "algorithm.json",
        "mdp": run / "mdp.json",
        "optimization": run / "optimization.json",
        "learning": run / "learning-summary.json",
        "experiment": experiment_table.resolve(strict=True),
        "policy_spec": policy_spec.resolve(strict=True),
        "native": native_summary.resolve(strict=True),
        "parity": parity_summary.resolve(strict=True),
        "trace": native_trace.resolve(strict=True),
    }
    objects = {name: _read(path) for name, path in sources.items() if name != "trace"}
    trace = _trace(sources["trace"])
    experiment = objects["experiment"]
    specification = objects["policy_spec"]
    learning = objects["learning"]
    curves: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in learning.get("losses", []):
        if isinstance(item.get("value"), (int, float)):
            curves[str(item["metric"])].append(
                {"x": float(item["update"]), "y": float(item["value"])}
            )
    for row in trace:
        time_s = float(row["time_s"])
        for key, label in (
            ("requested_speed_m_s", "Requested speed (m/s)"),
            ("applied_speed_m_s", "Applied speed (m/s)"),
            ("pelvis_height_m", "Pelvis height (m)"),
        ):
            curves[label].append({"x": time_s, "y": float(row[key])})
        action = row.get("action")
        if isinstance(action, list):
            curves["Action RMS (normalized)"].append(
                {"x": time_s, "y": math.sqrt(sum(float(x) ** 2 for x in action) / len(action))}
            )
    comparison_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}/16</td><td>{}/16</td><td>{:.3f}</td><td>{:.3f}</td></tr>".format(
            html.escape(str(row.get("arm", ""))),
            html.escape(str(row.get("completed_updates", ""))),
            html.escape(Path(str(row.get("checkpoint", "missing"))).name),
            row.get("selection", {}).get("function_pass_count", 0),
            row.get("selection", {}).get("function_style_pass_count", 0),
            float(row.get("selection", {}).get("median_command_rms_m_s", 0)),
            float(row.get("selection", {}).get("errors", {}).get("slip", 0)),
        )
        for row in experiment.get("rows", [])
    )
    rewards = objects["mdp"].get("rewards", {})
    reward_rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str(term.get('weight', 'missing')))}</td><td>{html.escape(str(term.get('func', 'missing')))}</td></tr>"
        for name, term in rewards.items()
    )
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    missing_telemetry = [
        "gradient norm time series (only final value recorded)",
        "joint torque and saturation time series",
        "raw/filtered contact and normal-force raster in native trace",
        "full-resolution reference foot overlay for this native trace",
        "rollout minibatch tensors including terminal observations",
        "qualified per-speed backend outcomes",
    ]
    payload = {
        "schema_version": 1,
        "status": "failed_hypothesis_unqualified_development",
        "sources": hashes,
        "curves": curves,
        "experiment": experiment,
        "policy_validation": specification.get("validation"),
        "native": objects["native"],
        "parity": objects["parity"],
        "missing_telemetry": missing_telemetry,
    }
    output.mkdir(parents=True)
    (output / "report-data.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "policy-spec.json").write_text(
        json.dumps(specification, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    best = experiment.get("best_observed", {})
    best_selection = best.get("selection", {})
    missing_cards = "".join(f"<li>{html.escape(item)}</li>" for item in missing_telemetry)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 walking — Stage 19 research report</title>
<style>:root{{--paper:#f5f3ed;--card:#fff;--ink:#141414;--muted:#62625d;--line:#d7d3ca;--accent:#e3532d;--blue:#275dad}}@media(prefers-color-scheme:dark){{:root{{--paper:#111;--card:#191919;--ink:#f4f2ec;--muted:#aaa69d;--line:#393833;--accent:#ff7652;--blue:#7da7ff}}}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}nav{{position:sticky;top:0;background:color-mix(in srgb,var(--paper) 92%,transparent);backdrop-filter:blur(12px);padding:.7rem max(1rem,calc((100vw - 1180px)/2));border-bottom:1px solid var(--line);z-index:2}}nav a{{color:var(--ink);margin-right:1rem}}main{{max-width:1180px;margin:auto;padding:4rem 1.5rem 7rem}}h1{{font:700 clamp(3rem,8vw,7rem)/.9 Georgia,serif;letter-spacing:-.055em;max-width:11ch}}h2{{font:600 clamp(1.8rem,4vw,3rem)/1.05 Georgia,serif;margin-top:5rem}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}}.card{{background:var(--card);border:1px solid var(--line);padding:1.3rem}}.number{{font:700 2.4rem/1 Georgia,serif;margin:.4rem 0}}.failed{{color:var(--accent)}}table{{border-collapse:collapse;width:100%;background:var(--card)}}th,td{{padding:.65rem;text-align:left;border-bottom:1px solid var(--line);font-size:.86rem}}.scroll{{overflow:auto}}canvas{{width:100%;height:330px;background:var(--card);border:1px solid var(--line)}}select{{font:inherit;padding:.5rem;background:var(--card);color:var(--ink)}}.flow{{display:flex;gap:.7rem;align-items:center;overflow:auto}}.flow div{{min-width:150px;background:var(--card);border:1px solid var(--line);padding:1rem}}.flow span{{color:var(--accent);font-size:1.4rem}}code{{font-family:ui-monospace,monospace;overflow-wrap:anywhere}}.note{{color:var(--muted)}}@media(max-width:768px){{.grid{{grid-template-columns:1fr}}.flow{{align-items:stretch;flex-direction:column}}.flow span{{transform:rotate(90deg);align-self:center}}}}@media(max-width:390px){{main{{padding:2rem 1rem 5rem}}nav a{{font-size:.8rem;margin-right:.5rem}}}}@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}</style></head>
<body><nav aria-label="Report sections"><a href="#result">Result</a><a href="#curves">Curves</a><a href="#policy">Policy</a><a href="#limits">Limits</a></nav><main>
<p class="eyebrow">Unitree G1 · reproducible locomotion research</p><h1>Walking learned. Natural gait did not.</h1><p>This page reports the evidence without a release claim. The best observed checkpoint completed all 16 functional development scenarios but passed 0/16 combined function-and-style gates.</p>
<section id="result" class="grid"><article class="card"><p class="eyebrow">Experiment result</p><p class="number failed">failed hypothesis</p><p>No Stage 19 arm earned extension.</p></article><article class="card"><p class="eyebrow">Best function</p><p class="number">{best_selection.get("function_pass_count", 0)}/16</p><p>Arm {html.escape(str(best.get("arm", "missing")))}, update {best.get("completed_updates", "missing")}</p></article><article class="card"><p class="eyebrow">Best style</p><p class="number failed">{best_selection.get("function_style_pass_count", 0)}/16</p><p>Not development-qualified.</p></article></section>
<h2>Run lineage</h2><p>Three bounded PPO reward/mechanics arms consumed 800 updates and 1,228,800 transitions. The exact retained checkpoint is <code>{html.escape(str(best.get("checkpoint_sha256", "missing")))}</code>. It is a best-observed diagnostic artifact, not a selected release candidate.</p>
<h2>Experiment comparison</h2><div class="scroll"><table><thead><tr><th>Arm</th><th>Updates</th><th>Checkpoint</th><th>Function</th><th>Function+style</th><th>Command RMS</th><th>Slip error</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<h2 id="curves">Recorded curves</h2><p class="note">The selector exposes every available loss/control series. X is PPO update for training metrics and seconds for native control metrics. Full source arrays are in report-data.json.</p><label for="metric">Series </label><select id="metric"></select><canvas id="chart" width="1100" height="330" aria-label="Selected telemetry chart"></canvas>
<h2>Controller/data path</h2><div class="flow"><div>102 actor values<br><small>physical sensors + previous action + gait host state</small></div><span>→</span><div>frozen normalizer</div><span>→</span><div>512–256–128 ELU actor</div><span>→</span><div>29 normalized actions</div><span>→</span><div>joint targets + built-in PD</div><span>→</span><div>4 × 5 ms MuJoCo physics</div></div>
<h2 id="policy">Exact policy specification</h2><section class="grid"><article class="card"><p class="eyebrow">Actor</p><p class="number">102 → 29</p><p>{specification.get("networks", {}).get("actor", {}).get("parameter_count", "missing"):,} affine parameters</p></article><article class="card"><p class="eyebrow">Critic</p><p class="number">114 → 1</p><p>Training only; privileged contact state.</p></article><article class="card"><p class="eyebrow">Parity</p><p class="number">{objects["parity"].get("samples", 0)}</p><p>shared physical states; result {objects["parity"].get("passed", False)}</p></article></section><p><a href="policy-spec.json">Open the full generated tensor-level specification</a>. The public Markdown is generated from the same object.</p>
<h2>Reward table</h2><div class="scroll"><table><thead><tr><th>Term</th><th>Weight</th><th>Implementation</th></tr></thead><tbody>{reward_rows}</tbody></table></div>
<h2>Deterministic evaluation</h2><p>mjlab development: {best_selection.get("function_pass_count", 0)}/16 function, {best_selection.get("function_style_pass_count", 0)}/16 combined. Independent native CPU: {objects["native"].get("passed_finite_rollouts", 0)}/{objects["native"].get("planned_rollouts", 0)} finite 60-second rollouts. The native count proves runtime stability only; it does not override failed gait-style metrics.</p>
<h2>Algorithm</h2><p>PPO uses 24-step rollouts from 64 parallel robots, five epochs, four minibatches, γ=0.99, λ=0.95, clip ε=0.2, clipped value loss, adaptive KL target 0.01, and zero entropy coefficient for this arm. The generated policy specification records configured and effective values plus the exact timeout-compensation semantics.</p>
<h2 id="limits">Unavailable telemetry and limitations</h2><ul>{missing_cards}</ul><p>Unavailable means not recorded; no zero value is substituted. The final 100×60 qualification suite was not opened because the development-qualified policy, 0–0.8 m/s coverage, and owner visual approval prerequisites are absent. This page has no OpenAI affiliation and makes no sim-to-real readiness claim.</p>
<script>const S={_script_json(curves)};const sel=document.querySelector('#metric'),canvas=document.querySelector('#chart'),ctx=canvas.getContext('2d');Object.keys(S).sort().forEach(k=>{{const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o)}});function draw(){{const a=S[sel.value]||[];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--line');ctx.strokeRect(0,0,canvas.width,canvas.height);if(!a.length)return;const xs=a.map(p=>p.x),ys=a.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),dx=xmax-xmin||1,dy=ymax-ymin||1;ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--accent');ctx.lineWidth=3;ctx.beginPath();a.forEach((p,i)=>{{const x=45+(p.x-xmin)/dx*(canvas.width-70),y=canvas.height-35-(p.y-ymin)/dy*(canvas.height-65);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke();ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ink');ctx.font='14px system-ui';ctx.fillText(sel.value+' · min '+ymin.toPrecision(4)+' · max '+ymax.toPrecision(4)+' · n='+a.length,50,24)}}sel.addEventListener('change',draw);draw()</script></main></body></html>"""
    index = output / "index.html"
    index.write_text(document, encoding="utf-8")
    return index
