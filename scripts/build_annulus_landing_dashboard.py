#!/usr/bin/env python3
"""Build a self-contained local HTML dashboard for the annulus landing runs."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from landing_rl.scenario import INNER_RING_RADIUS_M, OUTER_RING_RADIUS_M, QR_SIZE_M, sample_annulus_start


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts/rl_training"
ALGORITHMS = ("ppo", "ddpg", "sac")
DEMO_SEEDS = {"ppo": 7, "ddpg": 9, "sac": 11}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_flight(csv_path: Path) -> dict[str, float | str | None]:
    if not csv_path.exists():
        return {"phase": None, "min_altitude_m": None, "max_altitude_m": None}
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    if not rows:
        return {"phase": None, "min_altitude_m": None, "max_altitude_m": None}
    altitudes = [float(row["altitude_m"]) for row in rows if row.get("altitude_m")]
    return {
        "phase": rows[-1].get("phase"),
        # PX4's local-position estimate can report a few centimetres below
        # ground immediately after touchdown; present physical AGL in the UI.
        "min_altitude_m": max(0.0, min(altitudes)) if altitudes else None,
        "max_altitude_m": max(altitudes) if altitudes else None,
    }


def format_metric(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    training = load_json(ARTIFACTS / "training_metrics.json")
    output = ARTIFACTS / "annulus_landing_dashboard.html"
    run_manifest: dict[str, object] = {
        "scenario": {
            "qr_size_m": QR_SIZE_M,
            "inner_ring_radius_m": INNER_RING_RADIUS_M,
            "outer_ring_radius_m": OUTER_RING_RADIUS_M,
            "video": "Full-monitor H.264 MP4 with maximized Gazebo third-person view",
        },
        "training": training,
        "demos": {},
    }
    cards: list[str] = []
    for algorithm in ALGORITHMS:
        seed = DEMO_SEEDS[algorithm]
        start = sample_annulus_start(seed)
        video_name = f"{algorithm}_annulus_qr_landing_third_person.mp4"
        video_path = ARTIFACTS / video_name
        csv_path = PROJECT_ROOT / "logs" / f"{algorithm}_annulus_qr_landing.csv"
        log_path = PROJECT_ROOT / "logs" / f"{algorithm}_annulus_qr_landing.log"
        flight = latest_flight(csv_path)
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        land_command = "MAV_CMD_NAV_LAND sent" in log
        metrics = training.get("metrics", {}).get(algorithm, {})
        run = {
            "seed": seed,
            "start_x_m": start.x_m,
            "start_y_m": start.y_m,
            "start_radius_m": start.radius_m,
            "video": video_name,
            "video_exists": video_path.exists(),
            "land_command_logged": land_command,
            **flight,
        }
        run_manifest["demos"][algorithm] = run
        result_label = "LAND COMMAND logged" if land_command else "Pending / not verified"
        result_class = "pass" if land_command else "pending"
        video = (
            f'<video controls preload="metadata"><source src="{html.escape(video_name)}" type="video/mp4">'
            "This browser cannot play the local MP4.</video>"
            if video_path.exists()
            else '<div class="video-placeholder">MP4 is being generated.</div>'
        )
        cards.append(
            f"""<article class=\"card\">
  <div class=\"card-title\"><h2>{algorithm.upper()}</h2><span class=\"badge {result_class}\">{result_label}</span></div>
  {video}
  <dl class=\"stats\">
    <div><dt>Evaluation success</dt><dd>{format_metric(metrics.get('success_rate'), 2)}</dd></div>
    <div><dt>Terminal error</dt><dd>{format_metric(metrics.get('mean_terminal_error_m'), 3)} m</dd></div>
    <div><dt>Mean reward</dt><dd>{format_metric(metrics.get('mean_reward'), 1)}</dd></div>
    <div><dt>Random start</dt><dd>seed {seed} · r={start.radius_m:.2f} m</dd></div>
    <div><dt>Start position</dt><dd>({start.x_m:+.2f}, {start.y_m:+.2f}) m</dd></div>
    <div><dt>Flight altitude</dt><dd>{format_metric(flight['min_altitude_m'], 2)}–{format_metric(flight['max_altitude_m'], 2)} m</dd></div>
  </dl>
  <p class=\"files\"><a href=\"{html.escape(video_name)}\">Download MP4</a> · <a href=\"../../logs/{algorithm}_annulus_qr_landing.csv\">Flight CSV</a> · <a href=\"../../logs/{algorithm}_annulus_qr_landing.log\">Run log</a></p>
</article>"""
        )

    best = html.escape(str(training.get("best_algorithm", "pending")).upper())
    html_document = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>QR Annulus Landing — PPO · DDPG · SAC</title>
  <style>
    :root {{ color-scheme: dark; --ink:#edf4ff; --muted:#aab9ce; --panel:#111a2a; --edge:#273853; --cyan:#49c8ff; --orange:#ffb54a; --green:#49d49d; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 20% 0,#17335c 0,#09111e 38rem); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1440px,calc(100% - 32px)); margin:auto; padding:42px 0 64px; }}
    h1 {{ margin:0; font-size:clamp(2rem,5vw,3.6rem); letter-spacing:-.045em; }} h2 {{ margin:0; }} p {{ color:var(--muted); line-height:1.55; }}
    .eyebrow {{ color:var(--cyan); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.78rem; }}
    .lead {{ max-width:780px; font-size:1.08rem; }}
    .scenario {{ margin:30px 0; display:grid; grid-template-columns:170px 1fr; gap:28px; align-items:center; padding:22px; border:1px solid var(--edge); border-radius:20px; background:rgba(17,26,42,.82); }}
    .rings {{ position:relative; aspect-ratio:1; border:3px solid var(--orange); border-radius:50%; }} .rings:before {{ content:\"\"; position:absolute; inset:28%; border:3px solid var(--cyan); border-radius:50%; }} .rings:after {{ content:\"QR\"; position:absolute; width:25%; aspect-ratio:1; left:37.5%; top:37.5%; display:grid; place-items:center; color:#111; font-weight:900; background:white; border:4px solid #111; }}
    .scenario ul {{ padding-left:20px; color:var(--muted); line-height:1.8; }} .scenario strong {{ color:var(--ink); }}
    .summary {{ color:var(--muted); margin:0 0 20px; }} .summary strong {{ color:var(--green); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }} .card {{ padding:18px; border:1px solid var(--edge); border-radius:20px; background:linear-gradient(145deg,rgba(20,32,52,.96),rgba(10,16,27,.96)); box-shadow:0 16px 45px rgba(0,0,0,.22); }}
    .card-title {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:15px; }} .badge {{ font-size:.73rem; padding:6px 9px; border-radius:99px; white-space:nowrap; }} .pass {{ color:#052415; background:var(--green); }} .pending {{ color:#352400; background:var(--orange); }}
    video,.video-placeholder {{ width:100%; aspect-ratio:16/9; object-fit:contain; border-radius:12px; background:#03070d; border:1px solid #263b59; }} .video-placeholder {{ display:grid; place-items:center; color:var(--muted); }}
    .stats {{ margin:16px 0 6px; display:grid; grid-template-columns:1fr 1fr; gap:10px; }} .stats div {{ padding:9px; min-width:0; background:rgba(7,13,23,.58); border-radius:9px; }} dt {{ font-size:.72rem; color:var(--muted); }} dd {{ margin:3px 0 0; font-size:.92rem; overflow-wrap:anywhere; }} .files {{ margin:13px 0 0; font-size:.83rem; }} a {{ color:var(--cyan); }}
    footer {{ margin-top:32px; color:var(--muted); font-size:.84rem; }} @media (max-width:600px) {{ main {{ width:min(100% - 20px,1440px); padding-top:24px; }} .scenario {{ grid-template-columns:105px 1fr; gap:16px; }} .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <div class=\"eyebrow\">PX4 · Gazebo Harmonic · visual precision landing</div>
  <h1>QR Annulus Landing</h1>
  <p class=\"lead\">A side-by-side record of PPO, DDPG, and SAC navigating from random ground starts to a half-size QR landing pad. Each player is the complete monitor recording with Gazebo maximized in third-person.</p>
  <section class=\"scenario\"><div class=\"rings\" aria-label=\"QR at center, blue 2 metre ring, orange 7 metre ring\"></div><div><h2>Scenario</h2><ul><li><strong>QR pad:</strong> {QR_SIZE_M:.1f} × {QR_SIZE_M:.1f} m — half the former size.</li><li><strong>Blue inner ring:</strong> {INNER_RING_RADIUS_M:.0f} m. <strong>Orange outer ring:</strong> {OUTER_RING_RADIUS_M:.0f} m.</li><li>Every vehicle starts on the ground at a seeded random position strictly between both rings, climbs, visually acquires QR, then the learned policy centers and lands.</li></ul></div></section>
  <p class=\"summary\">Best evaluated policy: <strong>{best}</strong>. Training seed: {html.escape(str(training.get('seed', '—')))} · timesteps/policy: {html.escape(str(training.get('timesteps_per_algorithm', '—')))}.</p>
  <section class=\"grid\">{''.join(cards)}</section>
  <footer>Open this file locally; MP4, CSV, and log links are relative to the project artifacts. Generated by scripts/build_annulus_landing_dashboard.py.</footer>
</main></body></html>"""
    output.write_text(html_document, encoding="utf-8")
    (ARTIFACTS / "annulus_demo_results.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
