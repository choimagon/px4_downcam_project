#!/usr/bin/env python3
"""Build a local comparison site for moving-QR precision-landing flights."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

from landing_rl.scenario import MOVING_QR_HEADING_DEG, MOVING_QR_SPEED_MPS, QR_SIZE_M, sample_annulus_start


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "rl_training"
ALGORITHMS = ("ppo", "ddpg", "sac")
SEEDS = {"ppo": 17, "ddpg": 19, "sac": 13}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def flight_metrics(path: Path) -> dict[str, float | str | None]:
    if not path.exists():
        return {"phase": None, "min_altitude_m": None, "max_altitude_m": None, "mean_image_error": None, "centered_frame_rate": None}
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        return {"phase": None, "min_altitude_m": None, "max_altitude_m": None, "mean_image_error": None, "centered_frame_rate": None}
    altitudes = [float(row["altitude_m"]) for row in rows if row.get("altitude_m")]
    errors = [math.hypot(float(row["error_x"]), float(row["error_y"])) for row in rows if row.get("error_x")]
    aligned = [int(row["aligned"]) for row in rows if row.get("aligned")]
    return {
        "phase": rows[-1].get("phase"),
        "min_altitude_m": max(0.0, min(altitudes)) if altitudes else None,
        "max_altitude_m": max(altitudes) if altitudes else None,
        "mean_image_error": sum(errors) / len(errors) if errors else None,
        "centered_frame_rate": sum(aligned) / len(aligned) if aligned else None,
    }


def fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    training = read_json(ARTIFACTS / "moving_qr_training_metrics.json")
    demos: dict[str, object] = {}
    cards: list[str] = []
    for name in ALGORITHMS:
        start = sample_annulus_start(SEEDS[name])
        video = f"{name}_moving_qr_tracking_landing_third_person.mp4"
        csv_path = PROJECT_ROOT / "logs" / f"{name}_moving_qr_tracking_landing.csv"
        log_path = PROJECT_ROOT / "logs" / f"{name}_moving_qr_tracking_landing.log"
        run = flight_metrics(csv_path)
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        landed = "MAV_CMD_NAV_LAND sent" in log
        train = training.get("metrics", {}).get(name, {})
        demos[name] = {
            "seed": SEEDS[name], "start_x_m": start.x_m, "start_y_m": start.y_m, "start_radius_m": start.radius_m,
            "target_speed_mps": MOVING_QR_SPEED_MPS, "target_heading_deg": MOVING_QR_HEADING_DEG,
            "video": video, "video_exists": (ARTIFACTS / video).exists(), "land_command_logged": landed, **run,
        }
        badge = "LAND COMMAND logged" if landed else "Not verified"
        cards.append(f'''<article class="card"><div class="title"><h2>{name.upper()}</h2><span class="{'ok' if landed else 'wait'}">{badge}</span></div>
<video controls preload="metadata"><source src="{video}" type="video/mp4">MP4 playback unavailable.</video>
<dl><div><dt>RL eval success</dt><dd>{fmt(train.get('success_rate'), 2)}</dd></div><div><dt>Terminal relative error</dt><dd>{fmt(train.get('mean_terminal_error_m'), 3, ' m')}</dd></div><div><dt>Mean camera error</dt><dd>{fmt(run['mean_image_error'], 3)}</dd></div><div><dt>Centered frames</dt><dd>{fmt(run['centered_frame_rate'], 1, '%') if run['centered_frame_rate'] is None else fmt(float(run['centered_frame_rate']) * 100, 1, '%')}</dd></div><div><dt>Initial drone position</dt><dd>({start.x_m:+.2f}, {start.y_m:+.2f}) m</dd></div><div><dt>Flight altitude</dt><dd>{fmt(run['min_altitude_m'],2)}–{fmt(run['max_altitude_m'],2)} m</dd></div></dl>
<p><a href="{video}">MP4</a> · <a href="../../logs/{name}_moving_qr_tracking_landing.csv">flight CSV</a> · <a href="../../logs/{name}_moving_qr_tracking_landing.log">run log</a></p></article>''')
    manifest = {
        "scenario": {"qr_size_m": QR_SIZE_M, "motion": "constant velocity", "speed_mps": MOVING_QR_SPEED_MPS, "heading_deg": MOVING_QR_HEADING_DEG, "world": "aruco_moving_qr"},
        "training": training, "demos": demos,
    }
    (ARTIFACTS / "moving_qr_demo_results.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    best = html.escape(str(training.get("best_algorithm", "pending")).upper())
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Moving QR Landing — PPO · DDPG · SAC</title>
<style>:root{{color-scheme:dark;--bg:#07101d;--panel:#101f34;--line:#2c496d;--cyan:#5bd5ff;--green:#60d8a3;--amber:#ffc35c}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#1a416e,transparent 39rem),var(--bg);color:#eef6ff;font:16px Inter,system-ui,sans-serif}}main{{width:min(1440px,calc(100% - 32px));margin:auto;padding:42px 0 64px}}h1{{font-size:clamp(2rem,5vw,3.8rem);letter-spacing:-.05em;margin:.15rem 0}}h2{{margin:0}}p{{color:#b8c9db;line-height:1.55}}.eyebrow{{color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}.lead{{max-width:850px;font-size:1.08rem}}.route{{display:grid;grid-template-columns:180px 1fr;gap:25px;align-items:center;margin:28px 0;padding:22px;border:1px solid var(--line);border-radius:19px;background:#0d192a}}.route svg{{width:100%;height:auto}}.route strong{{color:#fff}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}.card{{padding:18px;border:1px solid var(--line);border-radius:19px;background:linear-gradient(145deg,#152945,#0a1423)}}.title{{display:flex;gap:12px;justify-content:space-between;align-items:center;margin-bottom:15px}}.title span{{font-size:.72rem;font-weight:700;border-radius:99px;padding:6px 9px}}.ok{{background:var(--green);color:#052817}}.wait{{background:var(--amber);color:#3d2700}}video{{width:100%;aspect-ratio:16/9;background:#02060a;border:1px solid #385678;border-radius:11px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:15px 0 7px}}dl div{{background:#081220aa;padding:9px;border-radius:9px}}dt{{font-size:.71rem;color:#9db2ca}}dd{{margin:3px 0 0;font-size:.92rem;overflow-wrap:anywhere}}a{{color:var(--cyan)}}footer{{margin-top:32px;color:#9db2ca;font-size:.85rem}}@media(max-width:600px){{main{{width:calc(100% - 20px);padding-top:25px}}.route{{grid-template-columns:105px 1fr;gap:15px}}.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><div class="eyebrow">PX4 SITL · Gazebo Harmonic · moving-platform controller</div><h1>Moving QR Tracking &amp; Landing</h1><p class="lead">PPO, DDPG, and SAC start from randomized annulus positions, acquire a 0.4 m QR pad, track its deterministic constant-velocity motion, and only descend while the decoded QR remains centered.</p>
<section class="route"><svg viewBox="0 0 180 110" aria-label="QR constant-velocity path"><path d="M12 55H166" stroke="#5bd5ff" stroke-width="4" stroke-linecap="round"/><path d="M162 45l12 10-12 10" fill="none" stroke="#5bd5ff" stroke-width="4"/><rect x="72" y="37" width="36" height="36" fill="white" stroke="#111" stroke-width="5"/><text x="90" y="61" text-anchor="middle" fill="#111" font-weight="900" font-size="14">QR</text></svg><div><h2>Constant-velocity target</h2><p>The drone first acquires and centers the stationary QR. After a 50 s world staging period (42 s after policy startup), the pad moves at <strong>{MOVING_QR_SPEED_MPS:.2f} m/s</strong> along Gazebo ENU +X (heading {MOVING_QR_HEADING_DEG:.0f}). The pad uses PX4's moving-platform controller with stochastic motion disabled. The online tracker keeps velocity feed-forward active during descent and pauses descent if QR leaves the alignment gate.</p><p>Best trained policy: <strong>{best}</strong> · {html.escape(str(training.get('timesteps_per_algorithm','—')))} steps per algorithm.</p></div></section><section class="grid">{''.join(cards)}</section><footer>Each video is a full-monitor H.264 MP4 with maximized Gazebo third-person view. This is SITL evidence, not a hardware-flight claim.</footer></main></body></html>'''
    (ARTIFACTS / "moving_qr_landing_dashboard.html").write_text(document, encoding="utf-8")
    print(f"Wrote {ARTIFACTS / 'moving_qr_landing_dashboard.html'}")


if __name__ == "__main__":
    main()
