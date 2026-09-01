#!/usr/bin/env python3
"""Build a local comparison site for curved-path, wavy-QR SITL landings."""

from __future__ import annotations

import csv
import html
import json
import math
import shutil
from pathlib import Path

from landing_rl.scenario import QR_SIZE_M, WAVY_QR_SPEED_MULTIPLIER, random_wavy_motion_profile, sample_annulus_start


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "rl_training"
ALGORITHMS = ("ppo", "ddpg", "sac")
SPAWN_SEEDS = {"ppo": 17, "ddpg": 19, "sac": 13}
TRAJECTORY_SEEDS = {"ppo": 913, "ddpg": 947, "sac": 977}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def flight_metrics(path: Path) -> dict[str, float | str | None]:
    empty = {
        "phase": None, "min_altitude_m": None, "max_altitude_m": None,
        "mean_image_error": None, "p95_image_error": None,
        "centered_frame_rate": None, "duration_s": None,
    }
    if not path.exists():
        return empty
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        return empty
    altitudes = [float(row["altitude_m"]) for row in rows if row.get("altitude_m")]
    errors = [math.hypot(float(row["error_x"]), float(row["error_y"])) for row in rows if row.get("error_x")]
    aligned = [int(row["aligned"]) for row in rows if row.get("aligned")]
    times = [float(row["time_s"]) for row in rows if row.get("time_s")]
    return {
        "phase": rows[-1].get("phase"),
        "min_altitude_m": max(0.0, min(altitudes)) if altitudes else None,
        "max_altitude_m": max(altitudes) if altitudes else None,
        "mean_image_error": sum(errors) / len(errors) if errors else None,
        "p95_image_error": percentile(errors, 0.95),
        "centered_frame_rate": sum(aligned) / len(aligned) if aligned else None,
        "duration_s": times[-1] - times[0] if len(times) > 1 else None,
    }


def fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def view_sync_offset(path: Path) -> float | None:
    """Return the recorded third-person lead-in removed during composition."""
    try:
        third_person_started, drone_view_started = map(float, path.read_text(encoding="utf-8").split())
    except (OSError, ValueError):
        return None
    return max(0.0, drone_view_started - third_person_started)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    published_logs = ARTIFACTS / "logs"
    published_logs.mkdir(exist_ok=True)
    training = read_json(ARTIFACTS / "wavy_qr_training_metrics.json")
    demos: dict[str, object] = {}
    cards: list[str] = []
    for name in ALGORITHMS:
        start = sample_annulus_start(SPAWN_SEEDS[name])
        profile = random_wavy_motion_profile(TRAJECTORY_SEEDS[name])
        video = f"{name}_wavy_qr_tracking_landing_third_person.mp4"
        drone_video = f"{name}_wavy_qr_tracking_landing_drone_view.mp4"
        dual_video = f"{name}_wavy_qr_tracking_landing_dual_view.mp4"
        sync_offset = view_sync_offset(ARTIFACTS / f"{name}_wavy_qr_tracking_landing_sync.txt")
        csv_path = PROJECT_ROOT / "logs" / f"{name}_wavy_qr_tracking_landing.csv"
        log_path = PROJECT_ROOT / "logs" / f"{name}_wavy_qr_tracking_landing.log"
        # The site is served from ARTIFACTS, so publish the supporting evidence
        # beside the MP4s instead of linking outside the HTTP document root.
        for source in (csv_path, log_path):
            if source.exists():
                shutil.copy2(source, published_logs / source.name)
        run = flight_metrics(csv_path)
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        landed = "MAV_CMD_NAV_LAND sent" in log
        train = training.get("metrics", {}).get(name, {})
        demos[name] = {
            "spawn_seed": SPAWN_SEEDS[name], "trajectory_seed": TRAJECTORY_SEEDS[name],
            "start_x_m": start.x_m, "start_y_m": start.y_m, "start_radius_m": start.radius_m,
            "motion_profile": profile.__dict__, "third_person_video": video, "drone_view_video": drone_video, "dual_view_video": dual_video,
            "third_person_video_exists": (ARTIFACTS / video).exists(),
            "drone_view_video_exists": (ARTIFACTS / drone_video).exists(), "dual_view_video_exists": (ARTIFACTS / dual_video).exists(),
            "third_person_lead_trim_s": sync_offset, "views_synchronized": sync_offset is not None, "land_command_logged": landed, **run,
        }
        status = "LAND 명령 확인" if landed else "실비행 미검증"
        sync_note = (
            f"시점 동기화 완료 · 3인칭 선행 {sync_offset:.3f}초 자동 보정"
            if sync_offset is not None else "기존 녹화 · 시점 보정 메타데이터 없음"
        )
        cards.append(f'''<article class="card"><div class="title"><h2>{name.upper()}</h2><span class="{'ok' if landed else 'wait'}">{status}</span></div>
<h3>동시 시점 녹화 · 왼쪽 3인칭 / 오른쪽 드론 하향 추론</h3><p class="scenario">{sync_note}</p><video controls preload="metadata"><source src="{dual_video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video>
<div class="video-pair"><div><h3>원본 3인칭 Gazebo</h3><video controls preload="metadata"><source src="{video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video></div><div><h3>원본 드론 하향 카메라 · 추론</h3><video controls preload="metadata"><source src="{drone_video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video></div></div>
<dl><div><dt>RL 평가 성공률</dt><dd>{fmt(float(train['success_rate']) * 100, 1, '%') if train.get('success_rate') is not None else '—'}</dd></div><div><dt>종단 상대 오차</dt><dd>{fmt(train.get('mean_terminal_error_m'), 3, ' m')}</dd></div><div><dt>평균 카메라 오차</dt><dd>{fmt(run['mean_image_error'], 3)}</dd></div><div><dt>P95 카메라 오차</dt><dd>{fmt(run['p95_image_error'], 3)}</dd></div><div><dt>중심 정렬 프레임 비율</dt><dd>{fmt(float(run['centered_frame_rate']) * 100, 1, '%') if run['centered_frame_rate'] is not None else '—'}</dd></div><div><dt>추적 비행 시간</dt><dd>{fmt(run['duration_s'], 1, ' s')}</dd></div></dl>
<p class="scenario">출발 ({start.x_m:+.2f}, {start.y_m:+.2f}) m · 경로 시드 {profile.seed} · 기준 {profile.base_speed_mps:.2f} m/s · roll/pitch/heave ±{profile.wave_roll_deg:.1f}°/±{profile.wave_pitch_deg:.1f}°/±{profile.wave_heave_m * 100:.1f} cm</p>
<p><a href="{dual_video}">동시 시점 MP4</a> · <a href="{video}">3인칭 MP4</a> · <a href="{drone_video}">드론 시점 MP4</a> · <a href="logs/{name}_wavy_qr_tracking_landing.csv">비행 CSV</a> · <a href="logs/{name}_wavy_qr_tracking_landing.log">실행 로그</a></p></article>''')

    technical = '''<section class="technical"><div class="section-title"><span class="eyebrow">Learning interface</span><h2>상태 · 출력 · 보상 · 손실 설정</h2><p>세 알고리즘은 같은 관측·행동·보상 환경에서 학습했습니다. 정책 출력은 안전 영상 서보를 대체하지 않는 <strong>잔차</strong>이며, 배치 평가와 실제 비행에서는 각 축 ±0.25로 제한합니다.</p></div>
<article class="common"><h3>공통 상태 입력 (6)</h3><ol><li><code>qr_error_x, qr_error_y</code> — 화면 중심 대비 QR 정규화 오차</li><li><code>altitude</code> — 흔들리는 데크 기준 상대 고도</li><li><code>detected</code> — QR 검출 유효 플래그</li><li><code>target_velocity_x, target_velocity_y</code> — 현재 ENU 목표 속도</li></ol><h3>공통 정책 출력 (2)</h3><p><code>lateral_x, lateral_y ∈ [-1, 1]</code>: 시각 서보와 목표 속도 feed-forward 위에 더하는 수평 속도 잔차입니다. 환경에서는 <code>0.04 × action</code>, 온라인 제어에서는 <code>0.08 × clipped_action</code>으로 반영합니다.</p><h3>공통 보상</h3><p><code>7.0 × (이전 거리 − 현재 거리) − 0.030 × 거리 − 0.025 × ‖action‖² + 0.10(0.30 m 이내) + 0.35 × 하강</code>. 성공 착륙은 <code>+100</code>, 잘못된 착륙은 <code>−50</code>, 이탈은 <code>−20</code>입니다.</p></article>
<div class="method-grid"><article><h3>PPO</h3><p><b>손실:</b> clipped surrogate policy objective + value-function MSE + entropy 항. Stable-Baselines3 기본 <code>clip_range=0.2</code>, <code>vf_coef=0.5</code>, <code>ent_coef=0.0</code>을 사용합니다.</p><p><b>설정:</b> <code>lr=2.5e−4</code>, <code>γ=0.997</code>, <code>GAE λ=0.96</code>, rollout <code>512</code>, batch <code>128</code>.</p></article><article><h3>DDPG</h3><p><b>손실:</b> critic TD-target MSE와 actor의 <code>−Q(s, π(s))</code> 목적함수. 탐색은 행동 정규잡음으로 수행합니다.</p><p><b>설정:</b> <code>lr=3e−4</code>, <code>γ=0.997</code>, <code>τ=0.01</code>, replay <code>180,000</code>, warm-up <code>2,000</code>, batch <code>256</code>, noise <code>σ=0.18</code>, MLP <code>[256,256]</code>.</p></article><article><h3>SAC</h3><p><b>손실:</b> twin critic의 soft Bellman MSE와 entropy-augmented actor objective. 온도 α는 목표 엔트로피를 향해 자동 조정됩니다.</p><p><b>설정:</b> <code>lr=3e−4</code>, <code>γ=0.997</code>, <code>τ=0.01</code>, replay <code>180,000</code>, warm-up <code>2,000</code>, batch <code>256</code>, initial <code>ent_coef=0.02</code>, MLP <code>[256,256]</code>.</p></article></div></section>'''

    manifest = {
        "scenario": {
            "qr_size_m": QR_SIZE_M,
            "motion": f"seeded smooth curved trajectory with bounded speed and direction variation ({WAVY_QR_SPEED_MULTIPLIER:.0f}× accelerated)",
            "deck_motion": "seeded boat-like roll/pitch/yaw/heave",
            "control_preset": "1.50 m/s QR acquisition, 0.65 m/s visual centering, 0.13 m/s gated descent",
            "world": "aruco_moving_qr",
        },
        "training": training,
        "demos": demos,
    }
    (ARTIFACTS / "wavy_qr_demo_results.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    best = html.escape(str(training.get("best_algorithm", "pending")).upper())
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>흔들리는 QR 착륙 — PPO · DDPG · SAC</title>
<style>:root{{color-scheme:dark;--bg:#07101d;--panel:#101f34;--line:#2c496d;--cyan:#5bd5ff;--green:#60d8a3;--amber:#ffc35c}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#1a416e,transparent 39rem),var(--bg);color:#eef6ff;font:16px Inter,system-ui,sans-serif}}main{{width:min(1440px,calc(100% - 32px));margin:auto;padding:42px 0 64px}}h1{{font-size:clamp(2rem,5vw,3.8rem);letter-spacing:-.05em;margin:.15rem 0}}h2{{margin:0}}h3{{margin:.1rem 0 .55rem}}p{{color:#b8c9db;line-height:1.55}}.eyebrow{{color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}.lead{{max-width:900px;font-size:1.08rem}}.route{{display:grid;grid-template-columns:190px 1fr;gap:25px;align-items:center;margin:28px 0;padding:22px;border:1px solid var(--line);border-radius:19px;background:#0d192a}}.route svg{{width:100%;height:auto}}.route strong{{color:#fff}}.grid,.method-grid,.video-pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}.card,.common,.method-grid article{{padding:18px;border:1px solid var(--line);border-radius:19px;background:linear-gradient(145deg,#152945,#0a1423)}}.title{{display:flex;gap:12px;justify-content:space-between;align-items:center;margin-bottom:15px}}.title span{{font-size:.72rem;font-weight:700;border-radius:99px;padding:6px 9px}}.ok{{background:var(--green);color:#052817}}.wait{{background:var(--amber);color:#3d2700}}video{{width:100%;aspect-ratio:16/9;background:#02060a;border:1px solid #385678;border-radius:11px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:15px 0 7px}}dl div{{background:#081220aa;padding:9px;border-radius:9px}}dt{{font-size:.71rem;color:#9db2ca}}dd{{margin:3px 0 0;font-size:.92rem;overflow-wrap:anywhere}}.technical{{margin:42px 0}}.section-title{{margin-bottom:16px}}.common{{margin-bottom:18px}}.common ol{{color:#b8c9db;line-height:1.7;padding-left:1.3rem}}code{{color:#c5efff;background:#07101d;border:1px solid #294766;border-radius:5px;padding:.08rem .28rem;font:0.83em ui-monospace,SFMono-Regular,Consolas,monospace}}.scenario{{font-size:.84rem;margin:.8rem 0}}a{{color:var(--cyan)}}footer{{margin-top:32px;color:#9db2ca;font-size:.85rem}}@media(max-width:600px){{main{{width:calc(100% - 20px);padding-top:25px}}.route{{grid-template-columns:105px 1fr;gap:15px}}.grid,.method-grid,.video-pair{{grid-template-columns:1fr}}}}</style></head>
<body><main><div class="eyebrow">PX4 SITL · Gazebo Harmonic · 시드 기반 이동 플랫폼</div><h1>3배 속도 · 흔들리는 QR 착륙</h1><p class="lead">PPO, DDPG, SAC는 2–7 m 고리 안의 임의 위치에서 출발해 0.4 m QR를 탐색합니다. 이전 실험보다 수평 속도를 정확히 <strong>{WAVY_QR_SPEED_MULTIPLIER:.0f}배</strong>로 높인 재현 가능한 곡선 경로를 따라가며, 목표 속도·방향은 부드럽게 바뀝니다. QR 데크는 작은 배처럼 roll·pitch·yaw·상하 운동을 하고, QR 중심이 정렬될 때만 하강합니다.</p><p class="lead"><strong>빠른 접근 · 신중 착륙 프리셋:</strong> QR 미검출 탐색은 <strong>1.50 m/s</strong>, QR 검출 뒤 영상 중심화는 최대 <strong>0.65 m/s</strong>로 감속합니다. QR 오차가 0.13 이내인 프레임을 14회 연속 확인한 뒤에만 <strong>0.13 m/s</strong>로 하강합니다. 목표 위치·속도는 Gazebo 카메라의 시뮬레이션 시각을 기준으로 동기화합니다.</p>
<section class="route"><svg viewBox="0 0 190 110" aria-label="곡선 QR 이동 경로"><path d="M10 70C45 12 92 100 177 39" fill="none" stroke="#5bd5ff" stroke-width="4" stroke-linecap="round"/><path d="M170 29l12 10-14 7" fill="none" stroke="#5bd5ff" stroke-width="4"/><g transform="translate(92 58) rotate(-10)"><rect x="-18" y="-18" width="36" height="36" fill="white" stroke="#111" stroke-width="5"/><text y="5" text-anchor="middle" fill="#111" font-weight="900" font-size="14">QR</text></g><path d="M82 49q10 -13 20 0M82 67q10 13 20 0" fill="none" stroke="#ffc35c" stroke-width="2"/></svg><div><h2>하나의 공통 시드 모션 모델</h2><p>Gazebo 플러그인, RL 환경, 온라인 컨트롤러가 같은 제한된 사인 곡선 경로를 계산합니다. 급격한 변화 없이 전진 속도와 측면 방향이 바뀌며, 플랫폼에는 무제한 잡음 대신 결정론적 roll/pitch/yaw/heave가 적용됩니다. 컨트롤러는 순간 목표 속도를 feed-forward하고, 카메라 QR을 다시 중심에 맞추며, 정렬 게이트를 벗어나면 하강을 멈춥니다.</p><p>최고 오프라인 정책: <strong>{best}</strong> · 기법별 {html.escape(str(training.get('timesteps_per_algorithm','—')))} 스텝.</p></div></section>{technical}<section class="grid">{''.join(cards)}</section><footer>표시 지표는 오프라인 RL 성공률, 종단 오차, 평균·P95 카메라 오차, 중심 정렬 프레임 비율, 추적 비행 시간입니다. 영상은 Gazebo 3인칭과 드론 하향 카메라를 따로 녹화한 PX4 SITL 증거이며, 실제 하드웨어 비행 주장은 아닙니다.</footer></main></body></html>'''
    output = ARTIFACTS / "wavy_qr_landing_dashboard.html"
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
