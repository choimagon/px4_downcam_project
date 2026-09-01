#!/usr/bin/env python3
"""Build the Korean Unitree Go2-back-QR landing training/report dashboard."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "rl_training"
ALGORITHMS = ("ppo", "ddpg", "sac")
DIFFICULTIES = (("easy", "초급"), ("medium", "중급"), ("hard", "고급"))
INFERENCE_CSV_FIELDS = {
    "sim_time_s", "qr_error_m", "altitude_m", "detected",
    "qr_center_u", "qr_center_v", "qr_pnp_depth_m", "qr_center_rate_u", "qr_center_rate_v",
    "imu_impact_latched", "landing_retry_active", "landing_retry_count",
    "offline_sim_landing_skid_contacts", "offline_sim_landing_normal_force_n",
    "offline_sim_max_contact_penetration_m", "offline_sim_visual_contact_plane_error_m",
    "offline_sim_go2_path_distance_m", "offline_sim_pad_speed_mps", "offline_sim_go2_speed_mps",
    "offline_sim_go2_stance_slip_mps", "offline_sim_go2_base_height_m", "offline_sim_go2_tilt_deg",
    "offline_sim_go2_root_wrench_max_abs", "onnx_provider",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value: object, digits: int = 3, suffix: str = "") -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def equation(*lines: str) -> str:
    return '<div class="equation">\\[\\begin{aligned}' + r" \\ ".join(lines) + r"\end{aligned}\]</div>"


def flight(path: Path) -> dict[str, float | bool | None]:
    empty = {
        "schema_ok": False,
        "duration": None,
        "mean_error": None,
        "final_error": None,
        "altitude": None,
        "detection_rate": None,
        "frames": None,
        "offline_contacts": None,
        "offline_force": None,
        "offline_penetration": None,
        "offline_visual_plane_error": None,
        "offline_path": None,
        "offline_pad_speed": None,
        "offline_go2_speed": None,
        "offline_go2_slip": None,
        "offline_go2_height": None,
        "offline_go2_tilt": None,
        "offline_root_wrench": None,
    }
    if not path.exists():
        return empty
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        schema_ok = INFERENCE_CSV_FIELDS.issubset(set(reader.fieldnames or ()))
    if not rows:
        return empty
    error = [float(row["qr_error_m"]) for row in rows]
    detections = [float(row["detected"]) for row in rows]
    final = rows[-1]
    return {
        "schema_ok": schema_ok,
        "duration": float(final["sim_time_s"]), "mean_error": float(sum(error) / len(error)), "final_error": float(final["qr_error_m"]),
        "altitude": float(final["altitude_m"]), "detection_rate": float(sum(detections) / len(detections)), "frames": float(len(rows)),
        "offline_contacts": float(final.get("offline_sim_landing_skid_contacts", "nan")),
        "offline_force": float(final.get("offline_sim_landing_normal_force_n", "nan")),
        "offline_penetration": float(final.get("offline_sim_max_contact_penetration_m", "nan")),
        "offline_visual_plane_error": float(final.get("offline_sim_visual_contact_plane_error_m", "nan")),
        "offline_path": float(final.get("offline_sim_go2_path_distance_m", "nan")),
        "offline_pad_speed": float(final.get("offline_sim_pad_speed_mps", "nan")),
        "offline_go2_speed": float(final.get("offline_sim_go2_speed_mps", "nan")),
        "offline_go2_slip": float(final.get("offline_sim_go2_stance_slip_mps", "nan")),
        "offline_go2_height": float(final.get("offline_sim_go2_base_height_m", "nan")),
        "offline_go2_tilt": float(final.get("offline_sim_go2_tilt_deg", "nan")),
        "offline_root_wrench": float(final.get("offline_sim_go2_root_wrench_max_abs", "nan")),
    }


def live_cells(values: dict[str, float | bool | None]) -> str:
    return "".join((
        f"<td>{fmt(values['duration'], 1, ' s')}</td>", f"<td>{fmt(values['mean_error'], 4, ' m')}</td>",
        f"<td>{fmt(values['final_error'], 4, ' m')}</td>", f"<td>{fmt(values['altitude'], 3, ' m')}</td>",
        f"<td>{fmt(values['detection_rate'] * 100 if values['detection_rate'] is not None else None, 1, '%')}</td>",
        f"<td>{fmt(values['frames'], 0)}</td>",
    ))


def evaluation_cells(values: dict) -> str:
    return "".join((
        f"<td>{fmt(values.get('mean_reward'), 2)}</td>", f"<td>{fmt(values.get('std_reward'), 2)}</td>",
        f"<td>{fmt(float(values.get('success_rate', float('nan'))) * 100, 1, '%')}</td>",
        f"<td>{fmt(values.get('mean_terminal_error_m'), 4, ' m')}</td>", f"<td>{fmt(values.get('mean_episode_duration_s'), 2, ' s')}</td>",
        f"<td>{fmt(values.get('mean_episode_steps'), 1)}</td>",
    ))


def offline_evaluation_cells(values: dict) -> str:
    """Render MuJoCo-only diagnostics that are never policy observations."""
    penetration = values.get("offline_sim_mean_max_penetration_m")
    return "".join((
        f"<td>{fmt(float(values.get('offline_sim_go2_fall_rate', float('nan'))) * 100, 1, '%')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_landing_skid_contacts'), 2, ' / 2')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_landing_normal_force_n'), 2, ' N')}</td>",
        f"<td>{fmt(penetration * 1000 if isinstance(penetration, (float, int)) else None, 3, ' mm')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_go2_path_distance_m'), 2, ' m')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_pad_speed_mps'), 2, ' m/s')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_go2_speed_mps'), 2, ' m/s')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_go2_stance_slip_mps'), 3, ' m/s')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_go2_base_height_m'), 3, ' m')}</td>",
        f"<td>{fmt(values.get('offline_sim_mean_go2_tilt_deg'), 2, '°')}</td>",
        f"<td>{fmt(values.get('offline_sim_go2_root_wrench_max_abs'), 3)}</td>",
    ))


def video_panel(algorithm: str, difficulty: str, korean: str) -> str:
    stem = f"{algorithm}_go2_back_qr_onnx_{difficulty}_follow"
    video, snapshot, log = f"{stem}.mp4", f"{stem}.png", f"{stem}.csv"
    value = flight(ARTIFACTS / log)
    verified = all((ARTIFACTS / item).exists() for item in (video, snapshot, log)) and value["schema_ok"] is True
    return f'''<section class="video-panel"><div class="video-title"><h3>{korean} <span>({difficulty})</span></h3><b class="{'ok' if verified else 'wait'}">{'완료 · ONNX 추론' if verified else '산출물 대기'}</b></div>
<video controls preload="metadata"><source src="{video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video>
<a class="shot" href="{snapshot}"><img src="{snapshot}" alt="{algorithm.upper()} {korean} Go2 QR 착륙 추론"><span>좌측: Go2·X500 3인칭 / 우측: X500 하향 QR</span></a>
<div class="table-scroll"><table class="live"><thead><tr><th>시간</th><th>평균 오차</th><th>종단 오차</th><th>종단 상대고도</th><th>QR 검출률</th><th>프레임</th></tr></thead><tbody><tr>{live_cells(value)}</tr></tbody></table></div>
<p class="offline-note"><code>offline_sim_*</code> 물리 진단(정책 입력 아님): stock 스키드 레일 {fmt(value['offline_contacts'], 0, ' / 2')}, 최대 침투 {fmt(value['offline_penetration'] * 1000 if value['offline_penetration'] is not None else None, 3, ' mm')}, 시각/충돌 바닥면 오차 {fmt(value['offline_visual_plane_error'] * 1000 if value['offline_visual_plane_error'] is not None else None, 3, ' mm')}, 데크 속도 {fmt(value['offline_pad_speed'], 2, ' m/s')}, Go2 속도 {fmt(value['offline_go2_speed'], 2, ' m/s')}, root wrench 최대 절댓값 {fmt(value['offline_root_wrench'], 3)}</p>
<p class="links"><a href="{video}">MP4</a><a href="{snapshot}">PNG</a><a href="{log}">CSV</a></p></section>'''


def terrain_video_panel(record: dict) -> str:
    """Render one verified physical-terrain landing card from its manifest."""
    video = html.escape(str(record["video"]))
    snapshot = html.escape(str(record["snapshot"]))
    log = html.escape(str(record["csv"]))
    algorithm = html.escape(str(record["algorithm"]).upper())
    scenario = html.escape(str(record["korean_name"]))
    value = flight(ARTIFACTS / str(record["csv"]))
    verified = all((ARTIFACTS / item).exists() for item in (record["video"], record["snapshot"], record["csv"], record["receipt"])) and value["schema_ok"] is True
    summary = record.get("summary", {})
    return f'''<section class="video-panel"><div class="video-title"><h3>{algorithm} · {scenario}</h3><b class="{'ok' if verified else 'wait'}">{'완료 · 지형 ONNX 추론' if verified else '산출물 대기'}</b></div>
<video controls preload="metadata"><source src="{video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video>
<a class="shot" href="{snapshot}"><img src="{snapshot}" alt="{algorithm} {scenario} QR 착륙 추론"><span>좌측: 실제 지형 위 Go2·X500 3인칭 / 우측: X500 하향 QR</span></a>
<div class="table-scroll"><table class="live"><thead><tr><th>시간</th><th>종단 QR 오차</th><th>스키드</th><th>최대 관입</th><th>Go2 최대 기울기</th><th>최소 경계 여유</th><th>지형 높이</th></tr></thead><tbody><tr><td>{fmt(summary.get('duration_s'), 1, ' s')}</td><td>{fmt(summary.get('terminal_qr_error_m'), 4, ' m')}</td><td>{fmt(summary.get('max_skid_contacts'), 0, ' / 2')}</td><td>{fmt(float(summary.get('max_penetration_m', float('nan'))) * 1000, 3, ' mm')}</td><td>{fmt(summary.get('max_go2_tilt_deg'), 2, '°')}</td><td>{fmt(summary.get('min_terrain_boundary_clearance_m'), 3, ' m')}</td><td>{fmt(summary.get('terminal_terrain_height_m'), 3, ' m')}</td></tr></tbody></table></div>
<p class="offline-note">모든 접촉·관입·지형높이·경계여유 값은 <code>offline_sim_*</code> 사후 물리 진단입니다. Go2 본체와 QR 판의 보수적 외곽이 실제 collision course 안에 있었던 프레임만 검증을 통과하며, X500 정책 입력에는 Go2 상태나 지형 정답값이 들어가지 않습니다.</p>
<p class="links"><a href="{video}">MP4</a><a href="{snapshot}">PNG</a><a href="{log}">CSV</a><a href="{html.escape(str(record['receipt']))}">검증 영수증</a></p></section>'''


def terrain_section(report: dict) -> str:
    """Build a Korean explanation and verified terrain inference cards."""
    if report.get("status") != "passed":
        return ""
    scenarios = report.get("scenarios")
    demos = report.get("demonstrations")
    if not isinstance(scenarios, list) or not isinstance(demos, list):
        return ""
    panels: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        scenario_id = scenario.get("id")
        cards = "".join(
            terrain_video_panel(record)
            for record in demos
            if isinstance(record, dict) and record.get("scenario") == scenario_id
        )
        if not cards:
            continue
        panels.append(
            f'<details class="terrain-task" open><summary>{html.escape(str(scenario.get("korean_name", scenario_id)))} · PPO/DDPG/SAC 동기화 추론 3개</summary><div class="video-grid">{cards}</div></details>'
        )
    if not panels:
        return ""
    return '''<section class="section" id="terrain-landing-suite"><h2>검증 완료 실제 지형: 10% 경사 상승·하강·요철 1/2/3단계</h2>
<div class="callout"><strong>평지 영상을 대체하지 않는 별도 지형 검증 세트입니다.</strong> Go2는 0.22 kg QR 브래킷을 단 채 MuJoCo 충돌 지형을 실제 발로 밟습니다. X500 PPO·DDPG·SAC는 이전과 같은 카메라 기반 7D 입력만 받습니다. 보행 명령은 종전 경로의 3배로 설정했으며, 실제 코스 진행량은 각 영상의 CSV 사후 진단으로 확인합니다.</div>
<div class="split"><div class="formula-card"><h3>지형은 무엇으로 만들었나</h3><p>공개 경사는 길이 16 m의 <strong>10% grade (5.71°) 회전 box collision surface</strong>입니다. 요철은 폭 2.4 m의 <strong>연속 MuJoCo heightfield 충돌면</strong>이며, 수직 타일 벽이 아닌 걸을 수 있는 연속 기복으로 모델링했습니다. 높이 진폭은 <strong>24 / 48 / 80 mm</strong>입니다. 저대비 흙·잔디 입자 재질은 같은 충돌면의 시각 재질일 뿐이고, 별도 돌·카메라 전용 바닥·발 통과 오브젝트는 없습니다.</p><p>Go2·QR 판의 보수적 물리 외곽이 이 finite collision surface 안에 있을 때만 착륙 성공을 허용합니다. 전방 경계에는 보행 명령을 감속하고, 측면을 포함해 한 번이라도 이탈하면 성공은 잠기며 해당 replay는 공개 검증에서 제외됩니다.</p></div><div class="formula-card"><h3>무엇을 검증했나</h3><p>Go2 저수준 PPO 후보의 상태 계약은 IMU·관절·속도명령·직전 행동 450D입니다. 다만 이 지형 공개 세트는 후보가 낙상 기준을 통과하지 못했으므로, 실제 IMU·자체 odometry 되먹임의 물리 기준 트로트를 사용합니다. 지형 높이·QR pose·드론 상태는 Go2 제어 입력에 넣지 않고, Go2 root wrench는 항상 0입니다.</p><p>아래 영상은 결합 Go2+X500 물리 장면에서 고정 시드를 사전 재생해 <strong>Go2 낙상 0, 최고 기울기 40° 이하, Go2·QR 판 코스 이탈 0, 두 스키드 물리 접촉, root wrench 0</strong>을 통과한 재현입니다. 이는 넓은 무작위화 강건성 인증이 아니라 공개된 고정 시드 재생 검증입니다.</p></div></div>
<p>요철 영상의 좌측 큰 화면은 Go2 발과 실제 높이장을 가까이 보는 3인칭이고, 같은 시점의 전체 X500 3인칭은 노란 테두리 삽입 화면으로 동기화해 함께 보입니다. 우측은 X500 하향 카메라입니다. 경사 상승·하강과 요철 1/2/3단계별로 PPO·DDPG·SAC를 실행했으므로 아래에 총 15개 MP4가 있습니다.</p>''' + "".join(panels) + "</section>"


def discrete_gravel_video_panel(record: dict) -> str:
    """Render one verified discrete-stone gravel landing card."""
    video = html.escape(str(record.get("video", "")))
    snapshot = html.escape(str(record.get("snapshot", "")))
    log = html.escape(str(record.get("csv", "")))
    receipt = html.escape(str(record.get("receipt", "")))
    algorithm = html.escape(str(record.get("algorithm", "—")).upper())
    korean = html.escape(str(record.get("korean_name", "—")))
    summary = record.get("summary", {})
    required = (record.get("video"), record.get("snapshot"), record.get("csv"), record.get("receipt"))
    verified = all(isinstance(item, str) and (ARTIFACTS / item).exists() for item in required)
    return f'''<section class="video-panel"><div class="video-title"><h3>{algorithm} · {korean}</h3><b class="{'ok' if verified else 'wait'}">{'완료 · 실제 돌 물리 재생' if verified else '산출물 대기'}</b></div>
<video controls preload="metadata"><source src="{video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video>
<a class="shot" href="{snapshot}"><img src="{snapshot}" alt="{algorithm} {korean} 실제 돌 자갈길 위 Go2와 X500 착륙"><span>좌측: Go2·X500 근접 3인칭 / 우측: X500 하향 QR / 삽입: 전체 3인칭</span></a>
<div class="table-scroll"><table class="live"><thead><tr><th>명령 속도</th><th>QR 종단 오차</th><th>Go2 이동창 속도</th><th>Go2 경로</th><th>최대 기울기</th><th>최소 코스 여유</th><th>스키드</th><th>최대 관입</th></tr></thead><tbody><tr><td>{fmt(record.get('command_speed_mps'), 2, ' m/s')}</td><td>{fmt(summary.get('terminal_qr_error_m'), 4, ' m')}</td><td>{fmt(summary.get('terminal_motion_window_speed_mps'), 3, ' m/s')}</td><td>{fmt(summary.get('terminal_go2_path_distance_m'), 2, ' m')}</td><td>{fmt(summary.get('max_go2_tilt_deg'), 2, '°')}</td><td>{fmt(summary.get('min_terrain_boundary_clearance_m'), 3, ' m')}</td><td>{fmt(summary.get('max_skid_contacts'), 0, ' / 2')}</td><td>{fmt(float(summary.get('max_penetration_m', float('nan'))) * 1000, 3, ' mm')}</td></tr></tbody></table></div>
<p class="offline-note">고정 시드 한 번을 사전 선언한 gate로 재생한 결과입니다. <code>Go2 낙상·코스 이탈·정지·root wrench</code>가 하나라도 발생하거나 1초 이동창 속도가 <code>0.12 m/s</code> 미만이면 성공을 금지했습니다. 표의 Go2·접촉 수치는 정책 입력이 아닌 사후 물리 검증값입니다.</p>
<p class="links"><a href="{video}">MP4</a><a href="{snapshot}">PNG</a><a href="{log}">CSV</a><a href="{receipt}">검증 영수증</a></p></section>'''


def gravel_landing_section(report: dict) -> str:
    """Render the nine declared-seed moving-landing results on real rocks."""
    demonstrations = report.get("demonstrations")
    terrain = report.get("terrain", {})
    if report.get("status") != "passed" or not isinstance(demonstrations, list):
        return '''<section class="section" id="gravel-landing-suite"><h2>실제 개별 돌 자갈길 이동 착륙</h2><p>자갈길 충돌 지형과 3단계 재생 산출물을 생성 중입니다.</p></section>'''
    panels: list[str] = []
    for difficulty, korean in DIFFICULTIES:
        records = [
            item for item in demonstrations
            if isinstance(item, dict) and item.get("difficulty") == difficulty
        ]
        records.sort(key=lambda item: ALGORITHMS.index(item.get("algorithm")) if item.get("algorithm") in ALGORITHMS else len(ALGORITHMS))
        if not records:
            continue
        speed = fmt(records[0].get("command_speed_mps"), 2, " m/s")
        cards = "".join(discrete_gravel_video_panel(record) for record in records)
        panels.append(
            f'<details class="terrain-task" open><summary>{korean} · Go2 전진 명령 {speed} · PPO/DDPG/SAC 3개</summary><div class="video-grid">{cards}</div></details>'
        )
    if not panels:
        return ""
    length = fmt(terrain.get("length_m"), 1, " m")
    width = fmt(terrain.get("width_m"), 1, " m")
    rocks = fmt(terrain.get("individual_collision_rocks"), 0)
    grade = fmt(terrain.get("soil_grade_percent"), 1, "%")
    undulation = fmt(terrain.get("soil_undulation_amplitude_mm"), 0, " mm")
    preview = "go2_discrete_gravel_road_preview.png"
    preview_html = (
        f'<a class="shot" href="{preview}"><img src="{preview}" alt="개별 돌이 깔린 완만한 자갈길 렌더"><span>실제 충돌 자갈길 렌더 미리보기</span></a>'
        if (ARTIFACTS / preview).exists() else ""
    )
    training_note = html.escape(str(report.get("locomotion", {}).get("training_note", "")))
    return f'''<section class="section" id="gravel-landing-suite"><div class="section-kicker">CURRENT PHYSICAL EXPERIMENT · 9 VERIFIED MP4</div><h2>실제 개별 돌 자갈길 · 초급/중급/고급 이동 착륙</h2>
<div class="callout"><strong>같은 자갈길·같은 드론 시작 조건에서 속도만 바꿨습니다.</strong> 초급/중급/고급 Go2 전진 명령은 각각 <strong>0.58 / 0.75 / 0.92 m/s</strong>입니다. Go2가 멈추거나 넘어지면, QR 데크·Go2 보수 외곽 중 하나라도 길 밖으로 나가면, 또는 실제 스키드 두 개가 QR 판에 닿지 않으면 그 재생은 착륙 성공으로 기록되지 않습니다.</div>
<div class="split"><div class="formula-card"><h3>길은 실제 개별 충돌 돌입니다</h3><p>MuJoCo 물리 코스는 <strong>{length} × {width}</strong>입니다. 다져진 흙 기반에는 <strong>정적 타원체 돌 {rocks}개</strong>가 각각 독립 collision geometry로 묻혀 있고, 바닥 자체도 <strong>{grade} 완만 경사</strong>와 <strong>{undulation}</strong> 미세 기복을 가집니다. 돌·흙·QR 판·Go2 발·X500 스키드는 같은 MuJoCo 충돌계에서 계산됩니다.</p><p>이전 연속 요철 재생은 이 공개 구역에 넣지 않았습니다. 아래는 개별 돌 자갈길 결과만 표시합니다.</p>{preview_html}</div><div class="formula-card"><h3>착륙을 막는 사전 선언 gate</h3><p>각 PPO/DDPG/SAC 영상은 난이도별로 한 개의 고정 시드만 사용했습니다. <strong>낙상 0</strong>, <strong>코스 이탈 0</strong>, <strong>Go2 root 외력 0</strong>, <strong>최대 기울기 ≤ 35°</strong>, <strong>스키드 2/2</strong>, <strong>수치 관입 ≤ 2.1 mm</strong>, 그리고 착륙 직전 1초 이동창 <strong>≥ 0.12 m/s</strong>를 모두 검사합니다. 따라서 Go2가 멈췄을 때의 착륙은 통과할 수 없습니다.</p><p>{training_note}</p></div></div>
<p>모든 MP4는 동일 MuJoCo state에서 좌측 근접 3인칭, 우측 X500 하향 QR, 삽입 전체 3인칭을 동기화해 캡처했습니다. 프로펠러는 영상에서 회전하며, X500·Go2·QR 판이 동시에 화면에 보이도록 카메라를 배치했습니다.</p>{''.join(panels)}</section>'''


def algorithm_card(name: str, metrics: dict, onnx: dict) -> str:
    policy = metrics.get("metrics", {}).get(name, {})
    held_out = policy.get("held_out", {})
    rows = "".join(f"<tr><th>{label}</th>{evaluation_cells(held_out.get(key, {}))}</tr>" for key, label in DIFFICULTIES)
    offline_rows = "".join(f"<tr><th>{label}</th>{offline_evaluation_cells(held_out.get(key, {}))}</tr>" for key, label in DIFFICULTIES)
    panels = "".join(video_panel(name, key, label) for key, label in DIFFICULTIES)
    max_diff = fmt(onnx.get("validation_max_abs_action_error"), 2)
    eval_episodes = html.escape(str(metrics.get("eval_episodes_per_difficulty", "—")))
    return f'''<article class="algorithm-card"><div class="algorithm-heading"><div><p class="eyebrow">{name.upper()} POLICY</p><h2>{name.upper()} · Go2 이동 QR 착륙</h2></div><span class="badge">ONNX action diff ≤ {max_diff}</span></div>
<p class="sub">각 난이도 평가는 독립 {eval_episodes} 에피소드 평균이며, 영상은 고정 시드 1회 재현입니다. ONNX는 SB3 결정론 행동과 직접 비교 검증했습니다.</p>
<div class="video-grid">{panels}</div><h3 class="table-heading">핵심 정량 평가 — 6개 지표</h3><div class="table-scroll"><table><thead><tr><th>분포</th><th><code>mean_reward</code></th><th><code>std_reward</code></th><th><code>success_rate</code></th><th><code>mean_terminal_error_m</code></th><th><code>mean_episode_duration_s</code></th><th><code>mean_episode_steps</code></th></tr></thead><tbody><tr><th>학습</th>{evaluation_cells(policy.get('training', {}))}</tr>{rows}</tbody></table></div>
<details><summary><code>offline_sim_*</code> 물리 진단 보기 — 정책 센서가 아님</summary><div class="table-scroll"><table><thead><tr><th>분포</th><th>Go2 전도율</th><th>stock 스키드 레일-판 접촉</th><th>정상력</th><th>최대 침투</th><th>Go2 경로</th><th>데크 속도</th><th>Go2 속도</th><th>접지 미끄럼</th><th>몸통 높이</th><th>몸통 기울기</th><th>root wrench max |component|</th></tr></thead><tbody><tr><th>학습</th>{offline_evaluation_cells(policy.get('training', {}))}</tr>{offline_rows}</tbody></table></div></details></article>'''


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metrics = read_json(ARTIFACTS / "go2_back_qr_training_metrics.json")
    loco = read_json(ARTIFACTS / "go2_legged_loco_metrics.json")
    onnx_models = {item.get("algorithm"): item for item in read_json(ARTIFACTS / "go2_back_qr_onnx_models.json").get("models", [])}
    best = html.escape(str(metrics.get("best_algorithm", "—")).upper())
    timesteps = html.escape(str(metrics.get("timesteps_per_algorithm", "—")))
    source = html.escape(str(metrics.get("go2_model_source", "unitreerobotics/unitree_mujoco")))
    cards = "".join(algorithm_card(name, metrics, onnx_models.get(name, {})) for name in ALGORITHMS)
    gravel_html = gravel_landing_section(read_json(ARTIFACTS / "go2_discrete_gravel_landing_suite.json"))
    loco_eval = loco.get("evaluation", {})
    loco_error = fmt(loco_eval.get("mean_terminal_velocity_error_mps"), 3, " m/s")
    loco_yaw_error = fmt(loco_eval.get("mean_terminal_yaw_rate_error_radps"), 3, " rad/s")
    loco_up = fmt(loco_eval.get("mean_terminal_base_up"), 3)
    loco_fall = fmt(float(loco_eval.get("fall_rate", float("nan"))) * 100, 1, "%")
    loco_path = fmt(loco_eval.get("mean_path_distance_m"), 2, " m")
    loco_slip = fmt(loco_eval.get("mean_stance_foot_slip_mps"), 3, " m/s")
    loco_height = fmt(loco_eval.get("mean_base_height_m"), 3, " m")
    loco_match = fmt(float(loco_eval.get("mean_gait_contact_match", float("nan"))) * 100, 1, "%")
    loco_root_wrench = fmt(loco_eval.get("root_wrench_max_abs"), 3)
    profiles = metrics.get("profiles", {})
    difficulty_rows = "".join(
        "<tr>"
        f"<th>{label}</th>"
        f"<td>{fmt(profiles.get(key, {}).get('path_speed'), 2, ' m/s')}</td>"
        f"<td>{fmt(math.degrees(float(profiles.get(key, {}).get('turn_angle_rad', float('nan')))), 1, '°')}</td>"
        f"<td>{fmt(profiles.get(key, {}).get('turn_frequency_hz'), 2, ' Hz')}</td>"
        "</tr>"
        for key, label in DIFFICULTIES
    )

    state_formula = equation(
        r"o_t &= [u_{qr},v_{qr},\min(1,z_{pnp}/8),\delta_{qr},\operatorname{clip}(\dot u_{qr}/3),\operatorname{clip}(\dot v_{qr}/3),\operatorname{clip}(v_{z,est}/3)]\in[-1,1]^7",
        r"\dot{\boldsymbol c}_{qr,t} &= \operatorname{LPF}\!\left((\boldsymbol c_{qr,t}-\boldsymbol c_{qr,t-1})/\Delta t_{cam}\right),\quad \Delta t_{cam}=1/30\ \mathrm{s},\quad \Delta t_{est}=1/50\ \mathrm{s}",
        r"a_t &= [a_x,a_y]\in[-1,+1]^2",
        r"\hat v_{qr}^{cam} &= \operatorname{LPF}\!\left(v_{est}+\frac{d}{dt}(R_{WB}t_{B}^{qr})\right)",
        r"R_{W,deck}^{*} &= R_{WB,est}R_{BC}^{cal}R_{CM}^{PnP}\quad(30\ \mathrm{Hz\ noisy\ sample/hold})",
        r"\sigma_{R,PnP}^{deg}(z) &= 0.15+0.03z_{pnp}\quad(\mathrm{rotation\ vector\ Gaussian,\ per\ axis\ clipped\ at}\ 3\sigma)",
        r"\lambda(h)&=\operatorname{clip}_{[0,1]}((h-0.45)/(1.20-0.45)),\quad \hat e=e_{xy}/\lVert e_{xy}\rVert",
        r"a_{safe}^{xy}&=0.001\lambda(h)\max(0,\hat e^{\mathsf T}R_{WB,xy}a_t)\hat e",
        r"v_{xy}^{*} &= \operatorname{clip}_{3.6}\!\left(\hat v_{qr,xy}^{cam}+0.92(R_{WB}t_B^{qr})_{xy}+a_{safe}^{xy}\right)\quad(\delta_{qr}=1)",
        r"\delta_{qr}=0 &\Rightarrow v_{xy}^{*}=v_{corridor\ sweep}(p_{est},t),\quad v_z^*=\operatorname{clip}_{[-0.8,1.2]}(1.8(z_{search}-z_{est}))",
    )
    reward_formula = equation(
        r"r_t &= 7(d_{t-1}-d_t)-0.035d_t-1.0\lVert a_t\rVert_2^2+0.12\mathbb{1}[d_t<0.30]",
        r"&\quad+110\mathbb{1}[\mathrm{stable\ landing}]-55\mathbb{1}[\mathrm{hard\ landing}]",
        r"&\quad-80\mathbb{1}[\mathrm{Go2\ fall}]-25\mathbb{1}[\mathrm{out\ of\ bounds}]",
        r"\mathrm{stable\ landing} &= \mathbb{1}[c_{skid}^{sim}=2\ \land\ h\le0.245\ \land\ d_t<\mathrm{landing\ limit}\ \land\ \lVert v_{drone}-v_{deck}\rVert<0.40]",
    )
    imu_landing_formula = equation(
        r"\epsilon_{z,t}^{imu} &= f_{B,z,t}^{imu}-f_{B,z,t}^{cmd},\qquad f_{B,z,t}^{cmd}=\left(R_{WB}^{\mathsf T}F_{cmd}/m\right)_z",
        r"I_t &= \mathbb{1}[\mathrm{final\ visual\ memory}\ \land\ h_{vis}\le0.25\ \land\ \epsilon_{z,t}^{imu}\ge4.0\ \land\ |v_{z,est}|\le0.45]",
        r"I_t=1 &\Rightarrow F_z^{settle}=0.88\,mg\ \mathrm{for}\ 0.35\ \mathrm{s};\quad \mathrm{no\ offline\ stable\ terminal}\Rightarrow v_z^{rel}=+0.45\ \mathrm{m/s}\ \mathrm{until}\ z_{pnp}\ge0.30\ \mathrm{m}",
        r"\delta_{qr}=0\ \land\ \mathrm{final\ visual\ memory} &\Rightarrow v_z^{rel}=-(0.16\ \mathrm{m/s});\quad n_{retry}\ge3\Rightarrow -(0.14\ \mathrm{m/s})",
        r"h_{skid}\simeq0.2260\ \mathrm{m} &\Rightarrow z_{pnp}\simeq0.161\ \mathrm{m}>z_{near}=0.10\ \mathrm{m}",
    )
    contact_formula = equation(
        r"c_{skid}^{sim} &= |\{s\in\{left,right\}: s\leftrightarrow\mathrm{landing\ surface}\}|",
        r"F_N^{sim} &= \sum\limits_{\mathrm{skid/surface\ contacts}} \max(0,f_{normal})",
        r"p_{max}^{sim} &= \max\limits_{\mathrm{skid/surface\ contacts}}\max(0,-\mathrm{contact.dist})",
        r"p_{max}^{sim} &\le 0.002\ \mathrm{m}\quad(2\ \mathrm{mm\ offline\ physics\ gate})",
        r"e_{plane}^{sim} &= |z_{sole}^{collision}-z_{sole}^{visual}|\le0.001\ \mathrm{m}",
        r"\mu &= (0.95,0.015,0.001),\quad \mathrm{solref}=(0.008,1),\quad \mathrm{solimp}=(0.96,0.99,0.001)",
    )
    loss_formula = equation(
        r"L_{PPO} &= -\mathbb{E}[\min(\rho_tA_t,\operatorname{clip}(\rho_t,0.80,1.20)A_t)]+0.50\mathbb{E}[(V_\theta-R_t)^2]",
        r"L_{DDPG,Q} &= \mathbb{E}[(Q_\phi(o_t,a_t)-y_t)^2],\qquad L_{DDPG,\mu}=-\mathbb{E}[Q_\phi(o_t,\mu_\theta(o_t))]",
        r"L_{SAC,Q_i} &= \mathbb{E}[(Q_{i,\phi}(o_t,a_t)-y_t)^2],\qquad L_{SAC,\pi}=\mathbb{E}[\alpha\log\pi_\theta-\min(Q_1,Q_2)]",
    )
    loco_formula = equation(
        r"o_t^{loco} &= [\omega_b,\mathrm{rpy}_b,v_{cmd},q-q_0,\dot q,a_{t-1}]\in\mathbb{R}^{45},\qquad \bar o_t=[o_t^{loco},o_{t-1}^{loco},\ldots,o_{t-9}^{loco}]\in\mathbb{R}^{450}",
        r"x_{foot}^{stance}(t) &= x_{front}-v_{leg}t,\qquad z_{foot}^{stance}=-0.300\ \mathrm{m},\qquad \beta=0.58",
        r"q_t^*&=q_t^{ref}+0.18\,g\,\operatorname{clip}(a_t^{PPO},-1,1),\quad g=0.50\ (\mathrm{flat});\quad g=0\ (\mathrm{published\ terrain\ reference\ gait})",
        r"\tau &= 60(q_t^*-q)+2(\dot q_t^{ref}-\dot q),\qquad \boldsymbol w_{root}^{applied}=\boldsymbol 0_{6}",
        r"\Delta t_{physics}&=0.005\ \mathrm{s},\quad \Delta t_{control}=4\Delta t_{physics}=0.020\ \mathrm{s},\quad d_{actuator}=4\ \mathrm{control\ steps}",
    )
    route_formula = equation(
        r"\psi(t) &= A_{turn}[\sin(2\pi f_{turn}t)+0.22\sin(\pi f_{turn}t+0.35)]",
        r"v(t) &= v_{nom}[1-\tfrac12m_{speed}(1-\cos(0.37\cdot2\pi f_{turn}t))]",
        r"[v_{Go2,x},v_{Go2,y}] &= v(t)[\cos\psi(t),\sin\psi(t)]",
    )
    policy_input_rows = "".join(
        f'''<tr data-observation="{name}"><th><span class="input-index">{index}</span><code>{name}</code></th><td>{meaning}</td><td>{raw_source}</td><td>{preprocess}</td><td>{timing}</td><td>{hardware}</td></tr>'''
        for index, name, meaning, raw_source, preprocess, timing, hardware in (
            (
                0,
                "qr_center_u",
                "QR 네 모서리 평균점의 가로 위치입니다. 화면 중심은 0, 왼쪽 끝은 -1, 오른쪽 끝은 +1입니다. 원본 픽셀이나 영상 자체가 신경망에 들어가는 것은 아닙니다.",
                "MuJoCo 센서 에뮬레이터는 카메라 기준 QR 이동벡터와 수평 FOV로 중심을 투영하고, PnP 이동 잡음과 중심 잡음을 적용합니다. 실기에서는 왜곡을 보정한 네 코너의 x 픽셀 평균 <code>c_x</code>를 구합니다.",
                "1280 px 폭 기준 <code>u=(c_x-640)/640</code>로 바꾸고 <code>[-1,1]</code>로 자릅니다. 별도 단위가 없는 정규화 화면좌표입니다.",
                "카메라 캐시가 갱신될 때만 새 값이 생기고 그 사이에는 유지됩니다. QR 미검출이면 0입니다.",
                "하향 RGB <code>/down_camera/image_raw</code> + QR 코너 검출. 중심 검출 코드는 있으나 이 7D 실기 어댑터에는 아직 연결되지 않았습니다.",
            ),
            (
                1,
                "qr_center_v",
                "QR 중심의 세로 위치입니다. 화면 중심은 0이고 위·아래 가장자리로 갈수록 절댓값이 1에 가까워집니다.",
                "MuJoCo에서는 카메라 기준 y 이동과 수직 FOV로 투영합니다. 실기에서는 왜곡 보정한 네 코너의 y 픽셀 평균 <code>c_y</code>를 사용합니다.",
                "960 px 높이 기준 <code>v=(c_y-480)/480</code>로 만들고 <code>[-1,1]</code>로 자릅니다. 카메라 optical 축의 세로 부호가 학습 규약과 같은지는 설치 후 표적 이동 시험으로 확인해야 합니다.",
                "카메라 캐시 sample-and-hold이며 미검출이면 0입니다.",
                "같은 하향 RGB 프레임의 QR 코너. 렌즈 보정 뒤 QR을 영상 위·아래로 움직여 실제 부호를 반드시 검증합니다.",
            ),
            (
                2,
                "qr_pnp_depth",
                "카메라 광축을 따라 QR까지 얼마나 떨어졌는지를 나타냅니다. 수직고도나 3차원 거리의 크기가 아니라 PnP 이동벡터의 전방 깊이입니다.",
                "MuJoCo에서는 카메라 상대 이동의 전방 성분을 detector/PnP 값처럼 합성합니다. 실기에서는 보정된 카메라 내부행렬 <code>K</code>, 왜곡계수 <code>D</code>, 23 cm QR 네 코너로 <code>solvePnP</code>를 실행합니다.",
                "원시 깊이 <code>z</code>는 m 단위 양수입니다. 모델에는 <code>min(1,z/8)</code>을 넣으므로 2.4 m는 0.30이고 8 m 이상은 1에서 포화합니다.",
                "네 코너와 PnP가 유효한 카메라 프레임에서 갱신합니다. 미검출이면 0입니다.",
                "<code>/down_camera/image_raw</code>와 <code>/down_camera/camera_info</code>, 정확히 측정한 QR 검은 인쇄 영역 한 변 0.23 m가 필요합니다. 현재 실기 코드에는 solvePnP 연결이 없습니다.",
            ),
            (
                3,
                "qr_detected",
                "앞의 0이 ‘QR이 화면 정중앙에 있음’인지 ‘QR이 아예 없음’인지 구분하는 유효성 마스크입니다.",
                "시뮬레이션은 깊이 0.10–12 m, 화면 안, 투영된 QR 최소 변 20 px 이상, 표식이 카메라를 향함, dropout 아님을 모두 검사합니다.",
                "유효하면 1.0, 하나라도 실패하면 0.0입니다. 실기에서는 올바른 QR payload, 네 코너, PnP 품질 검사를 모두 통과했을 때만 1로 둡니다.",
                "카메라 캐시에 유지됩니다. 난이도 공통 dropout 확률은 정책 step마다 1%로 설정되어 다음 구간의 카메라 갱신에 반영됩니다.",
                "QR detector의 decode 결과와 코너/PnP 유효성 검사. 단순히 사각형 하나가 보였다는 이유로 1을 주면 안 됩니다.",
            ),
            (
                4,
                "qr_center_rate_u",
                "QR 중심이 화면에서 가로로 얼마나 빨리 움직이는지 나타냅니다. 단위는 픽셀/s가 아니라 정규화 화면좌표/s입니다.",
                "연속으로 유효한 두 카메라 측정의 <code>(u_t-u_prev)/실제 timestamp 차이</code>로 구합니다. 두 중심 측정의 잡음도 이 차분에 자연스럽게 포함됩니다.",
                "원시 변화율을 ±5/s로 자르고 <code>0.65×이전 필터값 + 0.35×새 차분값</code>으로 저역통과한 뒤, 3으로 나누어 다시 <code>[-1,1]</code>로 자릅니다.",
                "카메라 측정 시에만 갱신합니다. 최초 검출, 재검출 첫 프레임, 미검출 상태에서는 0입니다.",
                "QR 중심과 캡처 timestamp를 보존해 같은 clip/저역통과를 구현합니다. 프레임 번호가 아니라 실제 timestamp로 나눠야 합니다.",
            ),
            (
                5,
                "qr_center_rate_v",
                "QR 중심의 세로 화면 이동속도입니다. 드론의 수직속도와 다른 값이며, 영상 안에서 표적이 움직이는 속도입니다.",
                "연속 유효 중심의 <code>(v_t-v_prev)/실제 timestamp 차이</code>입니다.",
                "가로 변화율과 동일하게 원시 ±5/s clip, 0.65/0.35 저역통과, 마지막 <code>rate/3</code> 정규화를 적용합니다.",
                "카메라 측정 시 갱신하며 최초·재검출 첫 프레임과 미검출 때 0입니다.",
                "QR 중심과 카메라 timestamp에서 계산합니다. 영상의 세로축 부호가 맞는지 먼저 검증해야 합니다.",
            ),
            (
                6,
                "drone_vertical_velocity",
                "QR이나 Go2가 아니라 드론 자신의 세계 수직속도입니다. 이 구현의 학습 좌표계는 위쪽이 양수이므로 하강 중에는 음수입니다.",
                "MuJoCo의 명시적 <code>drone_gps_velocity</code> 센서 world-Z에 표준편차 0.014 m/s 잡음을 넣은 PX4 상태추정 surrogate를 사용합니다.",
                "위쪽 양수 속도 <code>v_z(up)</code>를 3 m/s로 나누고 <code>[-1,1]</code>로 자릅니다. PX4 NED는 아래쪽 양수이므로 실기 변환은 반드시 <code>v_z(up) = -v_z(NED)</code>입니다.",
                "추정기 캐시는 정확히 50 Hz로 갱신되고 정책이 읽을 때까지 유지합니다. QR이 사라져도 이 일곱 번째 값만은 0으로 지우지 않습니다.",
                "PX4 <code>/fmu/out/vehicle_local_position</code>의 <code>vz</code>, <code>v_z_valid</code>, timestamp를 확인합니다. stale 또는 invalid면 추론을 계속하지 말고 failsafe로 전환해야 합니다.",
            ),
        )
    )
    input_guide_html = f'''<section class="section input-guide" id="drone-input-guide" data-real-adapter-status="not-implemented" data-source-frame="PX4-NED" data-vz-conversion="negate" data-policy-camera-resolution="1280x960" data-video-down-view-resolution="640x720"><div class="section-kicker">SENSOR → FLOAT32[7] → ONNX</div><h2>드론 정책 입력 7개: 수식 전에 실제 데이터부터</h2>
<div class="callout"><strong>정책은 하단 카메라 원본 영상을 직접 보지 않습니다.</strong> 카메라에서 QR 네 모서리를 찾아 만든 숫자 6개와 드론 자체 PX4 수직속도 1개를 정확한 순서의 <code>float32[7]</code> 배열로 묶어 0.1초마다 PPO·DDPG·SAC ONNX에 전달합니다. 착륙다리 접촉, Go2 위치·속도·경로, QR의 MuJoCo 정답 좌표는 이 배열에 들어가지 않습니다.</div>
<div class="truth-warning"><strong>현재 구현 범위:</strong> 아래 7D 경로는 MuJoCo 학습·ONNX 추론에서 구현되고 검증되었습니다. 기존 ROS 추론기는 다른 6D 관측을 사용하므로 <strong>실기 어댑터는 현재 구현되지 않았습니다</strong>. 아래의 ROS 2/PX4 항목은 같은 7D 계약을 실제 X500에 연결하기 위한 구체적인 구현 설계이지, 이미 비행 검증을 끝낸 연결이라는 뜻이 아닙니다.</div>
<h3 id="camera-to-vector-pipeline">카메라 프레임에서 7D 벡터까지</h3><ol class="pipeline"><li><b>하향 프레임 취득</b><span>실기 카메라 프레임·timestamp와 CameraInfo를 함께 받습니다.</span></li><li><b>QR 검출</b><span>올바른 payload와 네 모서리를 얻고 왜곡을 보정합니다.</span></li><li><b>pose 계산</b><span>23 cm QR와 보정값으로 solvePnP를 실행해 전방 깊이와 상대 pose를 얻습니다.</span></li><li><b>6개 카메라 값</b><span>중심 u/v, 깊이, detected, 중심 변화율 u/v를 계산·필터링합니다.</span></li><li><b>드론 자체 속도 결합</b><span>PX4 local velocity의 NED 부호·유효성·timestamp를 처리합니다.</span></li><li><b>10 Hz 추론</b><span>최신 캐시를 exact-order float32[7]로 만들고 ONNX에 한 번 전달합니다.</span></li></ol>
<p class="clock-note"><strong>실제 구현 시계:</strong> 정책은 100 ms(10 Hz), PX4 추정 surrogate는 20 ms(50 Hz)입니다. 카메라 계약은 명목 30 Hz이지만 현재 MuJoCo의 5 ms 물리 격자와 <code>next = now + 1/30</code> 예약 때문에 새 캐시는 실제 35 ms 간격, 약 28.57 Hz로 갱신됩니다. 정책은 각 센서의 최신 값을 sample-and-hold로 읽습니다.</p>
<h3>7개 입력 상세 사전</h3><div class="table-scroll"><table class="input-table"><thead><tr><th># · 모델 필드</th><th>무슨 값인가</th><th>원시값을 어떻게 얻나</th><th>모델 입력으로 가공</th><th>갱신·유실 규칙</th><th>실기 출처와 현재 상태</th></tr></thead><tbody>{policy_input_rows}</tbody></table></div>
<div class="example-grid"><article><h3>원시값 → ONNX 입력 예</h3><p>검출 결과가 <code>u=0.10</code>, <code>v=-0.08</code>, 깊이 <code>2.40 m</code>, 필터 뒤 중심속도 <code>0.60/-0.30 s⁻¹</code>, 위쪽 양수 드론 수직속도 <code>-0.45 m/s</code>라면 입력은 다음과 같습니다.</p><pre class="vector">[0.10, -0.08, 0.30, 1.00, 0.20, -0.10, -0.15]</pre><p>QR이 사라지면 카메라 여섯 값만 지우므로 같은 순간의 입력은 <code>[0, 0, 0, 0, 0, 0, -0.15]</code>입니다. 중앙 검출과 미검출은 네 번째 <code>detected</code>로 구별합니다.</p></article><article><h3>카메라 숫자와 녹화 화면은 다릅니다</h3><p><strong>정책 카메라는 1280×960</strong>, 수평 FOV 1.74 rad(약 99.7°), 수직 FOV 약 83.27°, near 0.10 m, QR 최소 변 20 px 계약입니다. 영상 오른쪽의 <strong>640×720 하단뷰는 시각화용</strong> 렌더이므로 ONNX에 직접 입력되지 않습니다.</p><p>MuJoCo에서는 렌더된 RGB를 실제 QR decoder로 읽지 않습니다. 센서 경계 함수 안에서만 정답 기하를 투영하고 거리 의존 PnP 이동 잡음 <code>σ=0.0015+0.0004z m</code>, 중심 잡음 <code>σ=0.001</code>, 1% dropout을 더해 detector/PnP형 캐시를 만듭니다. 따라서 조명·반사·rolling shutter·motion blur·실제 decode 실패는 아직 재현하지 않습니다.</p></article></div>
<h3>정책에 들어가는 값과 들어가지 않는 값의 경계</h3><div class="table-scroll"><table class="boundary-table"><thead><tr><th>구분</th><th>포함 값</th><th>쓰이는 곳</th><th>금지/주의</th></tr></thead><tbody><tr><th><span class="kind policy">정책 7D</span></th><td>QR 중심 u/v, PnP 깊이, 검출 플래그, 중심 변화율 u/v, 드론 자체 수직속도</td><td>PPO·DDPG·SAC 학습 모델과 ONNX 입력</td><td>배열 순서·float32·정규화가 한 칸이라도 바뀌면 기존 모델과 호환되지 않습니다.</td></tr><tr><th><span class="kind controller">비행제어 전용</span></th><td>QR PnP 전체 translation/rotation, 연속 PnP로 계산한 목표 속도, 드론 자체 위치·3축 속도·자세·각속도, body-Z 가속도, 자신이 보낸 추력, 임무 시간</td><td>탐색 회랑, 기본 시각서보, 수직 하강, 자세 정렬, IMU 충격·settle·재시도</td><td>신경망 7D에는 넣지 않습니다. 실기에서 추력 기반 충격 잔차를 재현하려면 PX4 내부 구현 또는 companion이 자신의 thrust 명령을 알아야 합니다.</td></tr><tr><th><span class="kind offline">학습·평가 전용</span></th><td>정확한 QR/Go2/판 world pose·속도, 수평거리·상대고도, 착륙발 접점·정상력·침투, Go2 발·전도·경로 상태</td><td>보상, 성공·실패 종단 라벨, <code>offline_sim_*</code> 사후 지표</td><td>정책과 비행제어에서는 읽지 않습니다. 실기 추론에는 이 값이나 보상 계산이 필요 없습니다.</td></tr></tbody></table></div>
<div class="example-grid"><article><h3>정책 밖 비행제어 입력은 어디서 오나</h3><ul class="plain-list"><li><b>자체 위치·속도:</b> MuJoCo <code>framepos/framelinvel</code>에 위치 σ=[1.2,1.2,1.8] cm, 속도 σ=[0.010,0.010,0.014] m/s 잡음을 넣어 50 Hz 캐시로 만듭니다. 실기는 PX4 local position/velocity를 좌표 변환해 사용합니다.</li><li><b>자세·각속도:</b> MuJoCo <code>framequat/gyro</code>이며 gyro σ=0.0015 rad/s입니다. 실기는 PX4 attitude와 IMU/vehicle angular velocity 계열 출력이 대응합니다.</li><li><b>QR 전체 pose:</b> 카메라 PnP translation/rotation과 알려진 camera↔body 외부보정입니다. 연속 3D PnP 차분에 자체 속도를 더하고 ±3 m/s clip, 0.65/0.35 저역통과하여 움직이는 데크 속도를 추정합니다.</li><li><b>미검출 탐색:</b> Go2 위치를 묻지 않고 선언된 탐색 중심·고도, 경과시간, 드론 자신의 추정 위치만으로 sweep합니다.</li><li><b>충격·재시도:</b> body-Z 가속도에서 이미 명령한 추력의 예측 specific force를 뺍니다. 착륙다리 스위치나 접촉 센서는 쓰지 않습니다.</li></ul></article><article><h3>실제 X500 연결 체크리스트</h3><ol class="plain-list"><li>하향 카메라를 강체 고정하고 camera optical↔body FRD 외부변환을 측정합니다.</li><li>실제 1280×960 모드에서 내부행렬·왜곡계수를 보정하고 23 cm 인쇄 크기를 실측합니다.</li><li>프레임 timestamp, 올바른 QR payload, 네 코너, PnP reprojection/깊이 품질을 함께 검사합니다.</li><li>PX4 <code>vehicle_local_position</code> validity와 timestamp를 확인하고 NED↔학습 world 축을 변환합니다.</li><li>카메라·PX4 캐시가 stale이면 추론을 중단하고 hover·상승·임무중단 같은 failsafe로 보냅니다.</li><li>원시값, 정규화 7D, timestamp, detected, ONNX 2D 출력, 안전층 뒤 명령을 모두 기록해 시뮬레이션과 부호·크기를 비교합니다.</li></ol></article></div>
<details><summary>정확한 관측·제어 수식 펼쳐 보기</summary>{state_formula}</details></section>'''
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unitree Go2 등부 QR 착륙</title>
<style>:root{{color-scheme:dark;--bg:#07101d;--panel:#10223a;--panel2:#0b192a;--line:#315477;--cyan:#64d7ff;--green:#62d5a4;--amber:#ffc65e;--muted:#b7cbe0}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#1d4c80,transparent 44rem),var(--bg);color:#eef7ff;font:16px/1.55 system-ui,sans-serif}}main{{width:min(1560px,calc(100% - 32px));margin:auto;padding:42px 0 64px}}h1{{font-size:clamp(2rem,5vw,3.8rem);line-height:1.12;letter-spacing:-.05em;margin:.1rem 0}}h2{{margin:.1rem 0 .55rem}}h3{{margin:.1rem 0 .5rem}}p{{color:var(--muted);margin:.55rem 0}}.eyebrow{{color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.13em}}.lead{{max-width:1100px;font-size:1.07rem}}.section,.algorithm-card{{margin:26px 0;padding:20px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#152c4a,#0a1524)}}.split,.video-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.video-grid{{grid-template-columns:1fr;gap:18px}}.formula-card,.video-panel{{min-width:0;padding:14px;border:1px solid #315477;border-radius:13px;background:var(--panel2)}}.equation{{margin:.7rem 0 0;padding:12px 14px;border:1px solid #294e72;border-radius:10px;background:#040b14;color:#e6f7ff;overflow-x:auto}}.equation mjx-container[display="true"]{{margin:0!important;text-align:left!important;font-size:108%!important}}.callout{{padding:14px 16px;border-left:4px solid var(--cyan);background:#0a1a2c;border-radius:0 10px 10px 0}}.algorithm-heading,.video-title{{display:flex;align-items:center;justify-content:space-between;gap:10px}}.video-title h3 span{{color:#9cb4ca;font-size:.8em;font-weight:500}}.ok,.wait,.badge{{display:inline-block;border-radius:99px;padding:5px 8px;font-size:.72rem;font-weight:800;white-space:nowrap}}.ok{{background:var(--green);color:#052817}}.wait{{background:var(--amber);color:#3c2800}}.badge{{border:1px solid #477ba4;color:#bcecff;background:#0a1b2b}}video,.shot img{{display:block;width:100%;height:auto;aspect-ratio:8/3;object-fit:contain;background:#02060a;border:1px solid #395d83;border-radius:9px}}.shot{{position:relative;display:block;margin-top:9px}}.shot span{{position:absolute;left:8px;bottom:8px;background:#07101ddd;border-radius:6px;padding:4px 7px;font-size:.72rem;color:#dff7ff}}table{{width:100%;border-collapse:collapse;font-size:.83rem}}th,td{{border-bottom:1px solid #294564;padding:8px;text-align:left;vertical-align:top}}th{{color:#dcecff;white-space:nowrap}}.live{{font-size:.72rem;margin-top:9px}}.live th,.live td{{padding:5px 4px}}.table-scroll{{overflow-x:auto}}.links{{display:flex;gap:10px;font-size:.83rem}}.offline-note{{font-size:.76rem}}details{{margin-top:14px;border:1px solid #294e72;border-radius:10px;padding:10px}}summary{{cursor:pointer;color:#c5efff;font-weight:700}}a{{color:var(--cyan)}}code{{color:#c5efff;background:#07101d;border:1px solid #294766;border-radius:5px;padding:.08rem .28rem;font:.86em ui-monospace,monospace}}.table-heading{{margin:24px 0 8px}}footer{{margin-top:28px;color:#a9bdd2;font-size:.84rem}}@media(max-width:760px){{main{{width:calc(100% - 20px);padding-top:24px}}.split{{grid-template-columns:1fr}}.section,.algorithm-card{{padding:14px}}.algorithm-heading{{align-items:flex-start;flex-direction:column}}}}</style>
<style>.section-kicker{{color:var(--green);font-size:.75rem;font-weight:850;letter-spacing:.12em;margin-bottom:.25rem}}.truth-warning{{margin:14px 0;padding:14px 16px;border:1px solid #946e2d;border-left:4px solid var(--amber);border-radius:0 10px 10px 0;background:#251b0d;color:#f5dfb4}}.pipeline{{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:10px;list-style:none;counter-reset:flow;margin:14px 0;padding:0}}.pipeline li{{position:relative;min-height:126px;padding:13px 12px 12px;border:1px solid #315477;border-radius:12px;background:#091827}}.pipeline li::before{{counter-increment:flow;content:counter(flow);display:grid;place-items:center;width:25px;height:25px;margin-bottom:10px;border-radius:50%;background:var(--cyan);color:#042033;font-weight:900}}.pipeline li:not(:last-child)::after{{content:"→";position:absolute;right:-10px;top:48px;z-index:2;color:var(--cyan);font-weight:900}}.pipeline b,.pipeline span{{display:block}}.pipeline span{{margin-top:5px;color:var(--muted);font-size:.8rem}}.clock-note{{padding:12px 14px;border:1px dashed #43759f;border-radius:10px;background:#081521}}.input-table{{min-width:1420px;table-layout:fixed}}.input-table th:first-child{{width:172px}}.input-table td{{line-height:1.48}}.input-index{{display:inline-grid;place-items:center;width:21px;height:21px;margin-right:5px;border-radius:50%;background:#28577e;color:#fff;font-size:.72rem}}.example-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:16px}}.example-grid article{{padding:16px;border:1px solid #315477;border-radius:13px;background:var(--panel2)}}.vector{{overflow-x:auto;margin:.9rem 0;padding:13px;border:1px solid #38648b;border-radius:9px;background:#030913;color:#8ee4ff;font:700 .95rem/1.4 ui-monospace,monospace}}.boundary-table{{min-width:900px}}.kind{{display:inline-block;padding:4px 8px;border-radius:99px;font-size:.75rem}}.kind.policy{{background:#174b68;color:#bdefff}}.kind.controller{{background:#3d3920;color:#ffe48a}}.kind.offline{{background:#482b46;color:#ffc9f7}}.plain-list{{margin:.65rem 0 0;padding-left:1.25rem;color:var(--muted)}}.plain-list li{{margin:.55rem 0}}.input-guide h3{{margin-top:1.35rem}}@media(max-width:1180px){{.pipeline{{grid-template-columns:repeat(3,minmax(0,1fr))}}.pipeline li:nth-child(3)::after{{display:none}}}}@media(max-width:760px){{.pipeline{{grid-template-columns:1fr}}.pipeline li{{min-height:0}}.pipeline li::after{{display:none}}.example-grid{{grid-template-columns:1fr}}}}</style>
<script>window.MathJax={{tex:{{packages:{{'[+]':['ams']}}}},chtml:{{fontURL:'vendor/node_modules/mathjax/es5/output/chtml/fonts/woff-v2'}},options:{{enableMenu:false}}}};</script><script defer src="vendor/node_modules/mathjax/es5/tex-mml-chtml.js"></script></head><body><main>
<div class="eyebrow">OFFICIAL UNITREE GO2 · MUJOCO · ONNX RUNTIME</div><h1>Go2 등부 고정 QR 패드로의 X500 이동 착륙</h1><p class="lead">공식 Unitree Go2 MJCF·메시와 legged-loco 계약을 이식해 재학습한 MuJoCo 저수준 로코모션으로, 보행 경로 중인 Go2의 등에 QR 패드를 <strong>고정 연결</strong>하고 X500을 착륙시켰습니다. PPO·DDPG·SAC를 각각 {timesteps} step 학습했습니다.</p>
<section class="section"><h2>모델·장착 구조</h2><div class="callout"><strong>최고 중급 정책: {best}</strong> · Go2 원본 모델: <a href="https://github.com/unitreerobotics/unitree_mujoco">unitreerobotics/unitree_mujoco</a> · {source}. legged-loco의 450D/12D 계약으로 Go2 PPO 후보도 재학습했지만, 새 10% 지형 통과 기준을 만족하지 않아 공개 지형 영상에는 쓰지 않았습니다. 대신 같은 Go2 IMU·자체 odometry만 쓰는 검증된 물리 기준 트로트를 사용합니다.</div><p><code>base_link child / freejoint 없음</code>으로 0.22 kg 등부 브래킷을 정확히 고정했습니다. <strong>QR 인쇄층은 23 cm, 실제 보이는 물리 판은 36 cm</strong>입니다. 착륙 물리는 QR 판 최상면에서 발생합니다. QR 잉크는 카메라의 coplanar z-fighting을 막기 위한 <strong>3 μm 시각 인쇄층</strong>만 그 위에 두며, 사람 눈·물리 스케일에서는 QR 바닥과 같은 면입니다. X500의 실제 착륙다리 바닥은 PX4 Gazebo 원본과 같은 연속 스키드 레일 2개의 <strong>보이는 MuJoCo 물체</strong>입니다. 각 레일은 <code>0.25 × 0.015 × 0.015 m</code>, body 기준 <code>x=0, y=±0.132 m</code>이며, 바닥면은 <code>z=-0.22759951 m</code>입니다. 이 두 물체가 QR 판에 직접 충돌합니다.</p><p>이전 영상의 다리 관통처럼 보인 현상은 수제 물리 접촉면이 stock X500 시각 스키드보다 101.6 mm 높았던 형상 불일치와, QR 잉크면 아래 0.4 mm에 있던 접촉판 때문이었습니다. 지금은 Gazebo 원본 레일 치수를 이식하고, 보이는 레일 바닥과 물리 QR 판 최상면을 직접 일치시켰습니다. 잉크의 3 μm 시각층은 렌더링 겹침만 피하며 접촉을 바꾸지 않습니다. 드론 옆 검은 상자는 센서가 아니라 종전 접촉 원통과 카메라 하우징·렌즈 렌더용 자리표시자였으므로 제거·숨겼고, 실제 하향 관측을 만드는 MuJoCo <code>down_camera</code>는 그대로 동작합니다. 프로펠러는 PX4 원본 SDF의 rotor-local 메시 오프셋을 반영해 각 모터 축 위에 정렬했습니다.</p></section>
{input_guide_html}
<section class="section"><h2>legged-loco 기반 MuJoCo 저수준 로코모션</h2><div class="callout"><strong>소스:</strong> <a href="https://github.com/yang-zj1026/legged-loco">yang-zj1026/legged-loco</a> (commit 87b0d3d). 이 저장소에는 공개 Go2 체크포인트가 없으므로, Isaac Lab 전용 상태·행동 계약을 MuJoCo에 이식하고 실제 접지 미끄럼·몸통 높이·대각 접촉 위상을 보상에 넣어 PPO를 재학습했습니다.</div><div class="split"><div class="formula-card"><h3>IK 트로트·상태·행동·PD</h3><p>Go2 저수준 정책은 몸통 각속도·자세·속도명령·12개 관절 위치와 속도·직전 행동을 10프레임 쌓은 450개 값을 받습니다. 12개 관절 보정값을 출력하며, 기준 IK 트로트 관절각에 더한 뒤 PD 토크로 보행시킵니다. 이 Go2 입력은 드론 정책 7D와 완전히 별개입니다.</p><details><summary>Go2 상태·행동·PD 수식 펼쳐 보기</summary>{loco_formula}</details></div><div class="formula-card"><h3>독립 로코모션 평가</h3><p><code>평균 속도 오차</code> {loco_error}<br><code>평균 yaw-rate 오차</code> {loco_yaw_error}<br><code>접지 발 미끄럼</code> {loco_slip}<br><code>몸통 높이</code> {loco_height}<br><code>대각 접촉 일치율</code> {loco_match}<br><code>root wrench 최대 절댓값</code> {loco_root_wrench}<br><code>종단 base-up</code> {loco_up}<br><code>전도율</code> {loco_fall}<br><code>평균 경로 길이</code> {loco_path}</p><p>QR 데크의 0.22 kg 고정 하중을 학습·평가에 포함했습니다. Go2 root에는 외력을 넣지 않으며 보행은 관절 토크와 실제 발 접촉만으로 발생합니다. 기체는 5 ms MuJoCo 접촉을 사용하고, PPO는 20 ms마다 행동을 갱신합니다.</p></div></div></section>
<section class="section"><h2>정책 출력·안전층·학습 보상</h2><div class="callout"><strong>X500에는 착륙다리 센서가 없습니다.</strong> 네 착륙다리는 물리 충돌 형상일 뿐이며 touch/load/contact 채널을 만들지 않았습니다. 모델 출력도 모터 PWM·추력·최종 속도 명령이 아니라, 기본 시각서보에 아주 작게 더할 수 있는 수평 보정 제안 두 개입니다.</div><div class="split"><div class="formula-card"><h3>ONNX 출력 2개는 실제로 무엇을 하나</h3><p><code>a_x, a_y ∈ [-1,1]</code>은 드론 body XY 방향의 무차원 residual 제안입니다. 제어기는 이를 현재 드론 자세로 world XY에 회전한 다음, 카메라가 측정한 QR 중심을 향하는 안쪽 성분만 남깁니다. QR에서 멀어지는 성분과 접선 성분은 버립니다.</p><p>held-out 평가·ONNX 영상에서 남은 값의 최대 영향은 <strong>0.001 m/s</strong>입니다. 상대높이 1.20 m 아래부터 선형으로 줄고 0.45 m 이내에서는 완전히 0이 됩니다. 실제 빠른 이동, 목표속도 feed-forward, 중심 복원, 단계별 하강과 자세 정렬은 정책 밖 결정론적 카메라/IMU 제어기가 담당하며 수평 목표속도는 3.6 m/s로 제한됩니다. 즉 RL 모델 하나가 착륙기 전체를 직접 조종하는 구조가 아닙니다.</p></div><div class="formula-card"><h3>보상은 학습 때 무엇을 가르치나</h3><ul class="plain-list"><li>이전 step보다 QR 정답 수평거리 <code>d_t</code>를 줄이면 진행 보상, 멀어지면 같은 항에서 손해를 줍니다.</li><li>남은 거리와 불필요하게 큰 raw residual 행동에는 매 step 패널티를 줍니다.</li><li>0.30 m 안에 들어오면 작은 정렬 보너스, 물리적으로 안정 착륙하면 +110, hard landing은 -55입니다.</li><li>Go2가 넘어지면 -80, 15 m 경계 또는 9 m 고도를 벗어나면 -25입니다.</li></ul><p><code>d_t</code>, 정확한 상대고도, 데크 상대속도와 MuJoCo 접촉은 <strong>privileged training/termination label</strong>입니다. 정책 관측이나 비행제어 입력이 아니며, 실기 ONNX 추론에서는 보상을 계산할 필요가 없습니다. 접촉 수·정상력·침투량을 읽는 dense reward 항도 없습니다.</p><details><summary>정확한 학습 보상 수식 펼쳐 보기</summary>{reward_formula}</details></div></div></section>
<section class="section"><h2>현재 구현과 실기 대응 센서 감사표</h2><div class="table-scroll"><table><thead><tr><th>값</th><th>검증된 MuJoCo 경로</th><th>실기에서 연결할 센서/토픽</th><th>정책/제어 사용</th></tr></thead><tbody><tr><td><code>u_qr, v_qr, z_pnp, detected, R_CM^PnP</code></td><td>명목 30 Hz, 5 ms 격자에서 실제 약 28.57 Hz인 QR/PnP 이동·회전 측정 에뮬레이터</td><td>별도 장착·보정한 하향 RGB 카메라 + CameraInfo + QR corner detector + 23 cm known-size solvePnP</td><td>u/v/depth/detected는 7D 정책 입력 · 회전은 정책 밖 자세제어</td></tr><tr><td><code>du_qr/dt, dv_qr/dt</code></td><td>연속 검출 QR 중심의 timestamp 차분, ±5/s clip과 0.65/0.35 저역통과</td><td>프레임별 detector 중심 좌표와 캡처 timestamp</td><td>예 · 7D 정책 입력</td></tr><tr><td><code>v_z,est</code></td><td>명시적 GNSS 속도 채널에 잡음을 더한 정확히 50 Hz sample-and-hold PX4 출력 surrogate</td><td>PX4 <code>vehicle_local_position.vz</code>의 validity·timestamp 확인 및 NED 하향 양수 → 학습 world-up 부호 반전. 실제 EKF2/기압계는 이 MuJoCo 구현에서 실행하지 않음</td><td>예 · 7D 중 1개</td></tr><tr><td><code>p_est, v_est, R_imu, omega_imu</code></td><td>명시적 framepos·framelinvel·framequat·gyro에서 만든 50 Hz PX4 출력 surrogate</td><td>실기 GNSS/IMU와 PX4 local position·attitude 상태추정 출력 및 FRD/NED 좌표 변환</td><td>아니오 · 정책 밖 비행제어</td></tr><tr><td><code>f_B,z^imu, f_B,z^cmd</code></td><td>body-Z accelerometer와 시뮬레이터 제어기가 이미 알고 있는 collective-thrust 명령</td><td>IMU 가속도계 + 제어기가 자신이 보낸 추력. 단순 Offboard 속도 setpoint만 보내는 현재 ROS 경로에서는 같은 commanded-force 잔차를 바로 얻을 수 없음</td><td>아니오 · 충격/재시도 제어</td></tr><tr><td><code>offline_sim_*</code> Go2/base/pad·경로·착륙다리-판 물리 접촉</td><td>MuJoCo 학습 종단·종단 라벨·사후 평가 로거</td><td>실기 대응 센서 없음·비행제어에서 사용 안 함</td><td><strong>아니오</strong></td></tr></tbody></table></div><p>MuJoCo 장면의 QR 정답 기하를 읽는 코드는 카메라 센서 에뮬레이터 한 함수 안으로 격리했습니다. 이 함수는 렌더 RGB를 decode하는 대신 실기 QR detector/solvePnP가 낼 수 있는 값 형태로 투영·잡음·sample-and-hold를 적용합니다. 비행제어는 그 캐시와 드론 자체 추정 surrogate만 읽고 정확한 QR 월드 자세를 직접 복사하지 않습니다. <strong>착륙다리 touch/load/contact 센서는 존재하지 않으며 추가하지도 않았습니다.</strong> 실기 7D 어댑터는 앞으로 구현·비행 검증해야 합니다.</p></section>
<section class="section"><h2>착륙 충격 완화와 시각 재시도</h2><div class="split"><div class="formula-card"><h3>착륙다리 센서 없는 IMU 판정</h3><p>body-Z IMU 가속도에서 비행제어기가 이미 알고 있는 명령 추력 효과를 뺀 잔차가 임계값을 넘고, 마지막 시각 높이와 수직속도 조건도 맞을 때만 충격으로 판정합니다. 이 판단은 정책 입력이 아닌 별도 상태기계입니다.</p><details><summary>충격·settle·재시도 수식 펼쳐 보기</summary>{imu_landing_formula}</details></div><div class="formula-card"><h3>제어 경계</h3><p>body-Z IMU specific force에서 비행제어기가 이미 알고 있는 명령 추력의 예측 specific force를 빼므로, 명령한 상승·감속을 접촉 충격으로 오인하는 경우를 줄입니다. 이 값은 RL 정책 입력이 아니라 비행제어 상태기계에만 사용합니다.</p><p>충격 뒤 0.35 s 동안 수평 추적은 유지하고 collective를 hover의 88%로 낮춥니다. 종전처럼 마지막 하강속도를 급히 막기 위해 추력을 올리면 데크 정상력과 합쳐져 선행 스키드가 다시 튀었기 때문입니다. 안정 종단에 이르지 않으면 0.45 m/s로 상승해 QR PnP 깊이 0.30 m에서 다시 정렬합니다. 보정된 스키드가 닿을 때 카메라-표식 깊이는 약 0.161 m로 0.10 m near plane보다 크므로 정상 접촉까지 QR이 near-plane 범위 밖으로 사라지는 구간을 가정하지 않습니다. 실제 dropout이나 순간 가림이 생긴 경우에만 마지막 표적 pose·속도를 최대 2.0 s 유지하며 0.16 m/s로 하강하고, 재시도가 3회 누적되면 0.14 m/s로 완화합니다. 접촉 수나 착륙다리 스위치는 이 제어에 쓰이지 않습니다.</p></div></div></section>
<section class="section"><h2>초급·중급·고급의 정의: Go2 속도와 방향전환 복잡도</h2><div class="split"><div class="formula-card"><h3>동일 조건, 경로만 난이도화</h3><p>세 난이도 모두 X500 시작점은 QR 중심에서 2–7 m 원환 영역, 고도 1.20–1.80 m이며 바람·검출 누락·착륙 판정도 같습니다. 달라지는 것은 Go2의 명목 전진속도와 선회 진폭·빈도뿐입니다. Go2는 경로 접선 방향으로 몸통 yaw를 돌리고 대각선 트로트로 이동합니다.</p><details><summary>Go2 경로 생성 수식 펼쳐 보기</summary>{route_formula}</details></div><div class="formula-card"><h3>실행 프로파일</h3><div class="table-scroll"><table><thead><tr><th>난이도</th><th>명목 전진</th><th>최대 선회각</th><th>전환 빈도</th></tr></thead><tbody>{difficulty_rows}</tbody></table></div><p>Go2는 각 난이도에 정의된 시간 함수 속도·선회 명령을 그대로 따르며, 드론의 정렬 또는 착륙 상태로 경로 속도를 바꾸지 않습니다.</p></div></div></section>
<section class="section"><h2><code>offline_sim_*</code> X500 착륙다리 바닥 ↔ QR 판 물리 진단</h2><div class="callout"><strong>착륙다리 센서가 아닙니다.</strong> 아래 레일 접촉·정상력·침투량·데크 속도는 MuJoCo 학습 보상, 종단 판정, 사후 검증에만 쓰이고 7D 정책 관측에는 들어가지 않습니다.</div><div class="split"><div class="formula-card"><h3>무엇을 검사하나</h3><p>MuJoCo contact 목록에서 PX4 Gazebo 원본과 같은 좌·우 스키드 레일 물체와 QR 판 최상면 사이의 접촉만 골라, 닿은 레일 수·정상력 합·가장 깊은 수치 침투를 계산합니다. QR 잉크는 이 면보다 3 μm 위의 렌더 레이어라 하향 카메라에서 깜박이지 않으며 물리 접촉에는 쓰이지 않습니다. 이 값은 실제 X500에서 얻는 센서값이 아니며 영상 성공 여부와 물리 형상 비관통을 사후 확인하기 위한 것입니다.</p><details><summary>접촉 진단 수식 펼쳐 보기</summary>{contact_formula}</details></div><div class="formula-card"><h3>보정 기준</h3><p><code>보이는 물리 착륙 스키드 레일 2개</code>는 각각 길이 250 mm·폭 15 mm·높이 15 mm이고 body 기준 <code>x=0, y=±0.132 m</code>, 바닥면 <code>z=-0.22759951 m</code>입니다. 수입한 stock X500 렌더 스키드의 최저면과 36 cm 물리 QR 데크 최상면을 일치시켜, 사용자가 보는 다리 바닥 물체가 곧 접촉 물체가 되게 했습니다. QR 잉크 3 μm는 실제 인쇄 두께 수준의 시각 분리입니다. 성공은 양쪽 레일 모두 접촉, 상대높이 0.245 m 이하, 중심오차 5.5 cm, 데크 대비 상대속도 0.40 m/s 미만을 동시에 만족해야 합니다.</p><p>MuJoCo soft-contact의 수치 침투 gate는 2 mm이며, 보이는 충돌 레일과 보이는 QR 물리 판이 직접 맞닿으므로 수치 침투와 영상 형상이 서로 어긋나지 않습니다. 각 추론 CSV의 <code>offline_sim_visual_contact_plane_error_m</code>도 1 mm 이하인지 검사합니다. 표와 CSV에는 실제 최대값을 숨기지 않고 <code>offline_sim_*</code>로 기록합니다.</p></div></div></section>
<section class="section"><h2>PPO · DDPG · SAC 학습방법과 하이퍼파라미터</h2><div class="callout">세 기법은 위에서 설명한 <strong>동일한 float32[7] 입력, 2D residual 출력, 환경 보상</strong>을 사용합니다. 차이는 경험을 모으고 actor/critic을 갱신하는 방식입니다.</div><div class="example-grid"><article><h3>PPO</h3><p>현재 정책으로 512 step rollout을 모은 뒤, advantage가 좋은 행동 확률은 올리고 나쁜 행동 확률은 내립니다. 한 번의 큰 업데이트로 정책이 무너지는 것을 막기 위해 이전 정책과의 확률비를 ±20% 범위로 clip하며 같은 rollout을 10 epoch 재사용합니다.</p><p><code>lr 2.5e-4 · batch 128 · γ 0.997 · GAE 0.96 · clip 0.20 · value coefficient 0.50</code></p></article><article><h3>DDPG</h3><p>결정론 actor가 한 행동을 critic이 평가합니다. 180,000개 replay buffer에서 과거 transition을 무작위로 뽑아 Q 오차를 줄이고, actor는 critic의 Q가 커지는 방향으로 갱신합니다. 처음 2,000 step은 buffer를 채우며, 학습 탐색에는 표준편차 0.18 행동 잡음을 사용합니다.</p><p><code>lr 3e-4 · batch 256 · γ 0.997 · target τ 0.01 · warm-up 2,000</code></p></article><article><h3>SAC</h3><p>두 Q critic 중 작은 값을 사용해 과대평가를 줄입니다. actor는 높은 Q뿐 아니라 행동 분포의 entropy도 유지하도록 학습해, DDPG보다 확률적인 탐색을 합니다. entropy 계수는 초기값 0.02에서 자동 조정됩니다.</p><p><code>lr 3e-4 · replay 180,000 · batch 256 · γ 0.997 · target τ 0.01 · warm-up 2,000</code></p></article><article><h3>공통 데이터 한 건</h3><p>각 transition은 <code>(현재 7D, 2D 행동, reward, 다음 7D, 종료 여부)</code>입니다. 보상과 종료 판정에는 MuJoCo 정답 라벨을 쓸 수 있지만, 저장된 actor가 받는 입력은 계속 7D뿐입니다. 평가와 영상에서는 학습 탐색 잡음을 끄고 결정론적으로 ONNX 출력을 사용합니다.</p></article></div><details><summary>PPO·DDPG·SAC 손실 수식 펼쳐 보기</summary>{loss_formula}</details><p>수식은 CDN이 아닌 이 서버의 로컬 MathJax로 렌더링됩니다.</p></section>
<section class="section"><h2>초급·중급·고급: 9개 동기화 영상과 평가</h2><p>각 MP4는 5 ms 물리 서브스텝에서 실제 30 fps로 캡처하며, 좌측 Go2·X500 3인칭과 우측 X500 하향 카메라를 같은 MuJoCo state에서 렌더링합니다. 자갈길 영상은 Go2 발이 닿는 실제 충돌면을 가까이 보는 3인칭 주 화면에, 전체 X500 3인칭 삽입 화면을 같은 시점으로 동기화합니다. 우측 하단뷰는 검정 레터박스 없이 640×720 전체 높이를 채웁니다. 넓은 3인칭의 X500은 매 프레임 현재 GL 카메라로 투영하고, 1 Hz MuJoCo segmentation 실루엣으로 실제 렌더 가시성과 크기를 반복 검증합니다. 프로펠러는 영상에서만 반대 방향으로 회전하며, Go2와 QR 장착부의 물리에는 영향을 주지 않습니다.</p>{cards}</section>
{gravel_html}
<footer>6개 핵심 평가지표: <code>mean_reward</code>, <code>std_reward</code>, <code>success_rate</code>, <code>mean_terminal_error_m</code>, <code>mean_episode_duration_s</code>, <code>mean_episode_steps</code>. 접촉·데크 속도·Go2 물리량은 정책 입력이 아닌 <code>offline_sim_*</code> 진단입니다. 모든 산출물은 MuJoCo에서 생성했습니다.</footer></main></body></html>'''
    output = ARTIFACTS / "go2_back_qr_landing_dashboard.html"
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
