#!/usr/bin/env python3
"""Publish Korean results for retrained fast-landing policies and OOD flights."""

from __future__ import annotations

import csv
import html
import json
import math
import shutil
from pathlib import Path

from landing_rl.scenario import MOTION_PROFILE_BOUNDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "rl_training"
ALGORITHMS = ("ppo", "ddpg", "sac")
DIFFICULTIES = (("easy", "초급"), ("medium", "중급"), ("hard", "고급"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def flight_metrics(path: Path) -> dict[str, float | None]:
    empty = {"duration_s": None, "mean_error": None, "p95_error": None, "centered_rate": None}
    if not path.exists():
        return empty
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        return empty
    errors = [math.hypot(float(row["error_x"]), float(row["error_y"])) for row in rows if row.get("error_x")]
    times = [float(row["time_s"]) for row in rows if row.get("time_s")]
    aligned = [int(row["aligned"]) for row in rows if row.get("aligned")]
    ordered = sorted(errors)
    return {
        "duration_s": times[-1] - times[0] if len(times) > 1 else None,
        "mean_error": sum(errors) / len(errors) if errors else None,
        "p95_error": ordered[round((len(ordered) - 1) * 0.95)] if ordered else None,
        "centered_rate": sum(aligned) / len(aligned) if aligned else None,
    }


def score_cells(metrics: dict) -> str:
    success = metrics.get("success_rate")
    success_text = fmt(float(success) * 100, 1, "%") if isinstance(success, (int, float)) else "—"
    return (
        f"<td>{success_text}</td>"
        f"<td>{fmt(metrics.get('mean_episode_duration_s'), 1, ' s')}</td>"
        f"<td>{fmt(metrics.get('mean_terminal_error_m'), 3, ' m')}</td>"
    )


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    published_logs = ARTIFACTS / "logs"
    published_logs.mkdir(exist_ok=True)
    training = read_json(ARTIFACTS / "generalization_training_metrics.json")
    cards: list[str] = []
    manifest: dict[str, object] = {"training": training, "demos": {}}

    for name in ALGORITHMS:
        result = training.get("metrics", {}).get(name, {})
        train_metrics = result.get("training", {})
        held_out = result.get("held_out", {})
        rows = "".join(
            f"<tr><th>{label}</th>{score_cells(held_out.get(key, {}))}</tr>"
            for key, label in DIFFICULTIES
        )
        demos: dict[str, object] = {}
        demo_sections: list[str] = []
        landed_count = 0
        for difficulty, label in DIFFICULTIES:
            video = f"{name}_generalization_{difficulty}_live_dual.mp4"
            csv_path = PROJECT_ROOT / "logs" / f"{name}_generalization_{difficulty}.csv"
            log_path = PROJECT_ROOT / "logs" / f"{name}_generalization_{difficulty}.log"
            for source in (csv_path, log_path):
                if source.exists():
                    shutil.copy2(source, published_logs / source.name)
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            landed = "MAV_CMD_NAV_LAND sent" in log
            landed_count += int(landed)
            flight = flight_metrics(csv_path)
            demos[difficulty] = {
                "video": video,
                "video_exists": (ARTIFACTS / video).exists(),
                "land_command_logged": landed,
                "flight": flight,
                "held_out": held_out.get(difficulty, {}),
            }
            status = "LAND 명령 확인" if landed else "실비행 미검증"
            demo_sections.append(f'''<section class="demo"><div class="title"><h3>{label} · 이동 QR</h3><span class="{'ok' if landed else 'wait'}">{status}</span></div>
<p class="note">전체 모니터 직접 녹화: 왼쪽 Gazebo 3인칭 · 오른쪽 하향 카메라. 두 뷰는 같은 프레임 시각입니다.</p>
<video controls preload="metadata"><source src="{video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video>
<dl><div><dt>평균 QR 오차</dt><dd>{fmt(flight['mean_error'], 3)}</dd></div><div><dt>P95 QR 오차</dt><dd>{fmt(flight['p95_error'], 3)}</dd></div><div><dt>중심 정렬 비율</dt><dd>{fmt(float(flight['centered_rate']) * 100 if flight['centered_rate'] is not None else None, 1, '%')}</dd></div><div><dt>추적 시간</dt><dd>{fmt(flight['duration_s'], 1, ' s')}</dd></div></dl>
<p><a href="{video}">MP4 열기</a> · <a href="logs/{name}_generalization_{difficulty}.csv">비행 CSV</a> · <a href="logs/{name}_generalization_{difficulty}.log">실행 로그</a></p></section>''')
        status = f"LAND {landed_count}/3 확인"
        manifest["demos"][name] = {"demos": demos, "training": train_metrics, "held_out": held_out}
        cards.append(f'''<article class="card"><div class="title"><h2>{name.upper()}</h2><span class="{'ok' if landed_count == 3 else 'wait'}">{status}</span></div>
<h3>초급 · 중급 · 고급 실비행 영상 3개</h3>{''.join(demo_sections)}
<h3>재학습 정책 평가</h3><table><thead><tr><th>분포</th><th>성공률</th><th>평균 시간</th><th>종단 오차</th></tr></thead><tbody><tr><th>학습 · 빠른 착륙</th>{score_cells(train_metrics)}</tr>{rows}</tbody></table></article>''')

    profile_rows = "".join(
        f"<tr><th>{label}</th><td>{bounds.base_speed_mps[0]:.3f}–{bounds.base_speed_mps[1]:.3f} m/s</td><td>{bounds.lateral_amplitude_mps[0]:.3f}–{bounds.lateral_amplitude_mps[1]:.3f} m/s</td><td>roll ±{bounds.roll_deg[1]:.1f}° · heave ±{bounds.heave_m[1] * 100:.1f} cm</td></tr>"
        for key, label, bounds in (("train", "학습", MOTION_PROFILE_BOUNDS["train"]), ("easy", "평가 · 초급", MOTION_PROFILE_BOUNDS["easy"]), ("medium", "평가 · 중급", MOTION_PROFILE_BOUNDS["medium"]), ("hard", "평가 · 고급", MOTION_PROFILE_BOUNDS["hard"]))
    )
    best = html.escape(str(training.get("best_algorithm", "학습 중")).upper())
    technical = '''<section class="technical"><div class="section-title"><span class="eyebrow">Learning interface</span><h2>상태 · 출력 · 보상 · 손실 설정</h2><p>세 기법은 같은 관측·행동·보상 환경을 사용합니다. 실제 추론에서는 정책이 영상 중심화와 목표 속도 feed-forward를 완전히 대체하지 않고, 안전한 수평 속도 <strong>잔차</strong>를 출력합니다.</p></div>
<article class="common"><h3>공통 상태 입력 (6)</h3><ol><li><code>qr_error_x, qr_error_y</code> — 이미지 중심 대비 QR의 정규화된 좌우·상하 오차</li><li><code>altitude</code> — 데크 기준 상대 고도</li><li><code>detected</code> — QR 검출 유효 플래그</li><li><code>target_velocity_x, target_velocity_y</code> — 현재 이동 QR의 ENU 속도</li></ol><h3>공통 정책 출력 (2)</h3><p><code>lateral_x, lateral_y ∈ [-1, 1]</code>: 시각 추적과 목표 속도 feed-forward 위에 더하는 수평 속도 잔차입니다. 실제 비행은 QR 오차 0.13 이내 14 프레임을 확인한 뒤에만 하강하고, 하강 중에는 0.22의 유지 게이트 및 약 0.43 m의 PX4 LAND 전환 높이를 사용합니다.</p><h3>공통 보상</h3><p><code>7.0 × (이전 거리 − 현재 거리) − 0.030 × 거리 − 0.080 × ‖action‖² + 0.10(0.30 m 이내) + 0.35 × 하강</code>. 성공 착륙은 <code>+100</code>, 잘못된 착륙은 <code>−50</code>, 경계 이탈은 <code>−20</code>입니다.</p></article>
<div class="method-grid"><article><h3>PPO</h3><p><b>손실:</b> clipped surrogate policy objective + value-function MSE + entropy 항입니다. Stable-Baselines3 기본 <code>clip_range=0.2</code>, <code>vf_coef=0.5</code>, <code>ent_coef=0.0</code>을 사용합니다.</p><p><b>설정:</b> <code>lr=2.5e−4</code>, <code>γ=0.997</code>, <code>GAE λ=0.96</code>, rollout <code>512</code>, batch <code>128</code>.</p></article><article><h3>DDPG</h3><p><b>손실:</b> critic TD-target MSE와 actor의 <code>−Q(s, π(s))</code> 목적함수입니다. 탐색에는 행동 정규잡음을 사용합니다.</p><p><b>설정:</b> <code>lr=3e−4</code>, <code>γ=0.997</code>, <code>τ=0.01</code>, replay <code>180,000</code>, warm-up <code>2,000</code>, batch <code>256</code>, noise <code>σ=0.18</code>, MLP <code>[256, 256]</code>.</p></article><article><h3>SAC</h3><p><b>손실:</b> twin critic의 soft Bellman MSE와 entropy-augmented actor objective입니다. 온도 α는 목표 엔트로피를 향해 자동 조정됩니다.</p><p><b>설정:</b> <code>lr=3e−4</code>, <code>γ=0.997</code>, <code>τ=0.01</code>, replay <code>180,000</code>, warm-up <code>2,000</code>, batch <code>256</code>, initial <code>ent_coef=0.02</code>, MLP <code>[256, 256]</code>.</p></article></div></section>'''
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>재학습 QR 착륙 일반화 평가</title>
<style>:root{{color-scheme:dark;--bg:#07101d;--panel:#10223a;--line:#315477;--cyan:#64d7ff;--green:#62d5a4;--amber:#ffc65e}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#1d4c80,transparent 42rem),var(--bg);color:#eef7ff;font:16px system-ui,sans-serif}}main{{width:min(1440px,calc(100% - 32px));margin:auto;padding:42px 0 64px}}h1{{font-size:clamp(2rem,5vw,3.8rem);letter-spacing:-.05em;margin:.1rem 0}}h2,h3{{margin:.1rem 0 .6rem}}p{{color:#b7cbe0;line-height:1.55}}.eyebrow{{color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}.lead{{max-width:950px;font-size:1.08rem}}.grid,.method-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px}}.card,.info,.common,.method-grid article{{padding:18px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#152c4a,#0a1524)}}.title{{display:flex;justify-content:space-between;gap:12px;align-items:center}}.title span{{font-size:.72rem;font-weight:800;border-radius:99px;padding:6px 9px}}.ok{{background:var(--green);color:#052817}}.wait{{background:var(--amber);color:#3c2800}}video{{width:100%;aspect-ratio:16/9;background:#02060a;border:1px solid #395d83;border-radius:10px}}.note{{font-size:.85rem;margin:.4rem 0 .7rem}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:14px 0}}dl div{{background:#081221aa;padding:9px;border-radius:9px}}dt{{font-size:.72rem;color:#a9bdd2}}dd{{margin:3px 0 0}}table{{width:100%;border-collapse:collapse;font-size:.84rem}}th,td{{border-bottom:1px solid #294564;padding:8px;text-align:left}}th{{color:#dcecff}}.info,.technical{{margin:26px 0}}.common{{margin-bottom:18px}}.common ol{{color:#b7cbe0;line-height:1.7;padding-left:1.3rem}}code{{color:#c5efff;background:#07101d;border:1px solid #294766;border-radius:5px;padding:.08rem .28rem;font:.83em ui-monospace,SFMono-Regular,Consolas,monospace}}a{{color:var(--cyan)}}footer{{margin-top:30px;color:#a9bdd2;font-size:.84rem}}@media(max-width:600px){{main{{width:calc(100% - 20px);padding-top:24px}}.grid,.method-grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><div class="eyebrow">PX4 SITL · 재학습 · out-of-distribution evaluation</div><h1>빠르게 학습하고, 더 긴 경로에서 검증한 QR 착륙</h1><p class="lead">PPO·DDPG·SAC는 짧고 차분한 최종 정렬·하강 분포에서 다시 학습했습니다. 추론·평가는 학습과 <strong>겹치지 않는 속도 범위</strong>, 더 긴 곡선 경로, 더 큰 속도 변화와 데크 흔들림을 사용하는 초급·중급·고급 분포에서 수행합니다. 아래에는 기법별 세 단계, 총 <strong>9개</strong>의 2–7 m 고리 임의 출발·이동 QR 실비행 영상이 있습니다.</p><section class="info"><h2>학습 분포와 평가 분포는 다릅니다</h2><table><thead><tr><th>용도</th><th>기준 전진 속도</th><th>측면 곡선 변화</th><th>데크 흔들림</th></tr></thead><tbody>{profile_rows}</tbody></table><p>학습은 빠른 착륙을 위해 짧은 범위·높은 하강률을 사용합니다. 평가는 더 빠른 장거리 이동을 추적해야 하며, 목표 속도 feed-forward와 QR 중심 정렬이 동시에 필요합니다. 최고 중급 난이도 평가 정책: <strong>{best}</strong>.</p></section>{technical}<section class="grid">{''.join(cards)}</section><footer>각 MP4는 같은 데스크톱 화면의 직접 녹화입니다. 좌측 Gazebo 3인칭과 우측 하향 카메라가 한 프레임 안에 있으므로 후처리 영상 합성이나 별도 시간 정렬을 사용하지 않습니다.</footer></main></body></html>'''
    (ARTIFACTS / "generalization_results.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output = ARTIFACTS / "generalization_landing_dashboard.html"
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
