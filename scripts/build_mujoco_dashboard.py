#!/usr/bin/env python3
"""Build the Korean MuJoCo QR-landing evidence and training-method dashboard."""

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

# Exact profile values from landing_rl/mujoco_environment.py.
DIFFICULTY_SETTINGS = {
    "train": ("학습", "0.035–0.060", "2.01–6.99", "1.35–1.75", "1.10", "0.22", "0.030", "0.008"),
    "easy": ("초급", "0.115–0.155", "2.01–4.00", "1.35–1.90", "1.15", "0.20", "0.035", "0.010"),
    "medium": ("중급", "0.234–0.306", "3.00–5.50", "1.45–2.20", "1.25", "0.18", "0.045", "0.014"),
    "hard": ("고급", "0.460–0.560", "4.50–6.99", "1.55–2.50", "1.38", "0.16", "0.060", "0.020"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value: object, digits: int = 3, suffix: str = "") -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def flight(path: Path) -> dict[str, float | None]:
    """Return directly measured values from one ONNX inference CSV."""
    empty = {"duration": None, "mean_error": None, "final_error": None, "final_altitude": None, "detection_rate": None, "frames": None}
    if not path.exists():
        return empty
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        return empty
    errors = [float(row["error_m"]) for row in rows]
    detections = [float(row["detected"]) for row in rows]
    return {
        "duration": float(rows[-1]["sim_time_s"]),
        "mean_error": sum(errors) / len(errors),
        "final_error": float(rows[-1]["error_m"]),
        "final_altitude": float(rows[-1]["altitude_m"]),
        "detection_rate": 100.0 * sum(detections) / len(detections),
        "frames": float(len(rows)),
    }


def equation(*lines: str) -> str:
    """Return a display-math block for the locally hosted MathJax runtime."""
    return '<div class="equation">\\[\\begin{aligned}' + r" \\ ".join(lines) + r"\end{aligned}\]</div>"


def evaluation_cells(value: dict) -> str:
    return "".join(
        (
            f"<td>{fmt(value.get('mean_reward'), 3)}</td>",
            f"<td>{fmt(value.get('std_reward'), 3)}</td>",
            f"<td>{fmt(float(value.get('success_rate', float('nan'))) * 100, 1, '%')}</td>",
            f"<td>{fmt(value.get('mean_terminal_error_m'), 4, ' m')}</td>",
            f"<td>{fmt(value.get('mean_episode_duration_s'), 2, ' s')}</td>",
            f"<td>{fmt(value.get('mean_episode_steps'), 2, ' step')}</td>",
        )
    )


def live_cells(value: dict[str, float | None]) -> str:
    return "".join(
        (
            f"<td>{fmt(value['duration'], 1, ' s')}</td>",
            f"<td>{fmt(value['mean_error'], 4, ' m')}</td>",
            f"<td>{fmt(value['final_error'], 4, ' m')}</td>",
            f"<td>{fmt(value['final_altitude'], 4, ' m')}</td>",
            f"<td>{fmt(value['detection_rate'], 1, '%')}</td>",
            f"<td>{fmt(value['frames'], 0)}</td>",
        )
    )


def video_panel(algorithm: str, difficulty: str, korean: str) -> str:
    stem = f"{algorithm}_mujoco_onnx_{difficulty}_follow"
    video, snapshot, csv_file = f"{stem}.mp4", f"{stem}.png", f"{stem}.csv"
    evidence = flight(ARTIFACTS / csv_file)
    verified = all((ARTIFACTS / file).exists() for file in (video, snapshot, csv_file))
    status = "완료 · ONNX 추론" if verified else "산출물 대기"
    status_class = "ok" if verified else "wait"
    return f'''<section class="video-panel"><div class="video-title"><h3>{korean} <span>({difficulty})</span></h3><b class="{status_class}">{status}</b></div>
<video controls preload="metadata"><source src="{video}" type="video/mp4">MP4 재생을 지원하지 않습니다.</video>
<a class="shot" href="{snapshot}"><img src="{snapshot}" alt="{algorithm.upper()} {korean} MuJoCo ONNX 추론"><span>좌측: 3인칭 X500 · 우측: 하향 QR 카메라</span></a>
<table class="live-table"><thead><tr><th>추론 시간</th><th>평균 QR 오차</th><th>최종 QR 오차</th><th>최종 상대고도</th><th>QR 검출률</th><th>프레임</th></tr></thead><tbody><tr>{live_cells(evidence)}</tr></tbody></table>
<p class="links"><a href="{video}">MP4</a><a href="{snapshot}">PNG</a><a href="{csv_file}">CSV</a></p></section>'''


def algorithm_card(name: str, metrics: dict, onnx_item: dict) -> str:
    policy = metrics.get("metrics", {}).get(name, {})
    held_out = policy.get("held_out", {})
    evaluation_rows = "".join(
        f"<tr><th>{label}</th>{evaluation_cells(held_out.get(key, {}))}</tr>" for key, label in DIFFICULTIES
    )
    videos = "".join(video_panel(name, key, label) for key, label in DIFFICULTIES)
    validation = fmt(onnx_item.get("validation_max_abs_action_error"), 2)
    return f'''<article class="algorithm-card"><div class="algorithm-heading"><div><p class="eyebrow">{name.upper()} POLICY</p><h2>{name.upper()} · 초급/중급/고급</h2></div><span class="badge">ONNX action diff ≤ {validation}</span></div>
<p class="sub">검증 정책은 MuJoCo에서 학습한 뒤 ONNX Runtime CPU로 추론했습니다. 아래 평가는 난이도별 독립 시드 20 에피소드 평균이고, 각 영상은 그 평가와 별개인 고정 시드 1회 재현입니다.</p>
<div class="video-grid">{videos}</div>
<h3 class="table-heading">정량 평가 — 6개 지표 전체</h3>
<div class="table-scroll"><table><thead><tr><th>분포</th><th>평균 누적 보상</th><th>보상 표준편차</th><th>성공률</th><th>평균 종단 오차</th><th>평균 시간</th><th>평균 step</th></tr></thead><tbody><tr><th>학습 분포</th>{evaluation_cells(policy.get('training', {}))}</tr>{evaluation_rows}</tbody></table></div>
<p class="note">성공 = 상대고도 ≤ 0.15 m 그리고 수평 QR 오차가 난이도별 착륙 허용오차보다 작음. 종단 오차와 시간은 성공·실패를 모두 포함한 에피소드 종점 평균입니다.</p>
</article>'''


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metrics = read_json(ARTIFACTS / "mujoco_training_metrics.json")
    onnx_models = {item.get("algorithm"): item for item in read_json(ARTIFACTS / "mujoco_onnx_models.json").get("models", [])}
    cards = "".join(algorithm_card(name, metrics, onnx_models.get(name, {})) for name in ALGORITHMS)
    best = html.escape(str(metrics.get("best_algorithm", "—")).upper())
    version = html.escape(str(metrics.get("mujoco_version", "—")))
    timesteps = html.escape(str(metrics.get("timesteps_per_algorithm", "—")))
    difficulty_rows = "".join(
        f"<tr><th>{label}</th><td>{speed} m/s</td><td>{radius} m</td><td>{altitude} m</td><td>{max_speed} m/s</td><td>{descent} m/s</td><td>{wind}</td><td>{dropout}</td></tr>"
        for _, (label, speed, radius, altitude, max_speed, descent, wind, dropout) in DIFFICULTY_SETTINGS.items()
    )

    observation_formula = equation(
        r"o_t &= [e_x, e_y, h, \mathrm{detected}, v_{qr,x}, v_{qr,y}] \in \mathbb{R}^{6}",
        r"r_{xy} &= p_{drone,xy} - p_{qr,xy}",
        r"h &= \max(0, z_{drone} - z_{qr} - 0.025)",
        r"\mathrm{detected} &= \mathbb{1}[\lVert r_{xy}\rVert \le \max(1.15, 1.55h) \land \mathrm{no\ dropout}]",
        r"[e_x,e_y] &= \operatorname{clip}\!\left(-\frac{r_{xy}}{\max(1.20,1.55h)},-1,+1\right)",
        r"a_t &= [a_x,a_y] \in [-1,+1]^2",
    )
    control_formula = equation(
        r"v_{xy}^{*} &= v_{qr,xy} + 0.92\!\left(-\frac{r_{xy}}{\max(1.20,1.55h)}\right)v_{max} + 0.008a_tv_{max} \quad (\mathrm{detected})",
        r"v_{xy}^{*} &= v_{qr,xy} - 0.46r_{xy}, \qquad \lVert v_{xy}^{*}\rVert \le 1.50\ \mathrm{m/s} \quad (\mathrm{not\ detected})",
        r"\mathrm{aligned} &= \mathrm{detected} \land \lVert r_{xy}\rVert < \mathrm{alignment\ limit} \land \lVert v_{drone,xy}-v_{qr,xy}\rVert < 0.26",
        r"v_z^{*} &= -\mathrm{max\_descent} \qquad (\mathrm{aligned\ for\ 5\ consecutive\ steps})",
        r"a_{phys} &= [4.6e_{vx},4.6e_{vy},5.8e_{vz}] + w_{xy}",
        r"F_{body} &= \operatorname{clip}\!\left(m(a_{phys}-g),-32\ \mathrm{N},+32\ \mathrm{N}\right)",
    )
    reward_formula = equation(
        r"d_t &= \lVert p_{drone,xy} - p_{qr,xy}\rVert",
        r"r_t &= 7.00(d_{t-1}-d_t)-0.030d_t-0.080\lVert a_t\rVert_2^2",
        r"&\quad +0.100\mathbb{1}[d_t<0.30\ \mathrm{m}]+0.350\cdot\mathrm{descent}_t",
        r"&\quad +100\mathbb{1}[\mathrm{success}]-50\mathbb{1}[\mathrm{hard\ landing}]-20\mathbb{1}[\mathrm{out\ of\ bounds}]",
        r"\mathrm{success} &= \mathbb{1}[h\le0.15\ \mathrm{m}\ \land\ d_t<\mathrm{landing\ limit}]",
    )
    ppo_formula = equation(
        r"A_t &= \operatorname{GAE}(\gamma=0.997,\lambda=0.96)",
        r"\rho_t &= \frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta,old}(a_t\mid o_t)}",
        r"L_{clip} &= -\mathbb{E}[\min(\rho_tA_t,\operatorname{clip}(\rho_t,0.80,1.20)A_t)]",
        r"L_{value} &= \mathbb{E}[(V_\theta(o_t)-R_t)^2]",
        r"L_{PPO} &= L_{clip}+0.50L_{value}-0.00\mathcal{H}(\pi_\theta)",
    )
    ddpg_formula = equation(
        r"a_{train} &= \operatorname{clip}(\mu_\theta(o_t)+\epsilon,-1,+1), \qquad \epsilon\sim\mathcal{N}(0,0.18^2I)",
        r"y_t &= r_t+\gamma(1-\mathrm{done}_t)Q_{\phi,target}(o_{t+1},\mu_{\theta,target}(o_{t+1}))",
        r"L_{critic} &= \mathbb{E}[(Q_\phi(o_t,a_t)-y_t)^2]",
        r"L_{actor} &= -\mathbb{E}[Q_\phi(o_t,\mu_\theta(o_t))]",
        r"\mathrm{target} &\leftarrow \tau\cdot\mathrm{online}+(1-\tau)\cdot\mathrm{target}, \qquad \tau=0.01",
    )
    sac_formula = equation(
        r"a' &\sim \pi_\theta(\cdot\mid o_{t+1})",
        r"y_t &= r_t+\gamma(1-\mathrm{done}_t)[\min(Q'_1,Q'_2)-\alpha\log\pi_\theta(a'\mid o_{t+1})]",
        r"L_{Q_i} &= \mathbb{E}[(Q_{i,\phi}(o_t,a_t)-y_t)^2], \qquad i\in\{1,2\}",
        r"L_{actor} &= \mathbb{E}[\alpha\log\pi_\theta(a\mid o_t)-\min(Q_{1,\phi},Q_{2,\phi})]",
        r"L_\alpha &= -\mathbb{E}[\log\alpha(\log\pi_\theta(a\mid o_t)+H_{target})], \qquad H_{target}=-2",
    )

    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MuJoCo QR 착륙 — 9개 영상·평가·학습 수식</title>
<style>
:root{{color-scheme:dark;--bg:#07101d;--panel:#10223a;--panel2:#0b192a;--line:#315477;--cyan:#64d7ff;--green:#62d5a4;--amber:#ffc65e;--muted:#b7cbe0}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#1d4c80,transparent 42rem),var(--bg);color:#eef7ff;font:16px/1.55 system-ui,sans-serif}}main{{width:min(1540px,calc(100% - 32px));margin:auto;padding:42px 0 64px}}h1{{font-size:clamp(2rem,5vw,3.8rem);letter-spacing:-.05em;line-height:1.12;margin:.1rem 0}}h2{{line-height:1.25;margin:.1rem 0 .6rem}}h3{{margin:.1rem 0 .55rem}}p{{color:var(--muted);margin:.55rem 0}}.eyebrow{{color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}.lead{{max-width:1060px;font-size:1.08rem}}.section,.algorithm-card{{margin:26px 0;padding:20px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#152c4a,#0a1524)}}.split{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.formula-card{{padding:16px;border:1px solid #315477;border-radius:14px;background:var(--panel2)}}.equation{{margin:.7rem 0 0;padding:12px 14px;border:1px solid #294e72;border-radius:10px;background:#040b14;color:#e6f7ff;overflow-x:auto}}.equation mjx-container[display="true"]{{margin:0!important;text-align:left!important;font-size:108%!important}}.callout{{padding:14px 16px;border-left:4px solid var(--cyan);background:#0a1a2c;border-radius:0 10px 10px 0}}.video-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.video-panel{{min-width:0;padding:13px;border:1px solid #315477;border-radius:13px;background:#081423}}.video-title,.algorithm-heading{{display:flex;align-items:center;justify-content:space-between;gap:10px}}.video-title h3 span{{color:#9cb4ca;font-size:.78em;font-weight:500}}.ok,.wait,.badge{{display:inline-block;font-size:.72rem;font-weight:800;border-radius:99px;padding:5px 8px;white-space:nowrap}}.ok{{background:var(--green);color:#052817}}.wait{{background:var(--amber);color:#3c2800}}.badge{{border:1px solid #477ba4;color:#bcecff;background:#0a1b2b}}video,.shot img{{display:block;width:100%;height:auto;aspect-ratio:8/3;object-fit:contain;background:#02060a;border:1px solid #395d83;border-radius:9px}}.shot{{position:relative;display:block;margin-top:9px}}.shot span{{position:absolute;left:8px;bottom:8px;background:#07101ddd;border-radius:6px;padding:4px 7px;font-size:.72rem;color:#dff7ff}}.live-table{{font-size:.73rem;margin-top:9px}}.live-table th,.live-table td{{padding:5px 4px}}.links{{display:flex;gap:10px;font-size:.83rem}}a{{color:var(--cyan)}}table{{width:100%;border-collapse:collapse;font-size:.84rem}}th,td{{border-bottom:1px solid #294564;padding:8px;text-align:left;vertical-align:top}}th{{color:#dcecff;white-space:nowrap}}.table-scroll{{overflow-x:auto}}.table-heading{{margin:24px 0 8px}}.note{{font-size:.8rem}}code{{color:#c5efff;background:#07101d;border:1px solid #294766;border-radius:5px;padding:.08rem .28rem;font:.86em ui-monospace,monospace}}footer{{margin-top:30px;color:#a9bdd2;font-size:.84rem}}@media(max-width:1080px){{.video-grid{{grid-template-columns:1fr}}}}@media(max-width:760px){{main{{width:calc(100% - 20px);padding-top:24px}}.split{{grid-template-columns:1fr}}.section,.algorithm-card{{padding:14px}}.algorithm-heading{{align-items:flex-start;flex-direction:column}}}}
</style><script>window.MathJax={{tex:{{packages:{{'[+]':['ams']}}}},chtml:{{fontURL:'vendor/node_modules/mathjax/es5/output/chtml/fonts/woff-v2'}},options:{{enableMenu:false}}}};</script><script defer src="vendor/node_modules/mathjax/es5/tex-mml-chtml.js"></script></head><body><main>
<div class="eyebrow">MuJoCo {version} · Stable-Baselines3 2.3.2 · ONNX Runtime CPU</div><h1>이동 QR 정밀 착륙: 9개 영상, 전체 평가, 학습 수식</h1>
<p class="lead">PPO·DDPG·SAC를 각각 MuJoCo에서 {timesteps} step 학습했습니다. 초급·중급·고급은 서로 다른 이동 QR 분포이며, 각 기법·난이도 조합의 동기화된 3인칭/하향 카메라 MP4를 제공합니다. 시각 기체는 기존 Gazebo <code>x500_mono_cam_down</code>의 원본 X500 프레임·모터·프로펠러 메시를 변환해 사용했고, 영상에서만 프로펠러 4개를 반대 방향으로 회전시킵니다.</p>
<section class="section"><h2>결과를 읽는 기준</h2><div class="callout"><strong>최고 중급 정책: {best}</strong> · 평가는 난이도별 독립 20 에피소드 평균입니다. 아래 영상의 수치는 해당 MP4 1회의 ONNX Runtime 재현 결과이며, 평가 평균과 섞지 않았습니다. 모든 MP4 프레임은 하나의 MuJoCo 상태에서 좌측 3인칭과 우측 하향 카메라를 함께 렌더링합니다.</div></section>
<section class="section"><h2>난이도·초기조건·환경 파라미터</h2><div class="table-scroll"><table><thead><tr><th>분포</th><th>QR 기본 속도</th><th>시작 반경</th><th>시작 고도</th><th>수평 최대속도</th><th>최대 하강</th><th>풍 노이즈 σ</th><th>검출 누락률</th></tr></thead><tbody>{difficulty_rows}</tbody></table></div><p class="note">속도·반경·고도는 각 에피소드에서 해당 범위 균등 샘플링입니다. 학습 분포와 평가 3개 분포는 QR 이동 속도 범위가 분리되어 있습니다.</p></section>
<section class="section"><h2>입력 상태, 출력 행동, MuJoCo 제어식</h2><div class="split"><div class="formula-card"><h3>관측과 출력</h3>{observation_formula}</div><div class="formula-card"><h3>시각 서보·물리 제어</h3>{control_formula}</div></div></section>
<section class="section"><h2>보상·종료 조건</h2><div class="formula-card">{reward_formula}</div><p class="note">학습 행동은 QR 검출 뒤의 미세 수평 보정만 담당합니다. QR 탐색·접근과 고도 안전 하강은 동일한 MuJoCo 시각 서보 규칙으로 제한되어 있습니다.</p></section>
<section class="section"><h2>알고리즘별 손실식과 하이퍼파라미터</h2><div class="split"><div class="formula-card"><h3>PPO</h3>{ppo_formula}<p><code>lr=2.5e-4</code> <code>rollout=512</code> <code>batch=128</code> <code>epochs=10</code> <code>gamma=0.997</code> <code>GAE lambda=0.96</code> <code>clip=0.20</code> <code>vf=0.50</code> <code>entropy=0</code> <code>grad clip=0.50</code></p></div><div class="formula-card"><h3>DDPG</h3>{ddpg_formula}<p><code>lr=3e-4</code> <code>replay=180000</code> <code>warm-up=2000</code> <code>batch=256</code> <code>train freq=1 step</code> <code>gradient steps=1</code> <code>gamma=0.997</code> <code>tau=0.01</code> <code>noise sigma=0.18</code></p></div></div><div class="formula-card" style="margin-top:18px"><h3>SAC</h3>{sac_formula}<p><code>lr=3e-4</code> <code>replay=180000</code> <code>warm-up=2000</code> <code>batch=256</code> <code>train freq=1 step</code> <code>gradient steps=1</code> <code>gamma=0.997</code> <code>tau=0.01</code> <code>target update=1</code> <code>entropy=auto_0.02</code></p></div><p class="note">MathJax는 대시보드 폴더에 로컬로 포함했습니다. CDN이나 인터넷 연결 없이도 LaTex 수식이 렌더링되며, 좁은 화면에서는 수식 영역만 수평 스크롤됩니다. 현재 학습 산출물은 평가 지표를 보관하며, step별 optimizer loss 곡선은 저장하지 않았습니다.</p></section>
<section class="section"><h2>초급·중급·고급 MP4 및 전체 평가</h2><p>각 카드의 영상 표는 MP4 한 회의 직접 측정값이고, 하단 표는 6개 평가 지표의 20-에피소드 평균입니다.</p>{cards}</section>
<footer>배포 산출물: 각 조합의 MP4(H.264 1920×720), PNG 스냅샷, ONNX 추론 CSV, ONNX 가중치. 외부 Gazebo/Isaac 실행 없이 MuJoCo에서 학습·추론·렌더링했습니다.</footer>
</main></body></html>'''
    output = ARTIFACTS / "mujoco_qr_landing_dashboard.html"
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
