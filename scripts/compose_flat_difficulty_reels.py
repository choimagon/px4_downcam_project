#!/usr/bin/env python3
"""Build Korean, stage-labelled flat-ground landing reels.

Each reel keeps one landing policy (PPO, DDPG or SAC) intact and joins its
beginner, intermediate and advanced flat-ground recordings.  A short title
card immediately before every recording exposes the Go2 motion contract and
the sensor/control route used by the X500, so the viewer does not need to
infer the test setup from the footage.  ``--px4-hil`` switches the source to
the verified PX4 SITL recordings and writes separate, explicitly named reels.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "rl_training"
WIDTH = 1920
HEIGHT = 720
FPS = 30
TITLE_DURATION_S = 4.8
FONT_FAMILY = "Noto Sans CJK KR"

STAGES = (
    ("easy", "초급", "0.70", "0.10", "0.05"),
    ("medium", "중급", "0.90", "0.28", "0.09"),
    ("hard", "고급", "1.10", "0.48", "0.14"),
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def korean_font() -> Path:
    result = subprocess.run(
        ["fc-match", "-f", "%{file}\\n", FONT_FAMILY],
        check=True,
        capture_output=True,
        text=True,
    )
    font = Path(result.stdout.strip().splitlines()[0])
    if not font.is_file():
        raise FileNotFoundError(f"Korean font was not found: {font}")
    return font


def title_text(
    algorithm: str,
    label: str,
    speed: str,
    turn: str,
    frequency: str,
    *,
    px4_hil: bool,
) -> str:
    common = (
        f"{algorithm.upper()} | 평지 이동 QR 착륙",
        label,
        f"Go2: 원래 보행 명령 {speed} m/s · 경로 회전 {turn} rad / {frequency} Hz",
        "X500 시작: QR 중심에서 반경 2.01–6.90 m · 고도 1.20–1.80 m",
    )
    if px4_hil:
        controller_line = (
            "MPC는 카메라/PnP·PX4 EKF 3축 속도로 8-step 3D 속도 비용을 최소화하며, 모터 PWM·직접 force는 쓰지 않음"
            if algorithm == "mpc"
            else "학습 정책은 [Δvx, Δvy, Δvz] 3축 보정만 제안하며, 모터 PWM·직접 force는 정책이 쓰지 않음"
        )
        return "\n".join(
            common
            + (
                "PX4 SITL + MuJoCo MAVLink HIL 검증",
                "MuJoCo IMU·기압·GPS → PX4 EKF2 / QR·PnP → PX4 Offboard vx·vy·vz",
                "PX4 위치·자세 제어와 모터 할당 → HIL_ACTUATOR_CONTROLS 4개 → MuJoCo 기체 물리",
                controller_line,
                "이후 영상: PX4 모터 출력으로 비행한 3인칭 + 하향 QR 카메라 동기화 화면",
            )
        )
    return "\n".join(
        common
        + (
            "하향 RGB 카메라 QR/PnP와 MuJoCo 기체 센서 상태추정(50 Hz)으로 속도·고도를 보정",
            "학습 정책은 수평 미세 보정, 최종 하강·접촉 판정은 카메라·IMU 안전 제어",
            "이후 영상: 3인칭 전체 화면 + 하향 카메라 동기화 화면",
        )
    )


def render_title_card(*, font: Path, text_path: Path, output: Path) -> None:
    # textfile avoids filter-escaping Korean/Unicode punctuation and makes
    # the exact card text auditable beside the temporary render.
    drawtext = (
        f"drawtext=fontfile={font}:textfile={text_path}:"
        "fontcolor=white:fontsize=35:line_spacing=19:"
        "x=150:y=(h-text_h)/2:shadowcolor=black@0.8:shadowx=2:shadowy=2"
    )
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x0d1b2a:s={WIDTH}x{HEIGHT}:r={FPS}:d={TITLE_DURATION_S}",
            "-vf", drawtext,
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
    )


def build_reel(algorithm: str, *, font: Path, px4_hil: bool) -> Path:
    stem = (
        f"px4_sitl_ekf2_{algorithm}_flat_easy_medium_hard_reel"
        if px4_hil else f"{algorithm}_flat_easy_medium_hard_reel"
    )
    output = ARTIFACTS / f"{stem}.mp4"
    with tempfile.TemporaryDirectory(prefix=f"{stem}_") as temporary_dir:
        temporary = Path(temporary_dir)
        concat_parts: list[Path] = []
        for index, (difficulty, label, speed, turn, frequency) in enumerate(STAGES, start=1):
            source = (
                ARTIFACTS / f"px4_sitl_ekf2_{algorithm}_flat_{difficulty}.mp4"
                if px4_hil else ARTIFACTS / f"{algorithm}_go2_back_qr_onnx_{difficulty}_follow.mp4"
            )
            if not source.is_file():
                raise FileNotFoundError(f"Missing flat-ground source video: {source}")
            text_path = temporary / f"{index}_{difficulty}.txt"
            text_path.write_text(
                title_text(algorithm, label, speed, turn, frequency, px4_hil=px4_hil),
                encoding="utf-8",
            )
            card = temporary / f"{index}_{difficulty}_title.mp4"
            render_title_card(font=font, text_path=text_path, output=card)
            concat_parts.extend((card, source))

        concat_list = temporary / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in concat_parts), encoding="utf-8"
        )
        # Re-encode the joined sequence to one portable H.264/MP4 stream.
        # The source recordings are all synchronized 1920x720/30 fps views.
        run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms", nargs="+", choices=("ppo", "ddpg", "sac", "mpc"),
        default=("ppo", "ddpg", "sac"), help="Policies to build (default: all three).",
    )
    parser.add_argument(
        "--px4-hil", action="store_true",
        help="Join verified PX4 SITL + MuJoCo HIL recordings, not MuJoCo-only recordings.",
    )
    arguments = parser.parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("fc-match") is None:
        raise RuntimeError("ffmpeg and fontconfig are required to compose the reels")
    font = korean_font()
    for algorithm in arguments.algorithms:
        output = build_reel(algorithm, font=font, px4_hil=bool(arguments.px4_hil))
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
