import unittest

import cv2
import mujoco
import numpy as np

from landing_rl.go2_onnx_inference import (
    assert_drone_visible_in_segmentation,
    compose_dual_view,
    draw_drone_locator,
    draw_down_hud,
)
from scripts.go2_suite_transaction import (
    ALGORITHMS,
    DASHBOARD_INPUT_GUIDE_MARKERS,
    DIFFICULTIES,
    OBSERVATION_NAMES,
    dashboard_content_issues,
    demo_stem,
    down_view_quality_metrics,
    evenly_spaced_frame_indices,
)


class Go2VideoContractTests(unittest.TestCase):
    def test_down_camera_fills_complete_right_panel(self) -> None:
        third = np.full((720, 1280, 3), 37, dtype=np.uint8)
        down = np.full((720, 640, 3), (61, 83, 107), dtype=np.uint8)
        composed = compose_dual_view(third, down)
        self.assertEqual(composed.shape, (720, 1920, 3))
        np.testing.assert_array_equal(composed[:, :1280], third)
        np.testing.assert_array_equal(composed[:, 1280:], down)

    def test_down_hud_has_no_opaque_panel(self) -> None:
        frame = np.full((720, 640, 3), (70, 90, 110), dtype=np.uint8)
        rendered = draw_down_hud(
            frame,
            error=0.2,
            contacts=0,
            force=0.0,
            penetration=0.0,
            detected=True,
            imu_impact=False,
            retry_active=False,
            retry_count=0,
        )
        # Glyphs and the centre cross may change pixels, but no large filled
        # rectangle is allowed to replace the camera image.
        unchanged = np.all(rendered == frame, axis=2)
        self.assertGreater(float(np.mean(unchanged)), 0.94)

    def test_segmentation_visibility_uses_real_x500_pixels(self) -> None:
        segmentation = np.full((720, 1280, 2), -1, dtype=np.int32)
        segmentation[250:270, 500:520, 0] = 42
        segmentation[250:270, 500:520, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
        self.assertEqual(
            assert_drone_visible_in_segmentation(
                segmentation, drone_geom_ids=np.array([42], dtype=np.int32)
            ),
            400,
        )
        with self.assertRaises(RuntimeError):
            assert_drone_visible_in_segmentation(
                segmentation, drone_geom_ids=np.array([99], dtype=np.int32)
            )

    def test_locator_clips_stale_box_but_keeps_visible_centre(self) -> None:
        frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
        rendered = draw_drone_locator(
            frame,
            center=(640.0, 360.0),
            box_size=(1600.0, 1000.0),
        )
        self.assertEqual(rendered.shape, frame.shape)
        with self.assertRaises(RuntimeError):
            draw_drone_locator(frame, center=(2.0, 2.0), box_size=(20.0, 20.0))

    def test_video_samples_include_start_and_end(self) -> None:
        indices = evenly_spaced_frame_indices(301, 7)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 300)
        self.assertEqual(len(indices), 7)

    def test_black_down_view_fails_quality_thresholds(self) -> None:
        frame = np.zeros((720, 1920, 3), dtype=np.uint8)
        metrics, _ = down_view_quality_metrics(frame)
        self.assertEqual(metrics["nonblack_fraction"], 0.0)
        self.assertEqual(metrics["luma_std"], 0.0)
        self.assertEqual(metrics["edge_energy"], 0.0)


class Go2DashboardInputGuideContractTests(unittest.TestCase):
    @staticmethod
    def complete_document() -> str:
        observation_rows = "".join(
            f'<tr data-observation="{name}"><td>{name}</td></tr>'
            for name in OBSERVATION_NAMES
        )
        video_links = "".join(
            f'<video src="{demo_stem(algorithm, difficulty)}.mp4"></video>'
            for algorithm in ALGORITHMS
            for difficulty in DIFFICULTIES
        )
        return f"""
        <html lang="ko">
          <script src="vendor/node_modules/mathjax/es5/tex-mml-chtml.js"></script>
          <body>
            <p>offline_sim_* 값은 평가 전용입니다.</p>
            <p>착륙다리 touch/load/contact 센서는 존재하지 않습니다.</p>
            <section id="drone-input-guide"
                     data-real-adapter-status="not-implemented">
              <h2>드론 정책 입력 7개</h2>
              <p>모델 필드, 무슨 값인가, 원시값을 어떻게 얻나, 모델 입력으로 가공,
                 갱신·유실 규칙, 실기 출처와 현재 상태를 설명합니다.</p>
              <ol id="camera-to-vector-pipeline">
                <li>카메라 프레임</li><li>QR 코너 검출</li><li>PnP</li><li>7D 벡터</li>
              </ol>
              <table>{observation_rows}</table>
              <p data-source-frame="PX4-NED" data-vz-conversion="negate">
                PX4 NED의 수직 속도는 부호를 바꿔 상승 양(+) 좌표로 변환합니다.
              </p>
              <p data-policy-camera-resolution="1280x960"
                 data-video-down-view-resolution="640x720">
                정책 카메라는 1280×960이고, 640×720 하단뷰는 시각화용입니다.
              </p>
              <p>실기 어댑터는 현재 구현되지 않았습니다.</p>
            </section>
            {video_links}
          </body>
        </html>
        """

    def test_complete_practical_7d_guide_passes_content_contract(self) -> None:
        self.assertEqual(dashboard_content_issues(self.complete_document()), [])

    def test_each_policy_observation_needs_a_documented_row(self) -> None:
        document = self.complete_document()
        for observation_name in OBSERVATION_NAMES:
            with self.subTest(observation_name=observation_name):
                missing_row = document.replace(
                    f'data-observation="{observation_name}"', "", 1
                )
                issues = dashboard_content_issues(missing_row)
                self.assertTrue(
                    any(observation_name in issue and "observation row" in issue for issue in issues),
                    issues,
                )

    def test_acquisition_and_hardware_semantics_need_explicit_markers(self) -> None:
        document = self.complete_document()
        for label, marker in DASHBOARD_INPUT_GUIDE_MARKERS:
            with self.subTest(label=label):
                missing_marker = document.replace(marker, "", 1)
                issues = dashboard_content_issues(missing_marker)
                self.assertTrue(any(label in issue for issue in issues), issues)

    def test_critical_statuses_must_also_be_explained_in_visible_korean(self) -> None:
        document = self.complete_document()
        for visible_fragment in (
            "PX4 NED",
            "부호",
            "정책 카메라",
            "1280×960",
            "640×720",
            "시각화용",
            "실기 어댑터",
            "구현되지 않았",
        ):
            with self.subTest(visible_fragment=visible_fragment):
                missing_text = document.replace(visible_fragment, "", 1)
                issues = dashboard_content_issues(missing_text)
                self.assertTrue(
                    any(visible_fragment in issue and "visible explanation" in issue for issue in issues),
                    issues,
                )


if __name__ == "__main__":
    unittest.main()
