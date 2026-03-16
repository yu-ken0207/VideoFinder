import asyncio
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import main


class AnalyzeImageParsingTests(unittest.TestCase):
    def test_analyze_image_parses_markdown_wrapped_fields_with_thresholded_match(self):
        fake_response = {
            "message": {
                "content": (
                    "**Answer:** YES\n"
                    "**Description:** The person in orange clothes is near the center.\n"
                    "**Confidence:** 8"
                )
            }
        }

        async def fake_chat(**_kwargs):
            return fake_response

        with patch.object(main.OLLAMA_CLIENT, "chat", side_effect=fake_chat):
            is_match, description, confidence = asyncio.run(
                main.analyze_image("fake-frame.jpg", "The person in the orange clothes")
            )

        self.assertFalse(is_match)
        self.assertEqual(
            description,
            "The person in orange clothes is near the center.",
        )
        self.assertEqual(confidence, 8)

    def test_parse_analysis_response_uses_none_when_confidence_missing(self):
        is_match, description, confidence = main.parse_analysis_response(
            "Answer: NO\nDescription: The image shows a crowded street."
        )

        self.assertFalse(is_match)
        self.assertEqual(description, "The image shows a crowded street.")
        self.assertIsNone(confidence)

    def test_parse_analysis_response_preserves_model_confidence_for_no_answer(self):
        is_match, description, confidence = main.parse_analysis_response(
            "Answer: NO\nDescription: The image shows a crowded street.\nConfidence: 10"
        )

        self.assertFalse(is_match)
        self.assertEqual(description, "The image shows a crowded street.")
        self.assertEqual(confidence, 10)

    def test_parse_analysis_response_requires_confidence_above_eight_for_yes_match(self):
        is_match, description, confidence = main.parse_analysis_response(
            "Answer: YES\nDescription: A person wearing orange is visible.\nConfidence: 6"
        )

        self.assertFalse(is_match)
        self.assertEqual(description, "A person wearing orange is visible.")
        self.assertEqual(confidence, 6)

    def test_parse_analysis_response_accepts_yes_match_when_confidence_is_nine(self):
        is_match, description, confidence = main.parse_analysis_response(
            "Answer: YES\nDescription: A person wearing orange is visible.\nConfidence: 9"
        )

        self.assertTrue(is_match)
        self.assertEqual(description, "A person wearing orange is visible.")
        self.assertEqual(confidence, 9)

    def test_build_analysis_prompt_embeds_target_description_without_bad_grammar(self):
        prompt = main.build_analysis_prompt("The person in the orange clothes")

        self.assertIn("Target description: The person in the orange clothes", prompt)
        self.assertNotIn("Is there a The person in the orange clothes", prompt)


class AnalyzeVideoStreamingTests(unittest.TestCase):
    def test_analyze_endpoint_emits_complete_event(self):
        class FakeVideoCapture:
            def __init__(self, _path):
                self._reads = 0

            def get(self, prop):
                if prop == main.cv2.CAP_PROP_FPS:
                    return 1
                return 1

            def read(self):
                if self._reads == 0:
                    self._reads += 1
                    return True, np.zeros((4, 4, 3), dtype=np.uint8)
                return False, None

            def release(self):
                return None

        async def fake_analyze_image(_image_path, _object_str):
            return False, "The image shows a crowded street.", None

        async def fake_unload_model():
            return None

        with patch.object(main.cv2, "VideoCapture", FakeVideoCapture), patch.object(
            main.cv2, "imwrite", return_value=True
        ), patch.object(main, "preprocess_image", return_value=True), patch.object(
            main, "analyze_image", side_effect=fake_analyze_image
        ), patch.object(
            main, "unload_vision_model", side_effect=fake_unload_model
        ):
            client = TestClient(main.app)
            response = client.post(
                "/analyze",
                files={"video": ("test.mp4", b"video-bytes", "video/mp4")},
                data={"object_str": "The person in the orange clothes"},
            )

        self.assertEqual(response.status_code, 200)
        payloads = [
            line for line in response.text.splitlines() if line.strip()
        ]
        self.assertEqual(len(payloads), 2)
        self.assertIn('"status": "success"', payloads[0])
        self.assertIn('"status": "complete"', payloads[1])

    def test_analyze_endpoint_stops_after_first_match(self):
        class FakeVideoCapture:
            def __init__(self, _path):
                self._reads = 0

            def get(self, prop):
                if prop == main.cv2.CAP_PROP_FPS:
                    return 1
                return 2

            def read(self):
                if self._reads < 2:
                    self._reads += 1
                    return True, np.zeros((4, 4, 3), dtype=np.uint8)
                return False, None

            def release(self):
                return None

        analyze_calls = 0
        events = []

        async def fake_analyze_image(_image_path, _object_str):
            nonlocal analyze_calls
            analyze_calls += 1
            return True, "A person wearing an orange jacket is visible.", 7

        async def fake_unload_model():
            return None

        def fake_log_event(message):
            events.append(message)

        with patch.object(main.cv2, "VideoCapture", FakeVideoCapture), patch.object(
            main.cv2, "imwrite", return_value=True
        ), patch.object(main, "preprocess_image", return_value=True), patch.object(
            main, "analyze_image", side_effect=fake_analyze_image
        ), patch.object(
            main, "unload_vision_model", side_effect=fake_unload_model
        ), patch.object(
            main, "log_analysis_event", side_effect=fake_log_event
        ):
            client = TestClient(main.app)
            response = client.post(
                "/analyze",
                files={"video": ("test.mp4", b"video-bytes", "video/mp4")},
                data={"object_str": "The person in the orange clothes"},
            )

        self.assertEqual(response.status_code, 200)
        payloads = [line for line in response.text.splitlines() if line.strip()]
        self.assertEqual(len(payloads), 2)
        self.assertIn('"is_match": true', payloads[0])
        self.assertIn('"status": "complete"', payloads[1])
        self.assertEqual(analyze_calls, 1)
        self.assertIn("match found at second 0, stopping analysis", events)
        self.assertIn(
            "summary: frames_analyzed=1, match_found=True, match_second=0",
            events,
        )

    def test_stream_video_analysis_stops_when_request_disconnects(self):
        class FakeVideoCapture:
            def __init__(self, _path):
                self._reads = 0

            def get(self, prop):
                if prop == main.cv2.CAP_PROP_FPS:
                    return 1
                return 3

            def read(self):
                if self._reads < 3:
                    self._reads += 1
                    return True, np.zeros((4, 4, 3), dtype=np.uint8)
                return False, None

            def release(self):
                return None

        class FakeRequest:
            def __init__(self):
                self.calls = 0

            async def is_disconnected(self):
                self.calls += 1
                return self.calls >= 2

        analyze_calls = 0
        events = []

        async def fake_analyze_image(_image_path, _object_str):
            nonlocal analyze_calls
            analyze_calls += 1
            return False, "still analyzing", None

        def fake_log_event(message):
            events.append(message)

        async def collect():
            async def fake_unload_model():
                return None

            with patch.object(main.cv2, "VideoCapture", FakeVideoCapture), patch.object(
                main.cv2, "imwrite", return_value=True
            ), patch.object(main, "preprocess_image", return_value=True), patch.object(
                main, "analyze_image", side_effect=fake_analyze_image
            ), patch.object(
                main, "unload_vision_model", side_effect=fake_unload_model
            ), patch.object(
                main, "log_analysis_event", side_effect=fake_log_event
            ):
                payloads = []
                async for chunk in main.stream_video_analysis(
                    video_path="uploads/test.mp4",
                    video_name="test.mp4",
                    object_str="The person in the orange clothes",
                    request=FakeRequest(),
                ):
                    payloads.append(chunk)
                return payloads

        payloads = asyncio.run(collect())
        self.assertEqual(len(payloads), 1)
        self.assertIn('"status": "success"', payloads[0])
        self.assertEqual(analyze_calls, 1)
        self.assertIn("client disconnected, aborting analysis", events)
        self.assertIn(
            "summary: frames_analyzed=1, match_found=False, match_second=None",
            events,
        )


class TemplateInteractionTests(unittest.TestCase):
    def test_results_preview_includes_lightbox_markup_and_controls(self):
        template = Path("templates/index.html").read_text(encoding="utf-8")

        self.assertIn('id="previewModal"', template)
        self.assertIn('id="previewModalImage"', template)
        self.assertIn("openPreviewModal(", template)
        self.assertIn("closePreviewModal()", template)

    def test_stream_video_analysis_unloads_model_after_completion(self):
        class FakeVideoCapture:
            def __init__(self, _path):
                self._reads = 0

            def get(self, prop):
                if prop == main.cv2.CAP_PROP_FPS:
                    return 1
                return 1

            def read(self):
                if self._reads == 0:
                    self._reads += 1
                    return True, np.zeros((4, 4, 3), dtype=np.uint8)
                return False, None

            def release(self):
                return None

        unload_calls = 0
        events = []

        async def fake_analyze_image(_image_path, _object_str):
            return False, "done", None

        async def fake_unload_model():
            nonlocal unload_calls
            unload_calls += 1

        def fake_log_event(message):
            events.append(message)

        async def collect():
            with patch.object(main.cv2, "VideoCapture", FakeVideoCapture), patch.object(
                main.cv2, "imwrite", return_value=True
            ), patch.object(main, "preprocess_image", return_value=True), patch.object(
                main, "analyze_image", side_effect=fake_analyze_image
            ), patch.object(
                main, "unload_vision_model", side_effect=fake_unload_model
            ), patch.object(
                main, "log_analysis_event", side_effect=fake_log_event
            ):
                return [
                    chunk async for chunk in main.stream_video_analysis(
                        video_path="uploads/test.mp4",
                        video_name="test.mp4",
                        object_str="The person in the orange clothes",
                        request=None,
                    )
                ]

        payloads = asyncio.run(collect())
        self.assertEqual(len(payloads), 2)
        self.assertEqual(unload_calls, 1)
        self.assertIn("analysis complete, unloading model", events)
        self.assertIn(
            "summary: frames_analyzed=1, match_found=False, match_second=None",
            events,
        )


if __name__ == "__main__":
    unittest.main()
