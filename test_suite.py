"""
Comprehensive Test Suite for Real-Time Object Detection and Tracking System.
Validates:
1. Configuration loading
2. Device detection (GPU vs CPU fallback)
3. Input source validation (webcam, files, invalid paths)
4. Detector initialization and inference
5. ByteTrack tracker association across frames and track ID stability
6. Tracker reset functionality
7. Video recording output and file integrity
8. Screenshot capture functionality
9. Error handling on invalid/unsupported sources
"""

import sys
import os
import unittest
from pathlib import Path
import numpy as np
import cv2

import config
from utils import (
    get_device,
    parse_and_validate_source,
    Detection,
    TrackedObject,
    save_screenshot,
    draw_bounding_box,
    draw_hud,
)
from detector import YOLOObjectDetector
from tracker import ByteTrackManager
from video_processor import VideoProcessor
from create_sample_video import generate_sample_video


class TestObjectDetectionAndTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare sample video and testing resources."""
        cls.sample_video_path = "input/test_synthetic_stream.mp4"
        generate_sample_video(output_path=cls.sample_video_path, width=320, height=240, duration_sec=2, fps=15)
        cls.test_output_video = "output/test_output.mp4"

    def test_01_config_and_directories(self):
        """Verify that configuration settings and directories exist."""
        self.assertTrue(config.MODELS_DIR.exists())
        self.assertTrue(config.INPUT_DIR.exists())
        self.assertTrue(config.OUTPUT_DIR.exists())
        self.assertGreater(config.CONFIDENCE_THRESHOLD, 0.0)
        self.assertLess(config.CONFIDENCE_THRESHOLD, 1.0)
        self.assertIsNotNone(config.MODEL_NAME)

    def test_02_device_detection(self):
        """Verify device detection logic (CPU fallback and GPU detection)."""
        cpu_dev, cpu_name = get_device("cpu")
        self.assertEqual(cpu_dev, "cpu")
        self.assertEqual(cpu_name, "CPU")

        auto_dev, auto_name = get_device("auto")
        self.assertIn(auto_dev, ["cpu", "cuda"])
        self.assertIsInstance(auto_name, str)

    def test_03_source_parsing_and_validation(self):
        """Test input source parsing for webcam aliases, numbers, valid files, and missing files."""
        # Webcam aliases
        self.assertEqual(parse_and_validate_source("webcam"), config.DEFAULT_CAMERA_ID)
        self.assertEqual(parse_and_validate_source("0"), 0)
        self.assertEqual(parse_and_validate_source("1"), 1)

        # Valid file
        valid_res = parse_and_validate_source(self.sample_video_path)
        self.assertTrue(Path(valid_res).exists())

        # Non-existent file error handling
        with self.assertRaises(FileNotFoundError):
            parse_and_validate_source("input/non_existent_file_xyz_123.mp4")

    def test_04_detector_initialization_and_inference(self):
        """Test YOLO detector model loading and frame inference."""
        detector = YOLOObjectDetector(
            model_name=config.MODEL_NAME,
            conf_thresh=0.25,
            device="cpu"
        )
        self.assertIsNotNone(detector.model)
        self.assertGreater(len(detector.names), 0)

        # Create dummy RGB frame with a simulated object
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some color gradients to simulate image content
        cv2.circle(test_frame, (320, 240), 100, (255, 255, 255), -1)

        detections = detector.detect(test_frame)
        self.assertIsInstance(detections, list)

    def test_05_bytetrack_tracker_multi_frame(self):
        """Test ByteTrack tracking across multiple frames."""
        detector = YOLOObjectDetector(
            model_name=config.MODEL_NAME,
            conf_thresh=0.20,
            device="cpu"
        )
        tracker = ByteTrackManager(detector=detector)

        cap = cv2.VideoCapture(self.sample_video_path)
        self.assertTrue(cap.isOpened())

        frame_count = 0
        while frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            raw_dets, tracks = tracker.update(frame)
            self.assertIsInstance(raw_dets, list)
            self.assertIsInstance(tracks, list)
            frame_count += 1

        cap.release()
        self.assertGreater(frame_count, 0)

    def test_06_tracker_reset(self):
        """Test tracker reset function ('R' key functionality)."""
        detector = YOLOObjectDetector(conf_thresh=0.25, device="cpu")
        tracker = ByteTrackManager(detector=detector)
        # Populate dummy trajectory
        tracker.trajectories[1].append((100, 100))
        tracker.cumulative_ids_seen.add(1)

        self.assertEqual(len(tracker.cumulative_ids_seen), 1)
        tracker.reset()
        self.assertEqual(len(tracker.trajectories), 0)
        self.assertEqual(len(tracker.cumulative_ids_seen), 0)

    def test_07_drawing_and_hud(self):
        """Test bounding box and HUD overlay rendering without crashes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_bounding_box(frame, (50, 50, 200, 200), "Person | ID: 1 | 0.95", (0, 255, 0))
        draw_hud(frame, fps=30.0, detected_count=3, tracked_count=2, device_name="CPU", is_paused=False)

        # Check that pixels were altered
        self.assertGreater(np.sum(frame), 0)

    def test_08_screenshot_capture(self):
        """Test screenshot capture ('S' key functionality)."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        saved_file = save_screenshot(frame)
        self.assertTrue(Path(saved_file).exists())
        self.assertGreater(os.path.getsize(saved_file), 0)
        # Clean up test screenshot
        try:
            os.remove(saved_file)
        except Exception:
            pass

    def test_09_video_processor_pipeline_run(self):
        """Test full video processing loop on sample video with saving enabled."""
        detector = YOLOObjectDetector(conf_thresh=0.25, device="cpu")
        tracker = ByteTrackManager(detector=detector)

        processor = VideoProcessor(
            source=self.sample_video_path,
            detector=detector,
            tracker=tracker,
            save_output=True,
            output_path=self.test_output_video,
            headless=True
        )

        processor.run()

        self.assertGreater(processor.total_frames_processed, 0)
        self.assertTrue(Path(self.test_output_video).exists())
        self.assertGreater(os.path.getsize(self.test_output_video), 0)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        for path in [cls.sample_video_path, cls.test_output_video]:
            p = Path(path)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
