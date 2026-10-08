"""
Video Processor Module.
Orchestrates video capture, frame processing, detection, tracking, HUD visualization,
FPS computation, keyboard event handling, and video recording.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Union, Tuple

import cv2
import numpy as np

import config
from utils import (
    draw_bounding_box,
    draw_hud,
    get_color_for_id,
    save_screenshot,
    logger,
    parse_and_validate_source,
)
from detector import YOLOObjectDetector
from tracker import ByteTrackManager


class VideoProcessor:
    """
    Main pipeline processor handling stream capture, detection, tracking,
    visualization, and saving.
    """

    def __init__(
        self,
        source: Union[int, str],
        detector: YOLOObjectDetector,
        tracker: ByteTrackManager,
        save_output: bool = False,
        output_path: Optional[str] = None,
        display_width: Optional[int] = None,
        display_height: Optional[int] = None,
        headless: bool = False,
    ):
        """
        Initializes the video processor.

        Args:
            source: Integer camera index or path string to video file.
            detector: Instance of YOLOObjectDetector.
            tracker: Instance of ByteTrackManager.
            save_output: Flag indicating whether to record output video.
            output_path: Optional custom file path for saved video.
            display_width: Optional width to resize frames for display/processing.
            display_height: Optional height to resize frames for display/processing.
            headless: If True, disables cv2.imshow for non-GUI environments.
        """
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.save_output = save_output
        self.display_width = display_width
        self.display_height = display_height
        self.headless = headless

        # Source type detection
        self.is_webcam = isinstance(source, int)

        # Video capture & writer state
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.output_filepath: Optional[Path] = self._resolve_output_path(output_path)

        # Runtime control flags
        self.is_paused = False
        self.is_running = True

        # FPS metrics
        self.fps = 0.0
        self.prev_time = time.perf_counter()
        self.fps_alpha = 0.1  # Smoothing factor for FPS display

        # Statistics
        self.total_frames_processed = 0

    def _resolve_output_path(self, output_path: Optional[str]) -> Path:
        """Determines and creates valid output destination path."""
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return config.OUTPUT_DIR / f"tracked_{timestamp}.mp4"

    def _open_stream(self) -> None:
        """
        Initializes the OpenCV VideoCapture object with comprehensive error checks.
        """
        source_desc = f"webcam (ID: {self.source})" if self.is_webcam else f"video file '{self.source}'"
        logger.info(f"Opening video source: {source_desc}...")

        if self.is_webcam:
            # On Windows, using cv2.CAP_DSHOW can prevent delay or capture locks
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    logger.warning("CAP_DSHOW failed, falling back to default capture API...")
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            if self.is_webcam:
                error_msg = (
                    f"Could not open webcam index {self.source}.\n"
                    "Troubleshooting steps:\n"
                    "  1. Ensure your camera is plugged in and recognized in Device Manager.\n"
                    "  2. Check Windows Camera Privacy Settings (Settings -> Privacy & Security -> Camera -> Allow desktop apps).\n"
                    "  3. Ensure no other application (Zoom, Teams, Skype, Browser) is currently using the camera.\n"
                    "  4. Try specifying a different index: e.g., '--source 1' or '--source 2'."
                )
            else:
                error_msg = (
                    f"Could not open or decode video file '{self.source}'.\n"
                    "Troubleshooting steps:\n"
                    "  1. Verify the file exists and is not corrupted.\n"
                    "  2. Verify the format is supported (.mp4, .avi, .mov, .mkv).\n"
                    "  3. Ensure appropriate video codecs are installed."
                )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("Video stream opened successfully.")

    def _init_writer(self, frame_w: int, frame_h: int, fps_hint: float) -> None:
        """Initializes OpenCV VideoWriter for saving processed video."""
        fps = fps_hint if (fps_hint > 0 and fps_hint <= 120) else 25.0
        fourcc = cv2.VideoWriter_fourcc(*config.DEFAULT_FOURCC)
        out_str = str(self.output_filepath)

        logger.info(f"Initializing video writer: {out_str} ({frame_w}x{frame_h} @ {fps:.1f} FPS)...")
        self.writer = cv2.VideoWriter(out_str, fourcc, fps, (frame_w, frame_h))

        if not self.writer.isOpened():
            # Fallback codec if mp4v fails
            fallback_fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fallback_path = str(self.output_filepath.with_suffix(".avi"))
            logger.warning(f"mp4v codec failed. Falling back to XVID codec: {fallback_path}")
            self.writer = cv2.VideoWriter(fallback_path, fallback_fourcc, fps, (frame_w, frame_h))
            self.output_filepath = Path(fallback_path)

        if not self.writer.isOpened():
            logger.error("Failed to initialize video writer. Video saving will be skipped.")
            self.writer = None
        else:
            logger.info(f"Recording output enabled: {self.output_filepath}")

    def run(self) -> None:
        """
        Executes the main video processing loop.
        """
        try:
            self._open_stream()
        except RuntimeError as e:
            logger.error(f"Stream initialization failed: {e}")
            return

        fps_hint = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        logger.info("Video processing loop started. Press [Q] to quit, [P] to pause, [S] to screenshot.")

        try:
            while self.is_running:
                if not self.is_paused:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        if self.is_webcam:
                            logger.warning("Empty frame received from webcam. Retrying...")
                            time.sleep(0.05)
                            continue
                        else:
                            logger.info("End of video stream reached.")
                            break

                    # Optional resizing for performance or display
                    if self.display_width and self.display_height:
                        frame = cv2.resize(
                            frame,
                            (self.display_width, self.display_height),
                            interpolation=cv2.INTER_LINEAR
                        )

                    frame_h, frame_w = frame.shape[:2]

                    # Initialize writer on first frame if saving is enabled
                    if self.save_output and self.writer is None:
                        self._init_writer(frame_w, frame_h, fps_hint)

                    # 1. Update Detection & ByteTrack Tracking
                    raw_detections, tracked_objects = self.tracker.update(frame)

                    # 2. Draw Bounding Boxes and Labels for each tracked object
                    # Format: "Class | ID: # | Conf"
                    for track in tracked_objects:
                        color = get_color_for_id(track.track_id)
                        label = f"{track.class_name} | ID: {track.track_id} | {track.confidence:.2f}"
                        draw_bounding_box(frame, track.bbox, label, color)

                    # If an object was detected but not yet assigned a track ID by ByteTrack
                    # (e.g. first frame of appearance), optionally draw subtle box:
                    tracked_boxes = [t.bbox for t in tracked_objects]
                    for det in raw_detections:
                        if det.bbox not in tracked_boxes:
                            # Draw detection with no ID badge
                            label = f"{det.class_name} | {det.confidence:.2f}"
                            draw_bounding_box(frame, det.bbox, label, (180, 180, 180))

                    # 3. Calculate Real-Time FPS
                    curr_time = time.perf_counter()
                    delta = curr_time - self.prev_time
                    self.prev_time = curr_time

                    if delta > 0:
                        instant_fps = 1.0 / delta
                        self.fps = (self.fps * (1.0 - self.fps_alpha)) + (instant_fps * self.fps_alpha)

                    # 4. Draw HUD (Heads-Up Display)
                    draw_hud(
                        frame=frame,
                        fps=self.fps,
                        detected_count=len(raw_detections),
                        tracked_count=len(tracked_objects),
                        device_name=self.detector.device_display,
                        is_paused=self.is_paused
                    )

                    # 5. Write frame to output video if enabled
                    if self.writer is not None:
                        self.writer.write(frame)

                    self.total_frames_processed += 1
                    current_display_frame = frame

                # 6. Display frame (if not headless)
                if not self.headless:
                    cv2.imshow(config.WINDOW_TITLE, current_display_frame)

                # 7. Handle keyboard input
                wait_time = 0 if self.is_paused else 1
                if not self.headless:
                    key = cv2.waitKey(wait_time) & 0xFF
                else:
                    key = 255
                    time.sleep(0.001)

                if key in (config.KEY_QUIT_LOWER, config.KEY_QUIT_UPPER, config.KEY_QUIT_ESC):
                    logger.info("Exit key pressed. Terminating video loop...")
                    break

                elif key in (config.KEY_PAUSE_LOWER, config.KEY_PAUSE_UPPER):
                    self.is_paused = not self.is_paused
                    logger.info(f"Stream state toggled: {'PAUSED' if self.is_paused else 'RESUMED'}")

                elif key in (config.KEY_SAVE_LOWER, config.KEY_SAVE_UPPER):
                    if current_display_frame is not None:
                        saved_path = save_screenshot(current_display_frame)
                        logger.info(f"Frame saved successfully to: {saved_path}")

                elif key in (config.KEY_RESET_LOWER, config.KEY_RESET_UPPER):
                    self.tracker.reset()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping...")

        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Releases video streams, writers, and OpenCV display windows."""
        logger.info("Cleaning up resources...")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info(f"Recorded video successfully finalized and saved to: {self.output_filepath}")

        if not self.headless:
            cv2.destroyAllWindows()

        logger.info(f"Session finished. Total frames processed: {self.total_frames_processed}")
