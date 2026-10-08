"""
Utility functions, data classes, visualization helpers, and device selection
for the Real-Time Object Detection and Tracking System.
"""

import sys
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np

import config


# ==========================================
# Logging Configuration
# ==========================================
class CustomFormatter(logging.Formatter):
    """Clean formatter displaying [LEVEL] Message without verbose boilerplates."""
    def format(self, record: logging.LogRecord) -> str:
        level_tag = f"[{record.levelname}]"
        return f"{level_tag:<8} {record.getMessage()}"


def setup_logger(name: str = "ObjectTracker", level: int = logging.INFO) -> logging.Logger:
    """Sets up and returns a standardized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()


# ==========================================
# Data Structures
# ==========================================
@dataclass
class Detection:
    """Represents a single raw object detection from YOLO."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedObject:
    """Represents an object that has been identified and tracked across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


# ==========================================
# Device Selection
# ==========================================
def get_device(preference: Optional[str] = "auto") -> Tuple[str, str]:
    """
    Detects hardware capabilities and selects optimal compute device.

    Returns:
        Tuple of (device_string, device_display_name)
        e.g. ("cuda:0", "CUDA (NVIDIA GeForce RTX 4070)") or ("cpu", "CPU")
    """
    if preference and preference.lower() == "cpu":
        return "cpu", "CPU"

    try:
        import torch
        if torch.cuda.is_available() and (preference is None or preference.lower() in ("auto", "cuda", "gpu")):
            device_name = torch.cuda.get_device_name(0)
            return "cuda", f"CUDA ({device_name})"
    except ImportError:
        pass

    return "cpu", "CPU"


# ==========================================
# Input Source Validation
# ==========================================
def parse_and_validate_source(source_input: str) -> Union[int, str]:
    """
    Parses and validates the input source argument.

    Supports:
      - 'webcam', '0', '1', etc. -> integer camera index
      - File path -> checks file existence and video format extension

    Returns:
        int for camera index or str for verified video file path.
    """
    source_str = str(source_input).strip()

    if source_str.lower() in ("webcam", "camera", "cam"):
        return config.DEFAULT_CAMERA_ID

    if source_str.isdigit():
        return int(source_str)

    # Must be a video file path
    video_path = Path(source_str)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found at: '{video_path}'. "
            f"Please verify the file path or place your video in the '{config.INPUT_DIR}' directory."
        )

    if video_path.is_dir():
        raise IsADirectoryError(
            f"Provided source '{video_path}' is a directory, not a video file."
        )

    ext = video_path.suffix.lower()
    if ext not in config.SUPPORTED_VIDEO_EXTENSIONS:
        logger.warning(
            f"File extension '{ext}' is not in standard list {config.SUPPORTED_VIDEO_EXTENSIONS}. "
            "Attempting to open with OpenCV anyway..."
        )

    return str(video_path.resolve())


# ==========================================
# Visualization & Drawing Helpers
# ==========================================
def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministically selects a distinct BGR color from the palette for a tracking ID."""
    return config.COLOR_PALETTE[abs(track_id) % len(config.COLOR_PALETTE)]


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int]
) -> None:
    """
    Draws a crisp bounding box with a rounded label badge for clear readability.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Clamp coordinates to frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return

    # Draw main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS, cv2.LINE_AA)

    # Add corner accent brackets for a modern CV aesthetic
    bracket_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
    thick = config.BOX_THICKNESS + 1
    if bracket_len > 3:
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, thick)
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, thick)
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, thick)
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, thick)

    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, config.FONT_SCALE, 1)

    # Determine badge position (place above box if room, otherwise inside)
    badge_y2 = y1 if y1 - th - baseline - 6 >= 0 else y1 + th + baseline + 6
    badge_y1 = badge_y2 - (th + baseline + 6)
    badge_x1 = x1
    badge_x2 = min(w - 1, x1 + tw + 10)

    # Draw label badge background
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)

    # Text label in contrasting black or white
    text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 130 else (255, 255, 255)
    cv2.putText(
        frame,
        label,
        (badge_x1 + 5, badge_y2 - baseline - 3),
        font,
        config.FONT_SCALE,
        text_color,
        1,
        cv2.LINE_AA
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    detected_count: int,
    tracked_count: int,
    device_name: str,
    is_paused: bool = False
) -> None:
    """
    Renders an elegant HUD panel showing FPS, detection counts, tracking counts,
    active device, and pause status.
    """
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 340
    panel_h = 135
    margin = 12

    # Draw semi-transparent background overlay
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (margin + panel_w, margin + panel_h),
        config.HUD_BG_COLOR,
        -1
    )
    cv2.addWeighted(overlay, config.HUD_ALPHA, frame, 1.0 - config.HUD_ALPHA, 0, frame)

    # Draw subtle accent border around HUD
    cv2.rectangle(
        frame,
        (margin, margin),
        (margin + panel_w, margin + panel_h),
        (70, 70, 70),
        1,
        cv2.LINE_AA
    )

    # Texts inside HUD
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_text = margin + 14
    y_start = margin + 24
    line_spacing = 22

    # Title header
    cv2.putText(
        frame,
        "TASK 4: OBJECT TRACKER",
        (x_text, y_start),
        font,
        0.55,
        config.HUD_ACCENT_COLOR,
        2,
        cv2.LINE_AA
    )

    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    fps_color = (80, 240, 80) if fps >= 20 else ((80, 220, 240) if fps >= 10 else (80, 80, 240))
    cv2.putText(
        frame,
        fps_text,
        (x_text, y_start + line_spacing),
        font,
        0.52,
        fps_color,
        1,
        cv2.LINE_AA
    )

    # Detected & Tracked counts
    stats_text = f"Objects Detected: {detected_count}"
    cv2.putText(
        frame,
        stats_text,
        (x_text, y_start + line_spacing * 2),
        font,
        0.50,
        config.HUD_TEXT_COLOR,
        1,
        cv2.LINE_AA
    )

    tracked_text = f"Objects Tracked:  {tracked_count}"
    cv2.putText(
        frame,
        tracked_text,
        (x_text, y_start + line_spacing * 3),
        font,
        0.50,
        config.HUD_TEXT_COLOR,
        1,
        cv2.LINE_AA
    )

    # Device & Status
    status_str = "PAUSED" if is_paused else "RUNNING"
    status_color = (0, 165, 255) if is_paused else (100, 220, 100)
    device_short = device_name.split()[0]
    dev_text = f"Device: {device_short} | [{status_str}]"
    cv2.putText(
        frame,
        dev_text,
        (x_text, y_start + line_spacing * 4),
        font,
        0.45,
        status_color,
        1,
        cv2.LINE_AA
    )

    # Bottom bar control guide
    guide_bar_y = h - 16
    guide_text = "[Q/ESC] Quit  |  [P] Pause  |  [S] Screenshot  |  [R] Reset Tracks"
    cv2.putText(
        frame,
        guide_text,
        (margin, guide_bar_y),
        font,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )


def save_screenshot(frame: np.ndarray, output_dir: Optional[Path] = None) -> str:
    """
    Saves the current frame to the output directory as a timestamped JPEG.
    """
    out_dir = output_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"screenshot_{timestamp}.jpg"
    filepath = out_dir / filename

    cv2.imwrite(str(filepath), frame)
    logger.info(f"Screenshot saved: {filepath}")
    return str(filepath)
