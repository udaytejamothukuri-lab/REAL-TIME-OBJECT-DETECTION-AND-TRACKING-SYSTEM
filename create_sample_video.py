"""
Utility to generate a sample video file for testing detection and tracking
without requiring an active webcam or external download.
Generates an MP4 video with moving shapes and labels into input/sample_video.mp4.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_sample_video(
    output_path: str = "input/sample_video.mp4",
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    duration_sec: int = 5
) -> str:
    """
    Creates a sample video with moving synthetic objects.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height))

    total_frames = fps * duration_sec

    # Object trajectories
    # Object 1: Moves left to right
    # Object 2: Moves top-right to bottom-left
    # Object 3: Bounces around
    for i in range(total_frames):
        # Create dark road/environment background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # Asphalt grey

        # Draw road lanes
        cv2.line(frame, (0, height // 2), (width, height // 2), (255, 255, 255), 2)
        cv2.line(frame, (0, height // 2 + 80), (width, height // 2 + 80), (100, 100, 100), 1)

        # Object 1 (Car-like rectangle moving horizontally)
        x1 = int(20 + (i * 4) % (width - 120))
        y1 = height // 2 - 60
        cv2.rectangle(frame, (x1, y1), (x1 + 90, y1 + 50), (40, 60, 220), -1)
        cv2.putText(frame, "Car 1", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Object 2 (Pedestrian-like vertical rectangle moving slowly)
        x2 = int(width - 60 - (i * 2) % (width - 100))
        y2 = height // 2 + 10
        cv2.rectangle(frame, (x2, y2), (x2 + 35, y2 + 65), (50, 180, 50), -1)
        cv2.putText(frame, "Pedestrian", (x2 - 15, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Object 3 (Another vehicle moving in bottom lane)
        x3 = int(width - 120 - (i * 5) % (width - 150))
        y3 = height // 2 + 90
        cv2.rectangle(frame, (x3, y3), (x3 + 110, y3 + 55), (200, 140, 30), -1)
        cv2.putText(frame, "Car 2", (x3 + 10, y3 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Frame counter watermark
        cv2.putText(
            frame,
            f"Sample Test Footage - Frame {i+1}/{total_frames}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

        writer.write(frame)

    writer.release()
    print(f"Sample video created: {out_file.resolve()} ({total_frames} frames, {width}x{height})")
    return str(out_file.resolve())


if __name__ == "__main__":
    generate_sample_video()
