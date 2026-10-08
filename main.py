"""
Real-Time Object Detection and Tracking System
Academic Task 4 Implementation.

Entry point script featuring full Command Line Interface (CLI),
dynamic device selection, YOLO detection, and ByteTrack multi-object tracking.
"""

import sys
import argparse
from pathlib import Path

import config
from utils import logger, parse_and_validate_source
from detector import YOLOObjectDetector
from tracker import ByteTrackManager
from video_processor import VideoProcessor


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="Real-Time Object Detection and Tracking",
        description=(
            "Academic Task 4: Real-Time Object Detection & Tracking System using "
            "Ultralytics YOLO (YOLO11/YOLOv8) and ByteTrack."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Run live webcam tracking (Default camera):
  python main.py --source webcam
  python main.py --source 0

  # 2. Process a local video file:
  python main.py --source input/traffic.mp4

  # 3. Process video and save output recording to output/:
  python main.py --source input/traffic.mp4 --save

  # 4. Filter specific classes with custom confidence threshold:
  python main.py --source 0 --classes person car --confidence 0.50

  # 5. Specify custom model and target device:
  python main.py --source 0 --model yolo11n.pt --device cuda

Keyboard Controls During Playback:
  [Q] or [ESC] : Quit application
  [P]          : Pause / Resume playback
  [S]          : Capture screenshot (saved to output/ folder)
  [R]          : Reset active tracks and trajectories
"""
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: 'webcam', camera index (0, 1), or path to video file (.mp4, .avi, etc.). Default: 0"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=f"YOLO model weight name or path (e.g., 'yolo11n.pt', 'yolov8n.pt'). Default: '{config.MODEL_NAME}'"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Detection confidence score threshold [0.0 - 1.0]. Default: {config.CONFIDENCE_THRESHOLD}"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=config.IOU_THRESHOLD,
        help=f"IoU threshold for Non-Maximum Suppression (NMS). Default: {config.IOU_THRESHOLD}"
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default=config.TRACKER_TYPE,
        help=f"Tracker configuration. Default: '{config.TRACKER_TYPE}'"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        default=config.SAVE_OUTPUT,
        help="Record and save the annotated tracked video to the output directory."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output file path for recorded video."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device preference: 'auto' (detects GPU), 'cuda', or 'cpu'. Default: 'auto'"
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional space-separated list of object classes to track (e.g. '--classes person car dog')."
    )

    parser.add_argument(
        "--width",
        type=int,
        default=config.INPUT_WIDTH,
        help="Optional display width resize (e.g. 1280)."
    )

    parser.add_argument(
        "--height",
        type=int,
        default=config.INPUT_HEIGHT,
        help="Optional display height resize (e.g. 720)."
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a graphical window (useful for headless servers / test suites)."
    )

    return parser


def main() -> int:
    """Main execution routine."""
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 65)
    print("  REAL-TIME OBJECT DETECTION AND TRACKING SYSTEM (TASK 4)")
    print("=" * 65)

    # 1. Validate Input Source
    try:
        source_target = parse_and_validate_source(args.source)
    except (FileNotFoundError, IsADirectoryError, ValueError) as err:
        logger.error(f"Input validation error: {err}")
        return 1

    # 2. Validate Confidence & IoU
    if not (0.0 <= args.confidence <= 1.0):
        logger.error(f"Confidence threshold must be between 0.0 and 1.0. Given: {args.confidence}")
        return 1

    if not (0.0 <= args.iou <= 1.0):
        logger.error(f"IoU threshold must be between 0.0 and 1.0. Given: {args.iou}")
        return 1

    # 3. Initialize YOLO Detector
    try:
        detector = YOLOObjectDetector(
            model_name=args.model,
            conf_thresh=args.confidence,
            iou_thresh=args.iou,
            device=args.device,
            allowed_classes=args.classes,
        )
    except Exception as err:
        logger.error(f"Failed to initialize YOLO detector: {err}")
        return 1

    # 4. Initialize ByteTrack Tracker
    try:
        tracker = ByteTrackManager(
            detector=detector,
            tracker_type=args.tracker
        )
    except Exception as err:
        logger.error(f"Failed to initialize ByteTrack tracker: {err}")
        return 1

    # 5. Initialize Video Processor Pipeline
    try:
        processor = VideoProcessor(
            source=source_target,
            detector=detector,
            tracker=tracker,
            save_output=args.save or (args.output is not None),
            output_path=args.output,
            display_width=args.width,
            display_height=args.height,
            headless=args.headless,
        )
    except Exception as err:
        logger.error(f"Failed to initialize video processing pipeline: {err}")
        return 1

    # 6. Run Pipeline
    try:
        processor.run()
    except Exception as err:
        logger.error(f"An unexpected error occurred during processing: {err}")
        return 1

    logger.info("Application execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
