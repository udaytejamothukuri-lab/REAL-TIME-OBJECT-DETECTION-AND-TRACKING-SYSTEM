# Real-Time Object Detection and Tracking System

An AI-powered computer vision application implementing real-time multi-object detection and tracking across video streams (webcam and local video files) using **Ultralytics YOLO** and the **ByteTrack** tracking algorithm. Built for academic Task 4.

---

## 1. Project Overview

Object detection identifies and localizes objects within single isolated frames, but fails to maintain continuity across temporal sequences. This project bridges that gap by combining state-of-the-art **YOLO** deep learning detection with **ByteTrack** temporal association. 

The application:
1. Captures live video from a webcam or loads pre-recorded video files.
2. Performs real-time inference on each frame using a pretrained YOLO model (YOLO11 / YOLOv8).
3. Associates detected bounding boxes across consecutive frames using Kalman filtering and bipartite matching (ByteTrack).
4. Assigns and maintains unique, persistent Tracking IDs for each entity.
5. Displays a real-time Heads-Up Display (HUD) showing live FPS, active detection counts, tracked object counts, and hardware device acceleration details.
6. Allows interactive keyboard controls (Pause, Screenshot, Tracker Reset, Quit) and optional video recording to disk.

---

## 2. Features

- **Dual Video Input Support**: Seamlessly processes real-time webcam streams (`--source webcam` or `--source 0`) or local video files (`.mp4`, `.avi`, `.mov`, `.mkv`).
- **Pretrained YOLO Deep Learning**: Uses the latest Ultralytics models (`yolo11n.pt` / `yolov8n.pt`) with automatic weights download.
- **Robust ByteTrack Association**: Tracks objects through temporary occlusions, scale variations, and camera movements while retaining consistent track IDs.
- **Real-Time Heads-Up Display (HUD)**: Semi-transparent glassmorphism overlay displaying real-time smoothed FPS, number of objects detected, and number of objects actively tracked.
- **Dynamic Color Assignment**: High-contrast, deterministic color palette generated per tracking ID for visual distinction.
- **Interactive Controls**:
  - `[P]`: Pause / resume video playback.
  - `[S]`: Instant screenshot capture saved directly to `output/`.
  - `[R]`: Reset tracker state and re-initialize trajectories.
  - `[Q]` or `[ESC]`: Clean shutdown and release of capture devices.
- **Video Export (`--save`)**: Encodes and records the annotated video with bounding boxes and tracking IDs into the `output/` directory using standard MP4/AVI codecs.
- **Hardware Acceleration**: Automatic GPU acceleration detection (NVIDIA CUDA) with fallback to multi-threaded CPU execution.
- **Class Filtering**: Selectively track target classes (e.g., `--classes person car`).

---

## 3. Technologies Used

- **Python 3.10+**: Core programming language.
- **OpenCV (`opencv-python`)**: Video capture, frame preprocessing, bounding box drawing, text rendering, and video recording.
- **Ultralytics YOLO**: Real-time object detection framework with pretrained COCO weights.
- **ByteTrack**: Multi-object tracker utilizing Kalman filters and two-stage Hungarian association for both high- and low-confidence detections.
- **NumPy**: Matrix computations, frame manipulation, and coordinate transformations.
- **PyTorch**: Deep learning inference engine with automatic CUDA/CPU dispatch.

---

## 4. System Architecture

```
                      ┌────────────────────────────┐
                      │    Video Input Source      │
                      │  (Webcam / Local Video)    │
                      └─────────────┬──────────────┘
                                    │ cv2.VideoCapture
                                    ▼
                      ┌────────────────────────────┐
                      │    Frame Preprocessing     │
                      │  (Optional resize/format)  │
                      └─────────────┬──────────────┘
                                    │
                                    ▼
                      ┌────────────────────────────┐
                      │    YOLO Object Detector    │
                      │  (Confidence Filtering)    │
                      └─────────────┬──────────────┘
                                    │ Detections [bbox, conf, class]
                                    ▼
                      ┌────────────────────────────┐
                      │    ByteTrack Tracking      │
                      │  (Kalman Filter + Matching)│
                      └─────────────┬──────────────┘
                                    │ Tracked Objects [ID, bbox, conf, class]
                                    ▼
                      ┌────────────────────────────┐
                      │     HUD & Visualization    │
                      │  (Boxes, Badges, FPS, IDs) │
                      └─────────────┬──────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐
       │     OpenCV Display GUI    │ │   Video Output / Save     │
       │   (Real-Time cv2.imshow)  │ │  (cv2.VideoWriter .mp4)   │
       └───────────────────────────┘ └───────────────────────────┘
```

---

## 5. Installation Guide

### Step 1: Clone or Open Workspace
Open your terminal in the project directory:
```bash
cd "REAL-TIME OBJECT DETECTION AND TRACKING SYSTEM"
```

### Step 2: Create a Virtual Environment
On Windows:
```cmd
python -m venv venv
```

On Linux / macOS:
```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment
On Windows (Command Prompt):
```cmd
venv\Scripts\activate
```
On Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```
On Linux / macOS:
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Running the Application

### Mode 1 — Webcam
Run live tracking on your default webcam (Camera 0):
```bash
python main.py --source webcam
```
or specify the device index directly:
```bash
python main.py --source 0
```

### Mode 2 — Local Video File
Process a pre-recorded video:
```bash
python main.py --source input/sample_video.mp4
```

*(You can generate a test synthetic video at any time by running `python create_sample_video.py`)*

### Mode 3 — Save Processed Video Output
Record the annotated video to `output/`:
```bash
python main.py --source input/sample_video.mp4 --save
```
Or specify a custom output path:
```bash
python main.py --source 0 --save --output output/webcam_session.mp4
```

### Mode 4 — Filter Specific Object Classes
Detect and track only pedestrians and cars:
```bash
python main.py --source input/sample_video.mp4 --classes person car
```

### Mode 5 — Adjust Confidence Threshold & Target Hardware
Run with a stricter confidence threshold (0.60) on CPU:
```bash
python main.py --source 0 --confidence 0.60 --device cpu
```

---

## 7. Keyboard Controls

| Key | Action | Description |
|---|---|---|
| `Q` or `ESC` | **Quit** | Gracefully closes the video stream, finalizes any active video recording, and exits. |
| `P` | **Pause / Resume** | Freezes video playback to inspect frame detections and resume when ready. |
| `S` | **Save Frame** | Captures the current annotated frame as a timestamped JPEG into `output/`. |
| `R` | **Reset Tracker** | Clears active tracking histories and re-initializes ByteTrack IDs. |

---

## 8. Hardware Requirements

- **CPU Mode (Universal Compatibility)**:
  - Works on any standard laptop or desktop with Python 3.10+.
  - Average frame rates with YOLO11 Nano: 25–45 FPS on modern CPUs.
- **GPU Mode (Automatic Acceleration)**:
  - Requires an NVIDIA GPU with CUDA support and CUDA-enabled PyTorch.
  - Automatically detected on startup (`Device: CUDA`).
  - Achieves 60–120+ FPS.

---

## 9. Project Structure

```
REAL-TIME OBJECT DETECTION AND TRACKING SYSTEM/
│
├── main.py                  # CLI interface, argument parser, and pipeline startup
├── detector.py              # YOLOObjectDetector wrapper (inference, confidence filtering)
├── tracker.py               # ByteTrackManager wrapper (association, ID persistence, resets)
├── video_processor.py       # Capture loop, HUD rendering, video recording, hotkeys
├── config.py                # Centralized configurations, defaults, color palettes, hotkeys
├── utils.py                 # Data classes, logger, hardware selector, HUD renderer
├── create_sample_video.py   # Utility generator for synthetic test videos
├── test_suite.py            # Unit test suite covering all modules and failure modes
├── requirements.txt         # Pinned and compatible Python dependencies
├── .gitignore               # Excludes virtual environments, models, and recordings
│
├── models/                  # Stores downloaded YOLO model weights (auto-downloaded)
│   └── README.md
│
├── input/                   # Storage for user input test videos (.mp4, .avi, etc.)
│   └── README.md
│
└── output/                  # Storage for saved recordings and captured screenshots
    └── README.md
```

---

## 10. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **`[ERROR] Could not open webcam index 0`** | Camera is in use or blocked by OS privacy settings. | 1. Ensure camera is plugged in.<br>2. On Windows: Settings -> Privacy -> Camera -> Enable "Allow desktop apps".<br>3. Close Zoom, Teams, or browser tabs using camera.<br>4. Try `--source 1`. |
| **`FileNotFoundError: Video file not found`** | The video path supplied to `--source` does not exist. | Verify path spelling or place video inside `input/` and run `python main.py --source input/your_file.mp4`. |
| **`RuntimeError: Unable to initialize YOLO model`** | Internet connection unavailable for initial weights download. | Connect to the internet once so Ultralytics can download `yolo11n.pt`, or place the weights directly in `models/`. |
| **Low FPS on CPU** | Large model or high input resolution. | 1. Use the default nano model (`yolo11n.pt`).<br>2. Add `--width 640 --height 480` to resize frames. |

---

## 11. Task 4 Requirement Mapping

The following table formally maps every academic Task 4 assignment requirement to its implementation in this codebase:

| Task 4 Requirement | Architectural Implementation | Code Reference |
|---|---|---|
| **Set up real-time video input from webcam or video file** | Implemented using OpenCV `cv2.VideoCapture` with validation for camera indexes and video paths. | [`video_processor.py`](file:///video_processor.py) (`_open_stream`), [`utils.py`](file:///utils.py) (`parse_and_validate_source`) |
| **Use pretrained YOLO object detection model** | Ultralytics YOLO framework with pretrained COCO weights (`yolo11n.pt` / `yolov8n.pt`). | [`detector.py`](file:///detector.py) (`YOLOObjectDetector`) |
| **Process every video frame in real time** | Frame-by-frame processing pipeline with optimized tensor operations and low latency. | [`video_processor.py`](file:///video_processor.py) (`run`) |
| **Detect objects and draw bounding boxes** | Confidence filtering with customizable threshold and OpenCV antialiased box drawing. | [`detector.py`](file:///detector.py) (`detect`), [`utils.py`](file:///utils.py) (`draw_bounding_box`) |
| **Display Class Name, Confidence, and Tracking ID** | Formatted badges above bounding boxes: `Class \| ID: # \| Conf`. | [`video_processor.py`](file:///video_processor.py), [`utils.py`](file:///utils.py) |
| **Track objects across consecutive frames** | ByteTrack multi-target association using Kalman filter prediction and two-stage matching. | [`tracker.py`](file:///tracker.py) (`ByteTrackManager`) |
| **Maintain unique tracking ID across frames** | Persistent tracking IDs retained across frames and occlusions. | [`tracker.py`](file:///tracker.py) (`update`), [`utils.py`](file:///utils.py) (`TrackedObject`) |
| **Provide statistics (FPS, Detected, Tracked)** | High-precision `time.perf_counter()` real-time smoothed FPS calculation and dynamic counters. | [`video_processor.py`](file:///video_processor.py), [`utils.py`](file:///utils.py) (`draw_hud`) |
| **Professional OpenCV User Interface** | Semi-transparent HUD overlay card with statistics and keyboard control hints. | [`utils.py`](file:///utils.py) (`draw_hud`) |
| **Keyboard Controls (Q, ESC, P, S, R)** | Full event loop handling: Quit (`Q`/`ESC`), Pause (`P`), Screenshot (`S`), Tracker Reset (`R`). | [`video_processor.py`](file:///video_processor.py) (`run`) |
| **Video output saving** | Video recording using OpenCV `cv2.VideoWriter` to `output/` directory via `--save`. | [`video_processor.py`](file:///video_processor.py) (`_init_writer`, `run`) |
| **Configuration file** | Centralized parameters in `config.py` without hardcoded constants. | [`config.py`](file:///config.py) |
| **Clean Command Line Interface** | `argparse` CLI with `--source`, `--confidence`, `--save`, `--output`, `--classes`, etc. | [`main.py`](file:///main.py) |
| **Error handling & logging** | Standardized `[INFO]`, `[WARNING]`, `[ERROR]` logging without unhandled tracebacks. | [`utils.py`](file:///utils.py), [`video_processor.py`](file:///video_processor.py) |
| **CPU and GPU support** | Automatic hardware selection (`CUDA` or `CPU`) with display of compute device. | [`utils.py`](file:///utils.py) (`get_device`) |
| **Automated tests and validation** | Complete unit test suite validating all components. | [`test_suite.py`](file:///test_suite.py) |
