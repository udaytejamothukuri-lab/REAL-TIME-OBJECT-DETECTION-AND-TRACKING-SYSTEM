"""
Configuration file for Real-Time Object Detection and Tracking System.
Defines default parameters, model settings, paths, visualization aesthetics,
and keyboard hotkey mappings.
"""

from pathlib import Path

# ==========================================
# Base Directories
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Model & Tracker Configuration
# ==========================================
# Recommended: YOLO11 nano model for optimal real-time performance
# Ultralytics will auto-download yolo11n.pt if not locally present.
MODEL_NAME = "yolo11n.pt"

# Fallback model in case the environment does not support YOLO11
FALLBACK_MODEL_NAME = "yolov8n.pt"

# Default confidence threshold for object detection filtering
CONFIDENCE_THRESHOLD = 0.40

# Default Intersection over Union (IoU) threshold for NMS
IOU_THRESHOLD = 0.45

# Tracker configuration file (ByteTrack algorithm used by Ultralytics)
TRACKER_TYPE = "bytetrack.yaml"

# ==========================================
# Video Stream & Input Configuration
# ==========================================
DEFAULT_CAMERA_ID = 0

# Optional frame resizing (set to None or 0 to keep native resolution)
INPUT_WIDTH = None
INPUT_HEIGHT = None

# Video formats allowed for file input
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm")

# ==========================================
# Output & Recording Configuration
# ==========================================
SAVE_OUTPUT = False
DEFAULT_OUTPUT_FILENAME = "tracked_output.mp4"
DEFAULT_FOURCC = "mp4v"  # High compatibility across platforms (Windows/Linux/macOS)

# ==========================================
# Visualization & HUD Settings
# ==========================================
WINDOW_TITLE = "Real-Time Object Detection & Tracking (Task 4)"
SHOW_FPS = True
SHOW_STATS = True

# Visual styling
BOX_THICKNESS = 2
FONT_SCALE = 0.55
FONT_THICKNESS = 2

# HUD panel styling
HUD_BG_COLOR = (24, 24, 24)        # Dark grey/black overlay
HUD_TEXT_COLOR = (240, 240, 240)    # Crisp off-white
HUD_ACCENT_COLOR = (0, 215, 255)    # Gold/Cyan accent
HUD_ALPHA = 0.70                    # Transparency blend

# Distinct vibrant color palette for visual distinction across tracking IDs
COLOR_PALETTE = [
    (255, 56, 56),    # Red
    (255, 157, 151),  # Coral
    (255, 112, 31),   # Orange
    (255, 178, 29),   # Amber
    (207, 210, 49),   # Lime
    (72, 249, 10),    # Bright Green
    (14, 222, 127),   # Emerald
    (61, 219, 217),   # Cyan
    (49, 173, 249),   # Sky Blue
    (90, 89, 255),    # Indigo
    (155, 78, 255),   # Purple
    (219, 61, 219),   # Magenta
    (255, 89, 174),   # Pink
    (255, 107, 107),  # Rose
    (46, 204, 113),   # Mint
    (52, 152, 219),   # Blue
    (241, 196, 15),   # Yellow
    (230, 126, 34),   # Carrot
    (231, 76, 60),    # Vermilion
    (142, 68, 173),   # Violet
]

# ==========================================
# Keyboard Controls Mapping (ASCII / Keycodes)
# ==========================================
KEY_QUIT_LOWER = ord('q')
KEY_QUIT_UPPER = ord('Q')
KEY_QUIT_ESC = 27
KEY_PAUSE_LOWER = ord('p')
KEY_PAUSE_UPPER = ord('P')
KEY_SAVE_LOWER = ord('s')
KEY_SAVE_UPPER = ord('S')
KEY_RESET_LOWER = ord('r')
KEY_RESET_UPPER = ord('R')
