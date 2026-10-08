# Pretrained Models Directory

This directory stores pretrained YOLO detection models (e.g., `yolo11n.pt`, `yolov8n.pt`).

## Automatic Download
Ultralytics automatically downloads the official pretrained model weights on the first run if they are not already present in this directory or the working environment.

## Supported Models
- **YOLO11**: `yolo11n.pt` (Nano - fastest), `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- **YOLOv8**: `yolov8n.pt` (Nano), `yolov8s.pt`, `yolov8m.pt`

By default, the application uses `yolo11n.pt` for optimal real-time performance on both CPU and GPU.
