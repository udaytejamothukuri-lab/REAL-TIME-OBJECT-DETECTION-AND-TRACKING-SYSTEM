"""
Object Detection Module using Ultralytics YOLO.
Handles model loading, inference, filtering by confidence, and returning
standardized Detection objects.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from ultralytics import YOLO

import config
from utils import Detection, get_device, logger


class YOLOObjectDetector:
    """
    Wraps Ultralytics YOLO models for real-time frame inference.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        conf_thresh: float = config.CONFIDENCE_THRESHOLD,
        iou_thresh: float = config.IOU_THRESHOLD,
        device: Optional[str] = "auto",
        allowed_classes: Optional[List[str]] = None,
    ):
        """
        Initializes the YOLO object detector.

        Args:
            model_name: Name or path to the YOLO model (e.g., 'yolo11n.pt', 'yolov8n.pt').
            conf_thresh: Minimum confidence score for detection filtering.
            iou_thresh: IoU threshold for Non-Maximum Suppression (NMS).
            device: 'cuda', 'cpu', or 'auto' for automatic hardware detection.
            allowed_classes: Optional list of class names to filter (e.g. ['person', 'car']).
        """
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.device_str, self.device_display = get_device(device)
        self.model_name = model_name

        logger.info(f"Using compute device: {self.device_display}")
        logger.info(f"Loading YOLO model '{model_name}'...")

        self.model = self._load_model(model_name)
        self.names = self.model.names if hasattr(self.model, "names") else {}

        # Configure optional class filtering
        self.filter_class_ids = self._resolve_class_ids(allowed_classes)
        if self.filter_class_ids:
            names_filtered = [self.names[cid] for cid in self.filter_class_ids if cid in self.names]
            logger.info(f"Class filtering active for: {names_filtered}")

        logger.info(f"Model successfully loaded. Classes available: {len(self.names)}")

    def _load_model(self, model_name: str) -> YOLO:
        """
        Loads the YOLO model with automatic fallback to stable models if needed.
        """
        # Check if the model exists in the models/ directory
        local_model_path = config.MODELS_DIR / model_name
        target_path = str(local_model_path) if local_model_path.exists() else model_name

        try:
            model = YOLO(target_path)
            # Move to target device if supported
            if self.device_str != "cpu":
                try:
                    model.to(self.device_str)
                except Exception as e:
                    logger.warning(f"Could not transfer model to {self.device_str}: {e}. Running on CPU.")
                    self.device_str = "cpu"
                    self.device_display = "CPU"
            return model
        except Exception as e:
            logger.warning(
                f"Failed to load '{model_name}': {e}. "
                f"Attempting fallback to '{config.FALLBACK_MODEL_NAME}'..."
            )
            try:
                fallback_model = YOLO(config.FALLBACK_MODEL_NAME)
                self.model_name = config.FALLBACK_MODEL_NAME
                return fallback_model
            except Exception as fb_err:
                logger.error(f"Critical: Failed to load fallback model: {fb_err}")
                raise RuntimeError(
                    f"Unable to initialize YOLO model. Please verify internet access for initial download "
                    f"or place '{model_name}' inside '{config.MODELS_DIR}'."
                ) from fb_err

    def _resolve_class_ids(self, class_list: Optional[List[str]]) -> Optional[List[int]]:
        """Resolves string class names to integer class IDs."""
        if not class_list:
            return None

        normalized_requested = [c.strip().lower() for c in class_list if c.strip()]
        if not normalized_requested:
            return None

        matched_ids = []
        for class_id, class_name in self.names.items():
            if str(class_name).lower() in normalized_requested or str(class_id) in normalized_requested:
                matched_ids.append(int(class_id))

        if not matched_ids:
            logger.warning(f"None of the requested classes {class_list} were found in model dictionary!")
            return None

        return matched_ids

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs YOLO object detection on a single video frame.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            List of Detection instances matching the confidence threshold.
        """
        if frame is None or frame.size == 0:
            return []

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device_str,
            classes=self.filter_class_ids,
            verbose=False
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract coordinates, confidences, and classes
        xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(xyxy_coords, confidences, class_ids):
            conf_val = float(conf)
            if conf_val < self.conf_thresh:
                continue

            x1, y1, x2, y2 = bbox
            cls_name = self.names.get(int(cls_id), f"cls_{cls_id}")

            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf_val,
                    class_id=int(cls_id),
                    class_name=cls_name
                )
            )

        return detections
