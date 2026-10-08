"""
Object Tracking Module implementing ByteTrack via Ultralytics YOLO framework.
Manages multi-object tracking, ID assignment, trajectory histories, and track persistence.
"""

from typing import List, Tuple, Dict, Optional, Deque
from collections import defaultdict, deque
import numpy as np

import config
from utils import Detection, TrackedObject, logger
from detector import YOLOObjectDetector


class ByteTrackManager:
    """
    Manages multi-target tracking across consecutive video frames using ByteTrack.
    Associates high and low confidence detections with existing tracks using Kalman
    filtering and IoU matching to maintain stable track IDs.
    """

    def __init__(
        self,
        detector: YOLOObjectDetector,
        tracker_type: str = config.TRACKER_TYPE,
        max_trajectory_len: int = 30
    ):
        """
        Initializes the ByteTrack tracker manager.

        Args:
            detector: Instance of YOLOObjectDetector containing the active model.
            tracker_type: Configuration name for the tracker (e.g., 'bytetrack.yaml').
            max_trajectory_len: Number of historical centroid positions to retain for paths.
        """
        self.detector = detector
        self.tracker_type = tracker_type
        self.max_trajectory_len = max_trajectory_len

        # Store trajectory centroids: track_id -> deque of (cx, cy)
        self.trajectories: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.max_trajectory_len)
        )

        # Unique active and cumulative IDs
        self.cumulative_ids_seen = set()
        logger.info(f"Initialized ByteTrack tracking engine with config: '{tracker_type}'")

    def update(self, frame: np.ndarray) -> Tuple[List[Detection], List[TrackedObject]]:
        """
        Processes a frame with YOLO detection and ByteTrack association.

        Args:
            frame: Current video frame (numpy BGR image).

        Returns:
            Tuple of:
              - raw_detections: List[Detection] (all objects passing detection threshold)
              - tracked_objects: List[TrackedObject] (objects actively tracked with stable IDs)
        """
        if frame is None or frame.size == 0:
            return [], []

        # Run model tracking with persist=True for inter-frame temporal association
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.detector.conf_thresh,
            iou=self.detector.iou_thresh,
            device=self.detector.device_str,
            classes=self.detector.filter_class_ids,
            verbose=False
        )

        raw_detections: List[Detection] = []
        tracked_objects: List[TrackedObject] = []

        if not results or len(results) == 0:
            return raw_detections, tracked_objects

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return raw_detections, tracked_objects

        # Extract box attributes
        xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for idx, (bbox, conf, cls_id) in enumerate(zip(xyxy_coords, confidences, class_ids)):
            conf_val = float(conf)
            if conf_val < self.detector.conf_thresh:
                continue

            x1, y1, x2, y2 = bbox
            cls_name = self.detector.names.get(int(cls_id), f"cls_{cls_id}")

            # Record as raw detection
            raw_detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf_val,
                    class_id=int(cls_id),
                    class_name=cls_name
                )
            )

            # Check if tracker assigned an ID
            if track_ids is not None and idx < len(track_ids):
                tid = int(track_ids[idx])
                tracked_objects.append(
                    TrackedObject(
                        track_id=tid,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf_val,
                        class_id=int(cls_id),
                        class_name=cls_name
                    )
                )

                # Update trajectory centroid
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                self.trajectories[tid].append((cx, cy))
                self.cumulative_ids_seen.add(tid)

        return raw_detections, tracked_objects

    def reset(self) -> None:
        """
        Resets tracking state, clearing active trajectories and reinitializing tracker.
        """
        logger.info("Resetting tracker state and clearing tracking IDs...")
        self.trajectories.clear()
        self.cumulative_ids_seen.clear()

        # Reset predictor tracker state inside Ultralytics YOLO model
        if hasattr(self.detector.model, "predictor") and self.detector.model.predictor is not None:
            if hasattr(self.detector.model.predictor, "trackers"):
                self.detector.model.predictor.trackers = None
        logger.info("Tracker state reset successfully.")
