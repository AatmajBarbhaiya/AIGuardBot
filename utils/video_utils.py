"""
Video/Camera utility functions for webcam handling
"""
import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages webcam access and frame capture"""

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_active = False
        self.frame_count = 0

    def start(self):
        """Initialize and start camera"""
        try:
            logger.info(f"Starting camera {self.camera_id}...")
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.is_active = True
            logger.info("Camera started successfully")
            return True

        except Exception as e:
            logger.error(f"Camera start error: {e}")
            return False

    def stop(self):
        """Stop and release camera"""
        try:
            if self.cap is not None:
                self.cap.release()
            self.is_active = False
            cv2.destroyAllWindows()
            logger.info("Camera stopped")
        except Exception as e:
            logger.error(f"Camera stop error: {e}")

    def read_frame(self):
        """Capture a single frame from camera"""
        if not self.is_active or self.cap is None:
            logger.warning("Camera not active")
            return None

        try:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                return frame
            else:
                logger.warning("Failed to read frame")
                return None

        except Exception as e:
            logger.error(f"Frame read error: {e}")
            return None

    def display_frame(self, frame, window_name="AI Guard - Camera Feed"):
        """Display frame in a window"""
        if frame is None:
            return

        # Add timestamp overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)