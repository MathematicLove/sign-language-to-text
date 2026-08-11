from typing import Optional

import cv2
import numpy as np

class Camera:

    def __init__(self, index: int = 0, width: int = 960, height: int = 720) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self.error: str = ""

    @property
    def opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        if self.opened:
            return True
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            self.error = (
                f"Camera {self.index} is not available. Check that no other app is "
                "using it; in Docker you need access to /dev/video* (see README)."
            )
            self._cap = None
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.error = ""
        return True

    def read(self, mirror: bool = True) -> Optional[np.ndarray]:
        if not self.opened:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.flip(frame, 1) if mirror else frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()