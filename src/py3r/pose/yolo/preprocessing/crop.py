from typing import Tuple

import numpy as np

from py3r.pose.yolo.preprocessing.pipeline import Transform


class Crop(Transform):
    """
    Crop by rectangle (x, y, w, h) where (x,y) is top-left in input coords.
    Works for gray (H,W) or color (H,W,3) np.uint8.
    """
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)

    def forward(self, img: np.ndarray):
        h, w = img.shape[:2]
        x0 = np.clip(self.x, 0, w)
        y0 = np.clip(self.y, 0, h)
        x1 = np.clip(self.x + self.w, 0, w)
        y1 = np.clip(self.y + self.h, 0, h)
        cropped = img[y0:y1, x0:x1].copy()
        meta = {"crop": (x0, y0), "orig_hw": (h, w), "new_hw": cropped.shape[:2]}
        return cropped, meta

    def invert_point(self, xy: Tuple[float, float], meta: dict) -> Tuple[float, float]:
        x0, y0 = meta["crop"]
        x, y = xy
        return x + x0, y + y0
