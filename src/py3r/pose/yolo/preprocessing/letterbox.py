from typing import Tuple

import cv2
import numpy as np

from py3r.pose.yolo.preprocessing.pipeline import Transform


class Letterbox(Transform):
    """
    Ultralytics-style letterbox (aspect keep + pad).
    imgsz: int or (H,W). If auto=True, pad to minimal stride-aligned rect (may be < imgsz).
    If auto=False, pad to exactly imgsz.
    Dependencies:
      - Uses OpenCV if available; else Pillow; else pure-NumPy bilinear fallback.
    """
    def __init__(self, imgsz=640, stride=32, auto=True, scaleup=True, pad_value=114):
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        self.imgsz = (int(imgsz[0]), int(imgsz[1]))
        self.stride = int(stride)
        self.auto = bool(auto)
        self.scaleup = bool(scaleup)
        self.pad_value = int(pad_value)

    # ---------------- internals ----------------

    def _compute(self, h, w):
        old_h, old_w = self.imgsz
        r = min(old_h / max(h, 1), old_w / max(w, 1))
        if not self.scaleup:
            r = min(r, 1.0)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        dw, dh = old_w - new_w, old_h - new_h
        if self.auto:
            dw %= self.stride
            dh %= self.stride
        pl, pr = int(dw // 2), int(dw - dw // 2)
        pt, pb = int(dh // 2), int(dh - dh // 2)
        return (new_h, new_w), (pt, pb, pl, pr), (r, r)

    # ---------------- public API ----------------

    def forward(self, img: np.ndarray):
        assert img.dtype == np.uint8 and img.ndim in (2, 3), "Expected uint8 (H,W) or (H,W,3)"
        h, w = img.shape[:2]
        new_hw, pads, (r_h, r_w) = self._compute(h, w)

        imr = cv2.resize(img, (new_hw[1], new_hw[0]), interpolation=cv2.INTER_LINEAR)
        pt, pb, pl, pr = pads

        im_out = cv2.copyMakeBorder(
            imr, pt, pb, pl, pr,
            borderType=cv2.BORDER_CONSTANT,
            value=self.pad_value if imr.ndim == 2 else [self.pad_value] * imr.shape[2],
        )

        meta = {
            "ratio_hw": (r_h, r_w),
            "pad": (pt, pb, pl, pr),
            "orig_hw": (h, w),
            "new_hw": im_out.shape[:2],
        }
        return im_out, meta

    def invert_point(self, xy: Tuple[float, float], meta: dict) -> Tuple[float, float]:
        pt, pb, pl, pr = meta["pad"]
        r_h, r_w = meta["ratio_hw"]
        x0 = (xy[0] - pl) / r_w
        y0 = (xy[1] - pt) / r_h
        return x0, y0
