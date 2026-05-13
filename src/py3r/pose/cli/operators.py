import reactivex as rx
from reactivex import operators as ops
import numpy as np


def _ensure_1_channel(img):
    if img.ndim == 3:
        return np.ascontiguousarray(img[..., 0])
    else:
        return img


def _ensure_3_channel(img):
    if img.ndim == 2:
        return np.broadcast_to(img[..., None], img.shape + (3,))
    elif img.shape[2] == 1:
        return np.broadcast_to(img, img.shape[:2] + (3,))
    else:
        return img


def ensure_1_channel(source: rx.Observable[np.ndarray]) -> rx.Observable[np.ndarray]:
    return source.pipe(
        ops.map(_ensure_1_channel)
    )


def ensure_3_channel(source: rx.Observable[np.ndarray]) -> rx.Observable[np.ndarray]:
    return source.pipe(
        ops.map(_ensure_3_channel)
    )
