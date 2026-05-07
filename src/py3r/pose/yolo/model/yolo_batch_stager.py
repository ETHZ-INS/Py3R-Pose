from typing import List, Tuple, Union

import numpy as np
import torch

class YOLOBatchStager:
    """
    Fixed-capacity stager for images

    Inputs (per call): list of N (1 <= N <= B) uint8 images, each either:
      - (H, W) if in_channels == 1 (grayscale)
      - (H, W, 3) if in_channels == 3 (RGB)

    Output:
      - CUDA float32 batch: (N, 3, H_b, W_b) in [0,1]
        where (H_b, W_b) = max height/width across the N images (each <= imgsz)
      - (H_b, W_b)
    """
    def __init__(
        self,
        b: int,
        imgsz: Union[Tuple[int, int], int] = 640,
        in_channels: int = 1,      # 1 for grayscale, 3 for RGB
        device: str = "cuda",
        pad_value: int = 114,
        channels_last: bool = True,
    ):
        assert in_channels in (1, 3), "in_channels must be 1 or 3"
        self.B = int(b)
        self.H, self.W = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        self.in_channels = int(in_channels)
        self.device = device
        self.pad_value = int(pad_value)
        self.channels_last = channels_last

        self.stream = torch.cuda.Stream(device=device)
        self._scale = torch.tensor(1.0 / 255.0, dtype=torch.float32, device=device)

        # Pinned host buffers (only allocate what we need)
        if self.in_channels == 1:
            self.cpu_u8 = torch.empty((self.B, self.H, self.W), dtype=torch.uint8, pin_memory=True)          # NHW (gray)
        else:
            self.cpu_u8 = torch.empty((self.B, self.H, self.W, 3), dtype=torch.uint8, pin_memory=True)       # NHWC (rgb)

        # Device staging (uint8) to prevent per-call allocs
        if self.in_channels == 1:
            self.gpu_u8 = torch.empty((self.B, self.H, self.W), dtype=torch.uint8, device=device)
        else:
            self.gpu_u8 = torch.empty((self.B, self.H, self.W, 3), dtype=torch.uint8, device=device)

        # Final device buffer (float32 RGB)
        memfmt = torch.channels_last if channels_last else torch.contiguous_format
        self.gpu_rgb_f32_full = torch.empty(
            (self.B, 3, self.H, self.W),
            dtype=torch.float32,
            device=device,
            memory_format=memfmt,
        )

    @staticmethod
    def _check_and_get_shape(arr: np.ndarray, in_channels: int):
        assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
        if in_channels == 1:
            assert arr.ndim == 2, f"Expected (H,W) for grayscale, got {arr.shape}"
            return arr.shape[0], arr.shape[1]  # H, W
        else:
            assert arr.ndim == 3 and arr.shape[2] >= 3, f"Expected (H,W,3), got {arr.shape}"
            return arr.shape[0], arr.shape[1]  # H, W

    def upload(self, images: List[np.ndarray]):
        """
        images: list of N augmented uint8 images matching in_channels (1 or 3).
        Returns:
          batch: (N, 3, H_b, W_b) float32 CUDA tensor in [0,1]
          (H_b, W_b)
        """
        n = len(images)
        assert 1 <= n <= self.B, f"N must be in 1..{self.B}, got {n}"

        # 1) Inspect shapes and determine batch canvas
        sizes = [self._check_and_get_shape(im, self.in_channels) for im in images]
        h_b = max(h for h, _ in sizes)
        w_b = max(w for _, w in sizes)
        assert h_b <= self.H and w_b <= self.W, \
            f"Input larger than configured imgsz {(self.H,self.W)}: max input size {(h_b,w_b)}"

        # 2) Fill pinned CPU buffer for the used region (pad right/bottom with pad_value)
        if self.in_channels == 1:
            self.cpu_u8[:n, :h_b, :w_b].fill_(self.pad_value)
            for i, im in enumerate(images):
                h, w = sizes[i]
                t = torch.from_numpy(np.ascontiguousarray(im))  # (h,w) u8
                self.cpu_u8[i, :h, :w].copy_(t, non_blocking=True)
        else:
            self.cpu_u8[:n, :h_b, :w_b].fill_(self.pad_value)
            for i, im in enumerate(images):
                h, w = sizes[i]
                t = torch.from_numpy(np.ascontiguousarray(im[:, :, :3]))  # (h,w,3) u8
                self.cpu_u8[i, :h, :w].copy_(t, non_blocking=True)

        # 3) H→D copy + normalize + layout convert (on a side stream)
        with torch.cuda.stream(self.stream):
            # uint8 copy
            self.gpu_u8[:n, :h_b, :w_b].copy_(self.cpu_u8[:n, :h_b, :w_b].to(self.device, non_blocking=True))

            if self.in_channels == 1:
                # (N,H_b,W_b) -> (N,1,H_b,W_b) -> expand to 3 -> write into final
                g = self.gpu_u8[:n, :h_b, :w_b].to(torch.float32).mul_(self._scale).unsqueeze(1)   # (N,1,H_b,W_b)
                self.gpu_rgb_f32_full[:n, :, :h_b, :w_b].copy_(g.expand(-1, 3, -1, -1))
            else:
                # (N,H_b,W_b,3) -> float -> NHWC->NCHW
                r = self.gpu_u8[:n, :h_b, :w_b].to(torch.float32).mul_(self._scale)                # (N,H_b,W_b,3)
                r = r.permute(0, 3, 1, 2).contiguous()                                             # (N,3,H_b,W_b)
                self.gpu_rgb_f32_full[:n, :, :h_b, :w_b].copy_(r)

            if self.channels_last:
                self.gpu_rgb_f32_full = self.gpu_rgb_f32_full.contiguous(memory_format=torch.channels_last)

        torch.cuda.current_stream().wait_stream(self.stream)

        # 4) Return a view cropped to the effective shape
        batch = self.gpu_rgb_f32_full[:n, :, :h_b, :w_b]
        return batch
