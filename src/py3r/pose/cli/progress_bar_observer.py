import queue
import time
from typing import Any, List, Optional

import reactivex as rx


def print_progress_bar(num_frames_processed, frame_count=None, fps=None, finished=False):
    if frame_count is not None:
        progress = int(60 * num_frames_processed / frame_count) if frame_count is not None else 0
        progress_bar = f"[{u'█' * progress}{('.' * (60 - progress))}] "

        absolute_progress = f"{num_frames_processed}/{frame_count} "
    else:
        progress_bar = ""
        absolute_progress = f"{num_frames_processed} "

    if fps is not None:
        fps_text = f"{fps:.2f} fps "
    else:
        fps_text = ""

    end = "\r" if not finished else "\n"
    print(f"{progress_bar}{absolute_progress}{fps_text}", end=end, flush=True)


class ProgressBarObserver(rx.Observer[Any]):
    def __init__(self, num_total: int):
        super().__init__()

        self.num_total = num_total
        self.num_processed = 0

        self.fps_window_size = 10
        self.fps_samples = queue.Queue(self.fps_window_size)

        self.first_time = None

    def _calculate_fps(self) -> Optional[float]:
        if self.fps_samples.qsize() >= 2:
            first_time, first_count = self.fps_samples.queue[0]
            last_time, last_count = self.fps_samples.queue[-1]

            elapsed = last_time - first_time
            frames_processed = last_count - first_count

            fps = frames_processed / elapsed if elapsed > 0 else 0.0
            return fps
        else:
            return None

    def _calculate_overall_fps(self) -> Optional[float]:
        if self.first_time is not None:
            last_time, last_count = self.fps_samples.queue[-1]
            elapsed = last_time - self.first_time
            fps = last_count / elapsed if elapsed > 0 else 0.0
            return fps
        else:
            return None

    def _on_next_core(self, items: List[Any]) -> None:
        if len(items) == 0:
            return

        now = time.monotonic()

        num_items = len(items)
        self.num_processed += num_items

        if self.fps_samples.full():
            self.fps_samples.get()

        self.fps_samples.put((now, self.num_processed))

        if self.first_time is None:
            self.first_time = now

        fps = self._calculate_fps()

        print_progress_bar(
            self.num_processed,
            frame_count=self.num_total,
            fps=fps,
            finished=False
        )

    def _on_completed_core(self) -> None:
        overall_fps = self._calculate_overall_fps()

        print_progress_bar(
            self.num_processed,
            frame_count=self.num_total,
            fps=overall_fps,
            finished=True
        )
