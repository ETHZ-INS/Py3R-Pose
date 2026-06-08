"""
Unit tests for PredictionJob.

Mocks used:
  - MockVideoSource  – VideoSource returning a fixed number of blank frames
  - CollectingPoseWriter – accumulates written VideoFramePoses for assertions
  - MockImageDisplay – IImageDisplay that closes itself after N is_open() calls
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np

from py3r.media.types import VideoFrame
from py3r.pose.cli.prediction_job import PredictionJob
from py3r.pose.core.model.mock_pose_model import MockPoseModel
from py3r.pose.core.types import PoseInstance, PoseInstanceType, PosePoint, VideoFramePoses


# ── Mock helpers ──────────────────────────────────────────────────────────────

class MockVideoSource:
    """VideoSource that yields a fixed number of blank frames and then EOF."""

    def __init__(
        self,
        num_frames: int,
        width: int = 64,
        height: int = 64,
        channels: int = 3,
        fps: float = 30.0,
    ):
        self._num_frames = num_frames
        self._width = width
        self._height = height
        self._channels = channels
        self._fps = fps
        self._frame_index = 0
        self._open = False

    # lifecycle
    def open(self) -> None:
        self._open = True
        self._frame_index = 0

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open

    # capability probes
    def has_timing(self) -> bool: return True
    def has_size(self) -> bool: return True
    def has_fps(self) -> bool: return True
    def has_num_frames(self) -> bool: return True
    def is_seekable(self) -> bool: return False

    # metadata
    def get_size(self) -> tuple[int, int]: return (self._width, self._height)
    def get_fps(self) -> float: return self._fps
    def get_num_channels(self) -> int: return self._channels
    def get_num_frames(self) -> int: return self._num_frames
    def seek(self, frame_index: int) -> None: raise NotImplementedError

    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        if self._frame_index >= self._num_frames:
            return None
        shape = (
            (self._height, self._width)
            if self._channels == 1
            else (self._height, self._width, self._channels)
        )
        frame = VideoFrame(
            img=np.zeros(shape, dtype=np.uint8),
            frame_index=self._frame_index,
            timestamp=self._frame_index / self._fps,
        )
        self._frame_index += 1
        return frame


class CollectingPoseWriter:
    """Pose writer that accumulates VideoFramePoses for inspection after the job."""

    def __init__(self):
        self.frames: List[VideoFramePoses] = []
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def write(self, frame: VideoFramePoses) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class MockImageDisplay:
    """
    IImageDisplay that closes itself after *close_after* calls to is_open().

    is_open() is called once per frame that passes through take_while, so
    close_after effectively limits the number of frames shown.
    display() is scheduled asynchronously on the display_scheduler, so
    len(self.frames) may lag slightly behind is_open_calls; use is_open_calls
    for reliable counting.
    """

    def __init__(self, close_after: int = 999_999):
        self._close_after = close_after
        self.is_open_calls: int = 0
        self.frames: List[np.ndarray] = []
        self.torn_down = False

    def setup(self) -> None:
        pass

    def display(self, img: np.ndarray) -> None:
        self.frames.append(img.copy())

    def is_open(self) -> bool:
        self.is_open_calls += 1
        return self.is_open_calls <= self._close_after

    def teardown(self) -> None:
        self.torn_down = True


class SlowMockPoseModel(MockPoseModel):
    """MockPoseModel with a per-frame sleep to simulate GPU latency.

    This creates real back-pressure in the pipeline: the pose_estimation_scheduler
    falls behind the frame source, filling the bounded queues and exposing any
    scheduler-sharing deadlocks that a fast mock would never trigger.
    """

    def __init__(self, instances, delay_per_frame: float = 0.02):
        super().__init__(instances)
        self.delay_per_frame = delay_per_frame

    def predict_batch(self, batch):
        time.sleep(self.delay_per_frame * len(batch))
        return super().predict_batch(batch)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _instance(name: str = "mouse") -> PoseInstance:
    t = PoseInstanceType(name, ["nose", "tail"])
    return PoseInstance(
        id="0",
        type=t,
        box=(10.0, 10.0, 50.0, 50.0),
        points=[PosePoint(20.0, 20.0, 0.9), PosePoint(40.0, 40.0, 0.8)],
        conf=0.95,
    )


def _job(source: MockVideoSource, model: MockPoseModel) -> PredictionJob:
    job = PredictionJob(model)
    job.set_source(source)
    job.set_no_progress(True)
    return job


# ── Tests: pose writing ───────────────────────────────────────────────────────

class TestPoseWriting:

    def test_all_frames_are_written(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=10), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.run()

        assert len(writer.frames) == 10

    def test_writer_is_opened_and_closed(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=3), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.run()

        assert writer.opened
        assert writer.closed

    def test_frame_indices_are_sequential_from_zero(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=5), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.run()

        assert [f.frame_index for f in writer.frames] == list(range(5))

    def test_frame_size_matches_source_dimensions(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=2, width=128, height=96), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.run()

        assert all(f.size == (128, 96) for f in writer.frames)

    def test_instances_from_model_appear_in_every_frame(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=4), MockPoseModel([_instance("mouse")]))
        job.set_pose_writer(writer)
        job.run()

        assert all(len(f.instances) == 1 for f in writer.frames)
        assert all(f.instances[0].type.name == "mouse" for f in writer.frames)

    def test_multiple_instances_per_frame(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=3), MockPoseModel([_instance("mouse"), _instance("arena")]))
        job.set_pose_writer(writer)
        job.run()

        assert all(len(f.instances) == 2 for f in writer.frames)

    def test_empty_model_writes_empty_instance_lists(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=4), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.run()

        assert all(f.instances == [] for f in writer.frames)


# ── Tests: frame range ────────────────────────────────────────────────────────

class TestFrameRange:

    def test_start_frame_skips_leading_frames(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=10), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.set_start_frame(5)
        job.run()

        assert len(writer.frames) == 5
        assert writer.frames[0].frame_index == 5

    def test_end_frame_limits_frames_processed(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=10), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.set_end_frame(5)
        job.run()

        assert len(writer.frames) == 5

    def test_start_and_end_frame_selects_a_slice(self):
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=10), MockPoseModel([]))
        job.set_pose_writer(writer)
        job.set_start_frame(3)
        job.set_end_frame(7)
        job.run()

        assert len(writer.frames) == 4
        assert writer.frames[0].frame_index == 3
        assert writer.frames[-1].frame_index == 6


# ── Tests: live preview ───────────────────────────────────────────────────────

class TestLivePreview:

    def test_job_terminates_when_display_closes(self):
        """
        Source has 100 frames but the display signals closed after 5 frames.
        run() must return (not hang), exercising the full adaptive_pace +
        sample + display_image + take_while pipeline.
        """
        display = MockImageDisplay(close_after=5)
        job = _job(MockVideoSource(num_frames=100), MockPoseModel([]))
        job.set_live_preview_display(display)
        job.run()

        assert display.torn_down

    def test_display_teardown_called_after_normal_completion(self):
        display = MockImageDisplay()
        job = _job(MockVideoSource(num_frames=5), MockPoseModel([]))
        job.set_live_preview_display(display)
        job.run()

        assert display.torn_down

    def test_display_receives_at_least_one_frame(self):
        """Smoke test that display_image actually delivers frames to the display."""
        display = MockImageDisplay()
        job = _job(MockVideoSource(num_frames=10), MockPoseModel([]))
        job.set_live_preview_display(display)
        job.run()

        assert len(display.frames) > 0

    def test_preview_and_writer_run_concurrently(self):
        """Both outputs can be active simultaneously without deadlock."""
        writer = CollectingPoseWriter()
        display = MockImageDisplay()
        job = _job(MockVideoSource(num_frames=10), MockPoseModel([_instance()]))
        job.set_pose_writer(writer)
        job.set_live_preview_display(display)
        job.run()

        assert len(writer.frames) == 10
        assert len(display.frames) > 0


# ── Tests: no-output mode ─────────────────────────────────────────────────────

class TestNoOutputs:

    def test_runs_without_error_when_no_outputs_configured(self):
        """With no writer and no display the job has no drains and returns
        immediately after connecting — verifying it doesn't hang or raise."""
        job = _job(MockVideoSource(num_frames=5), MockPoseModel([_instance()]))
        job.run()


# ── Tests: deadlock detection ─────────────────────────────────────────────────

class TestDeadlockDetection:
    # 100 frames × 20 ms/frame ≈ 2 s with the correct scheduler.
    # A deadlock never finishes. 10 s gives a 5× safety margin.
    TIMEOUT = 10.0

    def _run_with_timeout(self, job) -> bool:
        """Run job.run() in a daemon thread; return True if it finished in time."""
        thread = threading.Thread(target=job.run, daemon=True)
        thread.start()
        thread.join(timeout=self.TIMEOUT)
        return not thread.is_alive()

    def test_completes_under_back_pressure(self):
        """
        Uses SlowMockPoseModel (20 ms/frame) with 100 frames so the bounded
        queues saturate — the condition that causes a deadlock when two pipeline
        stages share the same EventLoopScheduler.

        This test FAILS (timeout) with the buggy scheduler assignment and
        PASSES with the correct one, making the regression detectable.
        """
        writer = CollectingPoseWriter()
        job = _job(MockVideoSource(num_frames=200), SlowMockPoseModel([], delay_per_frame=0.02))
        job.set_pose_writer(writer)

        finished = self._run_with_timeout(job)

        assert finished, (
            f"job.run() did not complete within {self.TIMEOUT}s — "
            "likely deadlock caused by two pipeline stages sharing the same scheduler"
        )
        assert len(writer.frames) == 200




