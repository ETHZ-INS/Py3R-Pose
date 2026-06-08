import time
from concurrent.futures import Future
from typing import List, Optional

from py3r.media.streaming.observables.reader_observable import reader_observable
from py3r.media.streaming.operators import observe_on_bounded, finally_future, adaptive_pace
from py3r.media.streaming.operators.write_to import write_to
from py3r.media.video import VideoSource
from py3r.pose.core.filtering.abc.pose_filter import IPoseFilter
from py3r.pose.core.model.abc.pose_model import IPoseModel
from py3r.pose.core.streaming.filter_poses import filter_poses
from py3r.pose.core.streaming.predict_poses import predict_poses
from py3r.pose.core.streaming.render_poses import render_poses
from py3r.pose.core.tracking.fixed_instances_tracker import FixedInstancesTracker
from py3r.pose.core.types import VideoFramePoses
from py3r.pose.core.visualization.pose_renderer import PoseRenderer


import reactivex as rx
from reactivex import operators as ops, Subject
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import EventLoopScheduler

from py3r.pose.cli.image_display import IImageDisplay, display_image
from py3r.pose.cli.limit_rate import limit_rate
from py3r.pose.cli.operators import ensure_3_channel
from py3r.pose.cli.progress_bar_observer import ProgressBarObserver


class PredictionJob:
    def __init__(self, model: IPoseModel):
        self.pose_model = model
        self.batch_size = 4

        self.source: Optional[VideoSource] = None

        self.rate_limit: Optional[float] = None

        self.start_frame = 0
        self.end_frame: Optional[int] = None

        self.pose_filter: Optional[IPoseFilter] = None
        self.pose_renderer: Optional[PoseRenderer] = None
        self.tracker: Optional[FixedInstancesTracker] = None

        self.pose_writer = None
        self.video_writer = None

        self.live_preview_display: Optional[IImageDisplay] = None
        self.visualization_length_frames: Optional[int] = None

        self.no_progress: bool = False

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def set_source(self, source: VideoSource):
        if not source.has_num_frames():
            raise ValueError("Video source must have a known number of frames")
        self.source = source

    def set_rate_limit(self, rate_limit: Optional[float]):
        self.rate_limit = rate_limit

    def set_start_frame(self, start_frame: int):
        self.start_frame = start_frame

    def set_end_frame(self, end_frame: Optional[int]):
        self.end_frame = end_frame

    def set_pose_filter(self, pose_filter: IPoseFilter):
        self.pose_filter = pose_filter

    def set_tracker(self, tracker: FixedInstancesTracker):
        self.tracker = tracker

    def set_pose_renderer(self, pose_renderer: PoseRenderer):
        self.pose_renderer = pose_renderer

    def set_pose_writer(self, writer):
        self.pose_writer = writer

    def set_live_preview_display(self, display: IImageDisplay):
        self.live_preview_display = display

    def set_video_writer(self, writer):
        self.video_writer = writer

    def set_visualization_length_frames(self, visualization_length_frames: Optional[int]):
        self.visualization_length_frames = visualization_length_frames

    def set_no_progress(self, no_progress: bool):
        self.no_progress = no_progress

    def run(self):
        num_frames = self.source.get_num_frames()
        grayscale = self.source.get_num_channels() == 1

        subscriptions = CompositeDisposable()
        schedulers = CompositeDisposable()

        drains: List[Future] = []

        main_scheduler = EventLoopScheduler()
        pose_estimation_scheduler = EventLoopScheduler()
        pose_results_scheduler = EventLoopScheduler()

        schedulers.add(main_scheduler)
        schedulers.add(pose_estimation_scheduler)
        schedulers.add(pose_results_scheduler)

        stop = Subject()

        frames = reader_observable(self.source, read_timeout_seconds=10.0).pipe(
            ops.take_while(lambda x: x is not None),
            ops.skip(self.start_frame)
        )

        if self.rate_limit is not None:
            frames = frames.pipe(
                limit_rate(self.rate_limit)
            )

        if self.end_frame is not None:
            end_frame = min(self.end_frame, num_frames)
            num_frames = end_frame - self.start_frame
            frames = frames.pipe(ops.take(num_frames))
        else:
            num_frames = num_frames - self.start_frame

        frames = frames.pipe(
            observe_on_bounded(main_scheduler, maxsize=30),
            ops.take_until(stop),
            ops.publish()
        )

        frame_images = frames.pipe(ops.map(lambda x: x.img), ops.share())
        if grayscale:
            color_frame_images = frame_images.pipe(ensure_3_channel, ops.share())
        else:
            color_frame_images = frame_images

        poses = frame_images.pipe(
            observe_on_bounded(pose_estimation_scheduler, maxsize=30),
            predict_poses(self.pose_model, batch_size=self.batch_size),
            observe_on_bounded(pose_results_scheduler, maxsize=30),
        )

        if self.pose_filter is not None:
            poses = poses.pipe(filter_poses(self.pose_filter))

        if self.tracker is not None:
            poses = poses.pipe(filter_poses(self.tracker))

        poses = poses.pipe(ops.share())

        if self.pose_writer is not None:
            pose_results_done = Future()
            drains.append(pose_results_done)

            pose_results_writer_scheduler = EventLoopScheduler()
            schedulers.add(pose_results_writer_scheduler)

            pose_results_sub = poses.pipe(
                ops.zip(frames),
                ops.map(lambda p: VideoFramePoses.from_pair(p)),
                observe_on_bounded(pose_results_writer_scheduler, maxsize=30),
                finally_future(pose_results_done),
                write_to(self.pose_writer)
            ).subscribe(on_error=lambda e: print(f"Error writing pose results: {e}"))
            subscriptions.add(pose_results_sub)

        frames_poses = rx.zip(color_frame_images, poses)

        visualizations = None
        if self.pose_renderer is not None:
            visualizations = frames_poses.pipe(
                render_poses(self.pose_renderer)
            )

            if self.visualization_length_frames is not None:
                visualizations = visualizations.pipe(
                    ops.take(self.visualization_length_frames)
                )

            visualizations = visualizations.pipe(ops.share())

        if self.video_writer is not None:
            assert visualizations is not None, "Visualizations are required to save a video"

            video_writer_done = Future()
            drains.append(video_writer_done)

            video_writer_scheduler = EventLoopScheduler()
            schedulers.add(video_writer_scheduler)

            video_writer_sub = visualizations.pipe(
                observe_on_bounded(video_writer_scheduler, maxsize=30),
                finally_future(video_writer_done),
                write_to(self.video_writer)
            ).subscribe(on_error=lambda e: print(f"Error writing visualization video: {e}"))
            subscriptions.add(video_writer_sub)

        if self.live_preview_display is not None:
            live_preview_input = visualizations if visualizations is not None else color_frame_images

            display_scheduler = EventLoopScheduler()
            schedulers.add(display_scheduler)

            live_preview_done = Future()
            drains.append(live_preview_done)

            live_preview_sub = live_preview_input.pipe(
                adaptive_pace(window_size=self.batch_size * 4, initial_interval=1/30),
                ops.sample(1/30, scheduler=display_scheduler),
                display_image(self.live_preview_display, scheduler=display_scheduler),
                # TODO: I guess since display_image opens the display async, there is a small chance that the display is not open yet
                #   when the first frame arrives at ops.take_while, which would cause this path to end immediately.
                #   Don't know how to prevent that
                ops.take_while(lambda _: self.live_preview_display.is_open()),
                finally_future(live_preview_done)
            ).subscribe(on_error=lambda e: print(f"Error in live preview: {e}"))
            subscriptions.add(live_preview_sub)

        progress_done: Optional[Future] = None
        if not self.no_progress:
            progress_update_interval_seconds = 1.0

            progress_done = Future()
            progress_bar_observer = ProgressBarObserver(num_frames)

            progress_subscription = poses.pipe(
                ops.buffer_with_time(progress_update_interval_seconds),
                finally_future(progress_done)
            ).subscribe(progress_bar_observer, on_error=lambda e: print(f"Error in progress bar: {e}"))
            subscriptions.add(progress_subscription)

        try:
            # connect the published source
            frames_sub = frames.connect()
            subscriptions.add(frames_sub)

            for drain in drains:
                while not drain.done():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            stop.on_next(None)
            for drain in drains:
                while not drain.done():
                    time.sleep(0.1)
            raise
        finally:
            subscriptions.dispose()

            if progress_done:
                while not progress_done.done():
                    time.sleep(0.1)

            schedulers.dispose()
