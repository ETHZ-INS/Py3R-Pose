import time
from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional

from py3r.media.streaming.observables.reader_observable import reader_observable
from py3r.media.streaming.operators import observe_on_bounded, finally_future
from py3r.media.streaming.operators.write_to import write_to
from py3r.media.video import VideoSource
from py3r.media.video.ffmpeg_video_file_writer import FFmpegVideoFileWriter
from py3r.pose.core.filtering.abc.pose_filter import IPoseFilter
from py3r.pose.core.serialization.csv_reader import PoseCSVReader
from py3r.pose.core.streaming.filter_poses import filter_poses
from py3r.pose.core.streaming.render_poses import render_poses
from py3r.pose.core.tracking.fixed_instances_tracker import FixedInstancesTracker
from py3r.pose.core.visualization.pose_renderer import PoseRenderer

import reactivex as rx
from reactivex import operators as ops, Subject
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import EventLoopScheduler

from py3r.pose.cli.image_display import IImageDisplay, display_image
from py3r.pose.cli.limit_rate import limit_rate
from py3r.pose.cli.operators import ensure_3_channel
from py3r.pose.cli.preview_pace import preview_pace
from py3r.pose.cli.progress_bar_observer import ProgressBarObserver


class RenderJob:
    def __init__(self):
        self.poses_source: Optional[PoseCSVReader] = None
        self.video_source: Optional[VideoSource] = None

        self.rate_limit: Optional[float] = None

        self.start_frame = 0
        self.end_frame: Optional[int] = None

        self.pose_filter: Optional[IPoseFilter] = None
        self.pose_renderer: Optional[PoseRenderer] = None
        self.tracker: Optional[FixedInstancesTracker] = None

        self.live_preview_display: Optional[IImageDisplay] = None
        self.visualization_file: Optional[Path] = None

        self.quiet: bool = False

    def set_poses_source(self, source: PoseCSVReader):
        self.poses_source = source

    def set_video_source(self, source: VideoSource):
        if not source.has_num_frames():
            raise ValueError("Video source must have a known number of frames")
        self.video_source = source

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

    def set_live_preview_display(self, display: IImageDisplay):
        self.live_preview_display = display

    def set_visualization_file(self, visualization_file: Path):
        self.visualization_file = visualization_file

    def set_quiet(self, no_progress: bool):
        self.quiet = no_progress

    def run(self):
        num_frames = self.video_source.get_num_frames()
        grayscale = self.video_source.get_num_channels() == 1

        subscriptions = CompositeDisposable()
        schedulers = CompositeDisposable()

        drains: List[Future] = []

        main_scheduler = EventLoopScheduler()
        schedulers.add(main_scheduler)

        stop = Subject()

        frames = reader_observable(self.video_source, read_timeout_seconds=30.0).pipe(
            ops.take_while(lambda x: x is not None),
            ops.skip(self.start_frame)
        )

        if self.rate_limit is not None:
            frames = frames.pipe(
                limit_rate(self.rate_limit)
            )

        poses = reader_observable(self.poses_source, read_timeout_seconds=30.0).pipe(
            ops.take_while(lambda x: x is not None),
            #ops.skip(self.start_frame)
        )

        if self.end_frame is not None:
            end_frame = min(self.end_frame, num_frames)
            num_frames = end_frame - self.start_frame
            frames = frames.pipe(ops.take(num_frames))
            poses = poses.pipe(ops.take(num_frames))
        else:
            num_frames = num_frames - self.start_frame

        frames = frames.pipe(
            observe_on_bounded(main_scheduler, maxsize=30),
            ops.take_until(stop)
        )

        poses = poses.pipe(
            observe_on_bounded(main_scheduler, maxsize=30),
            ops.take_until(stop),
            ops.map(lambda x: x.instances)
        )

        if self.pose_filter is not None:
            poses = poses.pipe(filter_poses(self.pose_filter))

        if self.tracker is not None:
            poses = poses.pipe(filter_poses(self.tracker))

        frames = frames.pipe(
            ops.publish()
        )

        poses = poses.pipe(
            ops.publish()
        )

        frame_images = frames.pipe(
            ops.map(lambda x: x.img),
            ops.share()
        )
        frame_images.subscribe()
        color_frame_images = frame_images.pipe(ensure_3_channel) if grayscale else frame_images
        color_frame_images = color_frame_images.pipe(ops.share())

        frames_poses = rx.zip(color_frame_images, poses).pipe(ops.share())
        visualizations = frames_poses.pipe(
            render_poses(self.pose_renderer),
            ops.share()
        )

        if self.visualization_file is not None:
            self.visualization_file.parent.mkdir(parents=True, exist_ok=True)

            video_writer_done = Future()
            drains.append(video_writer_done)

            video_writer_scheduler = EventLoopScheduler()
            schedulers.add(video_writer_scheduler)

            frame_size = self.video_source.get_size()
            assert frame_size is not None, "Frame size is not known"
            fps = self.video_source.get_fps()
            assert fps is not None, "FPS is not known"

            video_writer = FFmpegVideoFileWriter(self.visualization_file, size=frame_size, fps=fps, quality="medium", grayscale=grayscale)
            video_writer_sub = visualizations.pipe(
                observe_on_bounded(video_writer_scheduler, maxsize=30),
                finally_future(video_writer_done),
                write_to(video_writer)
            ).subscribe(on_error=print)
            subscriptions.add(video_writer_sub)

        if self.live_preview_display is not None:
            live_preview_input = visualizations if visualizations is not None else color_frame_images

            display_scheduler = EventLoopScheduler()
            schedulers.add(display_scheduler)

            live_preview_done = Future()
            drains.append(live_preview_done)

            live_preview_sub = live_preview_input.pipe(
                preview_pace(1/30, scheduler=display_scheduler),
                display_image(self.live_preview_display, scheduler=display_scheduler),
                ops.take_while(lambda _: self.live_preview_display.is_open()),
                finally_future(live_preview_done)
            ).subscribe(on_error=lambda e: print(f"Error in live preview: {e}"))
            subscriptions.add(live_preview_sub)

        progress_done: Optional[Future] = None
        if not self.quiet:
            progress_update_interval_seconds = 1.0

            progress_done = Future()
            progress_bar_observer = ProgressBarObserver(num_frames)
            progress_subscription = visualizations.pipe(
                ops.buffer_with_time(progress_update_interval_seconds),
                finally_future(progress_done)
            ).subscribe(progress_bar_observer)
            subscriptions.add(progress_subscription)

        try:
            # connect the published sources
            subscriptions.add(frames.connect())
            subscriptions.add(poses.connect())

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
