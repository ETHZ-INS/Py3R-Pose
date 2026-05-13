import time
from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional

import cv2
from py3r.media.streaming.observables.reader_observable import reader_observable
from py3r.media.streaming.operators import observe_on_bounded, finally_future, adaptive_pace
from py3r.media.streaming.operators.opencv_imshow import opencv_imshow
from py3r.media.streaming.operators.write_to import write_to
from py3r.media.video import VideoSource
from py3r.media.video.ffmpeg_video_file_writer import FFmpegVideoFileWriter
from py3r.pose.core.filtering.abc.pose_filter import IPoseFilter
from py3r.pose.core.model.pose_model import PoseModel
from py3r.pose.core.serialization.dynamic_csv_writer import DynamicPoseCSVWriter
from py3r.pose.core.streaming.filter_poses import filter_poses
from py3r.pose.core.streaming.predict_poses import predict_poses
from py3r.pose.core.streaming.render_poses import render_poses
from py3r.pose.core.tracking.fixed_instances_tracker import FixedInstancesTracker
from py3r.pose.core.types import VideoFramePoses
from py3r.pose.core.visualization.pose_renderer import PoseRenderer

from py3r.pose.yolo.model.staged_yolo_pose_model import StagedYoloPoseModel

import reactivex as rx
from reactivex import operators as ops, Subject
from reactivex.disposable import CompositeDisposable
from reactivex.scheduler import EventLoopScheduler

from py3r.pose.cli.limit_rate import limit_rate
from py3r.pose.cli.operators import ensure_3_channel
from py3r.pose.cli.preview_pace import preview_pace
from py3r.pose.cli.progress_bar_observer import ProgressBarObserver


class PredictionJob:
    def __init__(self, model: PoseModel):
        self.pose_model = model
        self.batch_size = 4

        self.source: Optional[VideoSource] = None

        self.rate_limit: Optional[float] = None

        self.start_frame = 0
        self.end_frame: Optional[int] = None

        self.pose_filter: Optional[IPoseFilter] = None
        self.pose_renderer: Optional[PoseRenderer] = None
        self.tracker: Optional[FixedInstancesTracker] = None

        self.output_file: Optional[Path] = None
        self.live_preview: bool = False
        self.visualization_file: Optional[Path] = None

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

    def set_output_file(self, output_file: Path):
        self.output_file = output_file

    def set_live_preview(self, live_preview: bool):
        self.live_preview = live_preview

    def set_visualization_file(self, visualization_file: Path):
        self.visualization_file = visualization_file

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
        camera_scheduler = EventLoopScheduler()
        pose_estimation_scheduler = EventLoopScheduler()

        schedulers.add(main_scheduler)
        schedulers.add(camera_scheduler)
        schedulers.add(pose_estimation_scheduler)

        stop = Subject()

        frames = reader_observable(self.source, read_timeout_seconds=30.0).pipe(
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

        frame_images = frames.pipe(ops.map(lambda x: x.img))
        #frame_images = frame_images.pipe(ops.map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)))
        if grayscale:
            color_frame_images = frame_images.pipe(ensure_3_channel)
        else:
            color_frame_images = frame_images

        pose_model = StagedYoloPoseModel(self.pose_model, max_batch=self.batch_size, input_channels=1 if grayscale else 3)
        poses = frame_images.pipe(
            observe_on_bounded(pose_estimation_scheduler, maxsize=30),
            predict_poses(pose_model, batch_size=self.batch_size),
            observe_on_bounded(main_scheduler, maxsize=30),
        )

        if self.pose_filter is not None:
            poses = poses.pipe(filter_poses(self.pose_filter))

        if self.tracker is not None:
            poses = poses.pipe(filter_poses(self.tracker))

        poses = poses.pipe(ops.share())

        if self.output_file is not None:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            pose_results_done = Future()
            drains.append(pose_results_done)

            pose_results_scheduler = EventLoopScheduler()
            schedulers.add(pose_results_scheduler)

            pose_results_writer = DynamicPoseCSVWriter(self.output_file)

            pose_results_sub = poses.pipe(
                ops.zip(frames),
                ops.map(lambda p: VideoFramePoses.from_pair(p)),
                observe_on_bounded(pose_results_scheduler, maxsize=30),
                finally_future(pose_results_done),
                write_to(pose_results_writer)
            ).subscribe()
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

        if self.visualization_file is not None:
            assert visualizations is not None, "Visualizations are required to save a video"

            self.visualization_file.parent.mkdir(parents=True, exist_ok=True)

            video_writer_done = Future()
            drains.append(video_writer_done)

            video_writer_scheduler = EventLoopScheduler()
            schedulers.add(video_writer_scheduler)

            frame_size = self.source.get_size()
            assert frame_size is not None, "Frame size is not known"
            fps = self.source.get_fps()
            assert fps is not None, "FPS is not known"

            video_writer = FFmpegVideoFileWriter(self.visualization_file, size=frame_size, fps=fps, quality="medium", grayscale=False)

            video_writer_sub = visualizations.pipe(
                observe_on_bounded(video_writer_scheduler, maxsize=30),
                finally_future(video_writer_done),
                write_to(video_writer)
            ).subscribe()
            subscriptions.add(video_writer_sub)

        if self.live_preview:
            live_preview_input = visualizations if visualizations is not None else color_frame_images

            pacing_scheduler = EventLoopScheduler()
            schedulers.add(pacing_scheduler)

            display_scheduler = EventLoopScheduler()
            schedulers.add(display_scheduler)

            live_preview_done = Future()
            drains.append(live_preview_done)

            def window_still_open(_):
                # treat <= 0 (and -1 on some platforms) as closed
                return cv2.getWindowProperty("Live Preview", cv2.WND_PROP_VISIBLE) > 0.5

            live_preview_sub = live_preview_input.pipe(
                preview_pace(1/30, scheduler=display_scheduler),
                opencv_imshow("Live Preview", scheduler=display_scheduler),
                ops.take_while(window_still_open),
                finally_future(live_preview_done)
            ).subscribe()
            subscriptions.add(live_preview_sub)

        progress_done: Optional[Future] = None
        if not self.no_progress:
            progress_update_interval_seconds = 1.0

            progress_done = Future()
            progress_bar_observer = ProgressBarObserver(num_frames)

            progress_subscription = poses.pipe(
                ops.buffer_with_time(progress_update_interval_seconds),
                finally_future(progress_done)
            ).subscribe(progress_bar_observer)
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
