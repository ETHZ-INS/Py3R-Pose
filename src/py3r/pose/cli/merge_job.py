import time
from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional, Tuple

from py3r.media.streaming.observables.reader_observable import reader_observable
from py3r.media.streaming.operators import finally_future
from py3r.media.streaming.operators.write_to import write_to
from py3r.pose.core.serialization.csv_reader import PoseCSVReader
from py3r.pose.core.serialization.dynamic_csv_writer import DynamicPoseCSVWriter

from py3r.pose.core.types import VideoFramePoses
from reactivex import operators as ops

from py3r.pose.cli.zip_by_index import zip_by_index


class MergeJob:
    def __init__(self):
        self.pose_sources: List[PoseCSVReader] = []
        self.output_file: Optional[Path] = None

    def set_pose_sources(self, pose_sources: List[PoseCSVReader]):
        self.pose_sources = pose_sources
        
    def set_output_file(self, output_file: Path):
        self.output_file = output_file

    def run(self):
        assert len(self.pose_sources) > 0, "Need at least one pose source"
        assert self.output_file is not None, "Output file is not set"

        source_observables = [
            reader_observable(pose_source).pipe(ops.take_while(lambda x: x is not None))
            for pose_source in self.pose_sources
        ]
        poses = zip_by_index(*source_observables, fill_value=None, index=lambda x: x.frame_index)

        def _merge(frames: Tuple[VideoFramePoses]):
            frame_size = next(frame.size for frame in frames if frame is not None)
            frame_index = next(frame.frame_index for frame in frames if frame is not None)
            instances = sum([frame.instances for frame in frames if frame is not None], [])
            return VideoFramePoses(instances, frame_size, frame_index)

        poses = poses.pipe(ops.map(_merge))

        pose_results_done = Future()

        pose_results_writer = DynamicPoseCSVWriter(self.output_file)
        pose_results_sub = poses.pipe(
            finally_future(pose_results_done),
            write_to(pose_results_writer)
        ).subscribe(on_error=print)

        try:
            while not pose_results_done.done():
                time.sleep(0.1)
        finally:
            pose_results_sub.dispose()

#        pose_result_lists = []
#        for pose_source in self.pose_sources:
#            with pose_source:
#                pose_result_lists.append(pose_source.read_all())
#
#        all_poses = [pose for pose_list in pose_result_lists for pose in pose_list]
#
#        with DynamicPoseCSVWriter(self.output_file) as writer:
#            for pose in all_poses:
#                writer.write(pose)
