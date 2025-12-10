import json
from pathlib import Path
from typing import List, Union

from py3r.pose.core.types import VideoFramePoses


class PoseJSONWriter:
    def __init__(self, file_path: Union[Path, str]):
        self.file_path = Path(file_path)
        self.file = self.file_path.open("w+")

    def write(self, pose_results: VideoFramePoses):
        instance_dicts = [instance.as_dict() for instance in pose_results.instances]

        frame_data = {"frame_index": pose_results.frame_index, "frame_size": pose_results.size, "instances": instance_dicts}
        frame_json = json.dumps(frame_data)
        self.file.write(frame_json + "\n")

    def write_all(self, data: List[VideoFramePoses]):
        for frame in data:
            self.write(frame)

    def close(self):
        if not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def write_json(file_path: Union[Path, str], data: List[VideoFramePoses]):
        with PoseJSONWriter(file_path) as writer:
            writer.write_all(data)
