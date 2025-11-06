import json
from pathlib import Path
from typing import List

from py3r.pose.core.types import VideoFramePoses


class JSONWriter:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self.file = self.file_path.open("w+")
        self.file.write("[\n")

    def write(self, pose_results: VideoFramePoses):
        instance_dicts = [instance.as_dict() for instance in pose_results.instances]

        frame_data = {"frame_index": pose_results.frame_index, "frame_size": pose_results.size, "instances": instance_dicts}
        frame_json = json.dumps(frame_data)
        self.file.write(frame_json + ",\n")

    def write_all(self, data: List[VideoFramePoses]):
        for frame in data:
            self.write(frame)

    def close(self):
        if not self.file.closed:
            self.file.write("null\n]")
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def write_json(file_path: Path | str, data: List[VideoFramePoses]):
        frame_dicts = [
            {"frame_index": frame_index, "instances": [instance.as_dict() for instance in instances]}
            for frame_index, instances in data
        ]
        with Path(file_path).open("w+") as file:
            json.dump(frame_dicts, file)
