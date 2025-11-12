import json
from pathlib import Path
from typing import List

from py3r.pose.core.types import VideoFramePoses
from py3r.pose.core.serialization.json_writer import JSONWriter
from py3r.pose.core.serialization.static_csv_writer import StaticPoseCSVWriter


class DynamicPoseCSVWriter:
    def __init__(self, file_path: Path | str, keep_temp_file: bool = False):
        self.file_path = Path(file_path)
        self.temp_file_path = self.file_path.with_suffix(".jsonl")
        self.keep_temp_file = keep_temp_file

        self.json_writer = JSONWriter(self.temp_file_path)
        self.closed = False

    def write(self, frame: VideoFramePoses):
        self.json_writer.write(frame)

    def write_all(self, data: List[VideoFramePoses]):
        for frame in data:
            self.write(frame)

    def _finalize(self):
        with self.temp_file_path.open("r") as json_file:
            frame_dicts = [json.loads(json_line) for json_line in json_file]

        unique_instances = {}

        for frame_dict in frame_dicts:
            if frame_dict is None:
                continue
            for instance_dict in frame_dict["instances"]:
                if instance_dict["id"] not in unique_instances:
                    unique_instances[instance_dict["id"]] = instance_dict

        unique_instances = list(unique_instances.values())
        unique_instances.sort(key=lambda x: x["id"])

        columns = []
        for instance in unique_instances:
            instance_type = instance["type"]
            instance_id = instance["id"]
            for value_name in ["x1", "y1", "x2", "y2", "conf"]:
                columns.append((instance_type, instance_id, None, value_name))
            for point_name in instance["points"]:
                for value_name in ["x", "y", "conf"]:
                    columns.append((instance_type, instance_id, point_name, value_name))

        with StaticPoseCSVWriter(self.file_path, columns) as csv_writer:
            for frame_dict in frame_dicts:
                frame_index = frame_dict["frame_index"]
                frame_size = frame_dict["frame_size"]
                instance_dicts = frame_dict["instances"]
                csv_writer.write_dicts(frame_index, frame_size, instance_dicts)

        if not self.keep_temp_file:
            self.temp_file_path.unlink()

    def close(self):
        if not self.closed:
            self.json_writer.close()
            self._finalize()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def write_csv(file_path: Path | str, data: List[VideoFramePoses]):
        with DynamicPoseCSVWriter(file_path) as writer:
            writer.write_all(data)
