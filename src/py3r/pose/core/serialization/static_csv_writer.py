import csv
from pathlib import Path
from typing import List, Tuple

from py3r.pose.core.types import VideoFramePoses


# TODO: rethink all this

class StaticPoseCSVWriter:
    def __init__(self, file_path: Path | str, columns: List[Tuple[str, str, str, str]]):
        self.file_path = Path(file_path)
        self.columns = columns

        self.file = self.file_path.open("w+", newline="")
        self.csv_writer = csv.writer(self.file)
        self._write_header()

    def _write_header(self):
        header = ["frame_index", "max_dim.x", "max_dim.y"]
        for instance_type_name, instance_id, point_name, value_name in self.columns:
            if point_name is None:
                header.append(f"{instance_type_name}.{instance_id}.{value_name}")
            else:
                header.append(f"{instance_type_name}.{instance_id}.{point_name}.{value_name}")
        self.csv_writer.writerow(header)

    def write_dicts(self, frame_index: int, frame_size: Tuple[int, int], instance_dicts: List[dict]):
        instance_dicts = {instance["id"]: instance for instance in instance_dicts}
        row = [frame_index, frame_size[0], frame_size[1]]

        for column in self.columns:
            instance_type_name, instance_id, point_name, value_name = column

            if instance_id not in instance_dicts:
                row.append(None)
                continue

            instance = instance_dicts[instance_id]

            if point_name is None:
                value = instance["box"][value_name]
            else:
                point = instance["points"][point_name]
                if point is None:
                    value = None
                else:
                    value = instance["points"][point_name][value_name]
            row.append(value)
        self.csv_writer.writerow(row)

    def write(self, pose_results: VideoFramePoses):
        instance_dicts = [instance.as_dict() for instance in pose_results.instances]
        self.write_dicts(pose_results.frame_index, pose_results.size, instance_dicts)

    def write_all(self, data: List[Tuple[int, List[dict]]]):
        for row in data:
            self.write(*row)

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
    def write_csv(file_path: Path | str, columns: List[Tuple[str, str, str, str]], data: List[Tuple[int, List[dict]]]):
        with StaticPoseCSVWriter(file_path, columns) as writer:
            writer.write_all(data)
