import csv
from pathlib import Path
from typing import Tuple, List

from py3r.point_tracking.core.types import VideoFramePoseResults
from py3r.point_tracking.core.types.instance_type import PoseInstanceType
from py3r.point_tracking.core.types.instance import PoseInstance
from py3r.point_tracking.core.types.point import PosePoint


class CSVReader:
    def __init__(self, file_path: Path | str, instance_types: List[PoseInstanceType] = None):
        self.file_path = Path(file_path)

        # If you don't explicitly pass instance types, they will be inferred from the CSV file
        # This has the caveat that the instance types will be missing skeleton information etc.
        # and the point names may be in a different order than the original instance types
        if instance_types is not None:
            self.strict_types = True
            self.instance_types = {instance_type.name: instance_type for instance_type in instance_types}
        else:
            self.strict_types = False
            self.instance_types = {}

        self.file = self.file_path.open("r")
        self.csv_reader = csv.reader(self.file)

        self.columns = self._parse_header()

    def get_instance_types(self) -> List[PoseInstanceType]:
        return list(self.instance_types.values())

    def get_instance_ids(self) -> List[str]:
        return list(set(column[1] for column in self.columns))

    def get_columns(self) -> List[Tuple[str, str, str, str, str]]:
        return self.columns

    def _parse_header(self):
        try:
            header_row = next(self.csv_reader)
        except StopIteration:
            raise ValueError("CSV file is empty")

        # Create a map of which column corresponds to which instance, point and value
        columns = []
        for column_name in header_row:
            column_name_split = column_name.split(".")  # Split the column name by the dot separator

            if column_name_split[0] == "frame_index":
                columns.append(("frame", None, None, None, "index"))
                continue

            if column_name_split[0] == "min_dim":
                columns.append(("frame", None, None, "min_dim", column_name_split[1]))
                continue

            if column_name_split[0] == "max_dim":
                columns.append(("frame", None, None, "max_dim", column_name_split[1]))
                continue

            if len(column_name_split) == 3:
                point_name = None
                instance_type_name, instance_id, value_name = column_name_split
            elif len(column_name_split) == 4:
                instance_type_name, instance_id, point_name, value_name = column_name_split
            else:
                raise ValueError(f"Invalid column name: {column_name}")

            if not self.strict_types:
                if instance_type_name not in self.instance_types:
                    self.instance_types[instance_type_name] = PoseInstanceType(instance_type_name, [], [])

                if point_name is not None and point_name not in self.instance_types[instance_type_name].point_names:
                    self.instance_types[instance_type_name].point_names.append(point_name)

            columns.append(("instance", instance_type_name, instance_id, point_name, value_name))
        return columns

    def _parse_instance(self, instance_dict: dict) -> PoseInstance:
        instance_type_name = instance_dict["type"]
        instance_type = self.instance_types[instance_type_name]

        box = instance_dict["box"]
        box_conf = float(box["conf"]) if "conf" in box else 1.0
        if "x1" in box and "x2" in box and "y1" in box and "y2" in box:
            box = (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
        else:
            box = (0.0, 0.0, 0.0, 0.0)

        points = []
        for point_name in instance_type.point_names:
            point_data = instance_dict["points"].get(point_name)
            if point_data is None:
                points.append(None)
                continue

            x = float(point_data["x"])
            y = float(point_data["y"])
            conf = float(point_data["conf"])
            points.append(PosePoint(x, y, conf))

        return PoseInstance(
            id=instance_dict["id"],
            type=instance_type,
            box=box,
            points=points,
            conf=box_conf
        )

    def read(self) -> VideoFramePoseResults | None:
        instance_dicts = {}

        try:
            row = next(self.csv_reader)
        except StopIteration:
            return None

        frame_index = None
        frame_width = None
        frame_height = None

        for column_index, value in enumerate(row):
            column_type, instance_type_name, instance_id, point_name, value_name = self.columns[column_index]

            if value == "":
                value = None

            if column_type == "frame":
                if point_name is None and value_name == "index":
                    if value is None:
                        # If the frame index column is empty, we assume the file is done
                        return None

                    try:
                        frame_index = int(value)
                    except ValueError:
                        raise ValueError(f"Invalid frame index in first column: {row[0]}")
                elif point_name == "max_dim" and value_name == "x":
                    try:
                        frame_width = int(float(value))
                    except ValueError:
                        raise ValueError(f"Invalid frame width: {value}")
                elif point_name == "max_dim" and value_name == "y":
                    try:
                        frame_height = int(float(value))
                    except ValueError:
                        raise ValueError(f"Invalid frame height: {value}")
                else:
                    raise ValueError(f"Invalid frame value name: {value_name}")

            elif column_type == "instance":

                if instance_id not in instance_dicts:
                    instance_dicts[instance_id] = {"id": instance_id, "type": instance_type_name, "box": {}, "points": {}}

                instance_dict = instance_dicts[instance_id]

                if point_name is None:
                    if value_name not in ["x1", "y1", "x2", "y2", "conf"]:
                        raise ValueError(f"Invalid bounding box value name: {value_name}")
                    instance_dict["box"][value_name] = value
                else:
                    if value_name not in ["x", "y", "conf"]:
                        raise ValueError(f"Invalid point value name: {value_name}")
                    if point_name not in instance_dict["points"]:
                        instance_dict["points"][point_name] = {}
                    instance_dict["points"][point_name][value_name] = value

        instance_dicts = list(sorted(instance_dicts.values(), key=lambda x: x["id"]))
        instance_dicts = [instance_dict for instance_dict in instance_dicts if not any(value is None for value in instance_dict["box"].values())]

        for instance_dict in instance_dicts:
            instance_dict["points"] = {point_name: point_data for point_name, point_data in instance_dict["points"].items() if not any(value is None for value in point_data.values())}
            for point_name in self.instance_types[instance_dict["type"]].point_names:
                if point_name not in instance_dict["points"]:
                    instance_dict["points"][point_name] = None

        instances = [self._parse_instance(instance_dict) for instance_dict in instance_dicts]

        return VideoFramePoseResults(instances, (frame_width, frame_height), frame_index, 0.0)

    def read_all(self) -> List[VideoFramePoseResults]:
        frames = []
        while True:
            frame = self.read()
            if frame is None:
                break
            frames.append(frame)
        return frames

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
    def read_csv(file_path: Path | str) -> List[VideoFramePoseResults]:
        with CSVReader(file_path) as reader:
            return reader.read_all()
