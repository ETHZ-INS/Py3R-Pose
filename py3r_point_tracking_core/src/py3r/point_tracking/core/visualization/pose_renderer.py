from typing import Tuple, List, Dict

import cv2
import numpy as np
import colorsys

from py3r.point_tracking.core.data.instance import Instance
from py3r.point_tracking.core.data.instance_type import InstanceType
from py3r.point_tracking.core.data.point import Point

Color = Tuple[int, int, int]


def color_from_hue(hue: float) -> Color:
    color = colorsys.hsv_to_rgb(hue, 1, 1)
    return int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)


class PoseRenderer:
    def __init__(self, instance_types: List[InstanceType] = None, color_map: Dict[Tuple[str, str], Color] = None):
        self.color_map = {
            (instance_type.name, point_name): color_from_hue(point_index / len(instance_type.point_names))
            for instance_type in instance_types
            for point_index, point_name in enumerate(instance_type.point_names)
        } if color_map is None else color_map

        self.point_radius = 5

        self.show_skeleton = True
        self.skeleton_line_width = 2
        self.skeleton_color = (0, 0, 0)

        self.show_bounding_box = False
        self.bounding_box_width = 2
        self.bounding_box_color = (0, 0, 255)

        self.show_instance_id = False
        self.show_class_name = False
        self.show_confidence = False
        self.class_name_padding = 5
        self.class_name_font = cv2.FONT_HERSHEY_SIMPLEX
        self.class_name_scale = 1
        self.class_name_thickness = 2
        self.class_name_color = (255, 255, 255)

    def set_instance_types(self, instance_types: List[InstanceType]):
        self.color_map = {
            (instance_type.name, point_name): color_from_hue(point_index / len(instance_type.point_names))
            for instance_type in instance_types
            for point_index, point_name in enumerate(instance_type.point_names)
        }

    def set_color_map(self, color_map: Dict[Tuple[str, str], Color]):
        self.color_map = color_map

    def set_point_radius(self, radius: int):
        self.point_radius = radius

    def set_point_color(self, instance_type_name: str, point_name: str, color: Color):
        self.color_map[(instance_type_name, point_name)] = color

    def set_show_skeleton(self, show: bool):
        self.show_skeleton = show

    def set_skeleton_line_width(self, width: int):
        self.skeleton_line_width = width

    def set_skeleton_color(self, color: Color):
        self.skeleton_color = color

    def set_show_bounding_box(self, show: bool):
        self.show_bounding_box = show

    def set_bounding_box_width(self, width: int):
        self.bounding_box_width = width

    def set_bounding_box_color(self, color: Color):
        self.bounding_box_color = color

    def set_show_instance_id(self, show: bool):
        self.show_instance_id = show

    def set_show_class_name(self, show: bool):
        self.show_class_name = show

    def set_show_confidence(self, show: bool):
        self.show_confidence = show

    def set_class_name_scale(self, scale: float):
        self.class_name_scale = scale

    def set_class_name_thickness(self, thickness: int):
        self.class_name_thickness = thickness

    def set_class_name_color(self, color: Color):
        self.class_name_color = color

    def draw_point(self, frame: np.ndarray, point: Point, color: Color) -> np.ndarray:
        p = (int(point.x), int(point.y))
        return cv2.circle(frame, p, self.point_radius, color, -1)

    def draw_skeleton_line(self, frame: np.ndarray, point1: Point, point2: Point) -> np.ndarray:
        p1 = (int(point1.x), int(point1.y))
        p2 = (int(point2.x), int(point2.y))
        return cv2.line(frame, p1, p2, self.skeleton_color, self.skeleton_line_width)

    def draw_bounding_box(self, frame: np.ndarray, instance: Instance) -> np.ndarray:
        box = instance.box
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        return cv2.rectangle(frame, p1, p2, self.bounding_box_color, self.bounding_box_width)

    def draw_class_name(self, frame: np.ndarray, instance: Instance) -> np.ndarray:
        box = instance.box
        if box is None:
            if not any([point is not None for point in instance.points]):
                return frame
            min_x = min([point.x for point in instance.points if point is not None])
            max_x = max([point.x for point in instance.points if point is not None])
            min_y = min([point.y for point in instance.points if point is not None])
            max_y = max([point.y for point in instance.points if point is not None])
            box = (min_x, min_y, max_x, max_y)
        p = (int(box[0]), int(box[1]))

        if self.show_instance_id and self.show_class_name:
            instance_text = instance.id + ": " + instance.type.name
        elif self.show_instance_id:
            instance_text = instance.id
        elif self.show_class_name:
            instance_text = instance.type.name
        else:
            instance_text = ""

        if self.show_confidence:
            instance_text += f" ({instance.conf:.2f})" if instance_text else f"{instance.conf:.2f}"

        if not instance_text:
            return frame

        # calculate the size of the text
        text_size = cv2.getTextSize(instance_text, self.class_name_font, self.class_name_scale, self.class_name_thickness)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        text_baseline = text_size[1]

        # calculate the size of the text background
        bg_width = text_width + self.class_name_padding*2
        bg_height = text_height + text_baseline + self.class_name_padding*2

        bg_p1 = (int(p[0] - self.bounding_box_width/2), int(p[1] - bg_height - self.bounding_box_width/2))
        bg_p2 = (bg_p1[0] + bg_width, bg_p1[1] + bg_height)

        if self.show_bounding_box:
            frame = cv2.rectangle(frame, bg_p1, bg_p2, self.bounding_box_color, -1)

        # move the text up by the height of the text baseline so letters like p and q are above the bounding box
        # and move the text to the left by half the width of the bounding box line so it aligns with the left side of the box
        text_p = (bg_p1[0] + self.class_name_padding, bg_p1[1] + text_height + self.class_name_padding)

        return cv2.putText(frame, instance_text, text_p, self.class_name_font, self.class_name_scale, self.class_name_color, self.class_name_thickness)

    def draw_frame(self, frame: np.ndarray, instances: List[Instance]) -> np.ndarray:
        for instance in instances:
            if self.show_skeleton:
                for line in instance.type.skeleton:
                    point1 = instance.points[line[0]]
                    point2 = instance.points[line[1]]
                    if point1 is None or point2 is None:
                        continue
                    frame = self.draw_skeleton_line(frame, point1, point2)

            if self.show_bounding_box:
                frame = self.draw_bounding_box(frame, instance)

            frame = self.draw_class_name(frame, instance)

            for point_name, point in zip(instance.type.point_names, instance.points):
                if point is None:
                    continue
                if (instance.type.name, point_name) not in self.color_map:
                    continue
                color = self.color_map[(instance.type.name, point_name)]
                frame = self.draw_point(frame, point, color)
        return frame
