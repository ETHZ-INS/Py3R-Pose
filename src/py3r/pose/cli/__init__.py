import argparse
from pathlib import Path
from typing import List

from py3r.pose.core.serialization.csv_reader import PoseCSVReader

from py3r.pose.cli.concat_job import ConcatJob
from py3r.pose.cli.intermittent_pose_model import IntermittentPoseModel
from py3r.pose.cli.merge_job import MergeJob
from py3r.pose.cli.model_repository import ModelDetector
from py3r.pose.cli.render_job import RenderJob

available_models = ModelDetector().find_models().filter_by_task("pose_estimation").filter(lambda e: e.type in ["yolo_pose_3rhub", "pose_estimation_model_group"])


def build_pose_renderer(args, instance_types):
    from py3r.pose.core.visualization.pose_renderer import PoseRenderer
    pose_renderer = PoseRenderer(instance_types)

    pose_renderer.set_show_skeleton(not args.no_skeleton)
    pose_renderer.set_show_bounding_box(args.show_bounding_box)
    pose_renderer.set_show_instance_id(args.show_instance_id)
    pose_renderer.set_show_class_name(args.show_class_name)
    pose_renderer.set_show_confidence(args.show_confidence)
    pose_renderer.set_point_radius(args.point_radius)
    pose_renderer.set_skeleton_line_width(args.skeleton_thickness)

    bounding_box_color = (int(args.bounding_box_color[1:3], 16), int(args.bounding_box_color[3:5], 16), int(args.bounding_box_color[5:7], 16))
    pose_renderer.set_bounding_box_color(bounding_box_color)

    skeleton_color = (int(args.skeleton_color[1:3], 16), int(args.skeleton_color[3:5], 16), int(args.skeleton_color[5:7], 16))
    pose_renderer.set_skeleton_color(skeleton_color)

    class_name_color = (int(args.class_name_color[1:3], 16), int(args.class_name_color[3:5], 16), int(args.class_name_color[5:7], 16))
    pose_renderer.set_class_name_color(class_name_color)

    return pose_renderer

def build_model(model_identifier: str):
    from py3r.pose.cli.model_repository import ModelDetector
    from py3r.pose.yolo.model.yolo_pose_model import YoloPoseModel
    from py3r.pose.core.model.composite_pose_model import CompositePoseModel

    if ";" in model_identifier:
        model_identifier, intermittent_interval = model_identifier.rsplit(";", 1)
        try:
            intermittent_interval = int(intermittent_interval)
        except ValueError:
            raise ValueError(f"Invalid intermittent interval: {intermittent_interval}")
    else:
        intermittent_interval = None

    if Path(model_identifier).is_dir():
        model = YoloPoseModel.from_folder(Path(model_identifier), iou_threshold=0.5)
    else:
        if ":" in model_identifier:
            name, version = model_identifier.split(":", 1)
            model_entry = available_models.get(name, version)
        else:
            model_entry = available_models.get(model_identifier)

        if model_entry is None:
            raise ValueError(f"Model not found: {model_identifier}")

        if model_entry.type == "yolo_pose_3rhub":
            model = YoloPoseModel.from_folder(model_entry.folder, iou_threshold=0.5)
        elif model_entry.type == "pose_estimation_model_group":
            sub_model_identifiers = model_entry.manifest["models"]
            sub_models = []
            for sub_model_identifier in sub_model_identifiers:
                sub_model = build_model(sub_model_identifier)
                sub_models.append(sub_model)
            model = CompositePoseModel(sub_models)
        else:
            raise ValueError(f"Model type not supported: {model_entry.type}")

    if intermittent_interval is not None:
        model = IntermittentPoseModel(model, intermittent_interval)

    return model

def get_instance_types_from_model(model_identifier: str):
    from py3r.pose.cli.model_repository import ModelDetector
    from py3r.pose.yolo.model.yolo_pose_model import YoloPoseModel

    if ";" in model_identifier:
        model_identifier, _ = model_identifier.rsplit(";", 1)

    if Path(model_identifier).is_dir():
        instance_types = YoloPoseModel.load_instance_types(Path(model_identifier))
    else:
        if ":" in model_identifier:
            name, version = model_identifier.split(":", 1)
            model_entry = available_models.get(name, version)
        else:
            model_entry = available_models.get(model_identifier)

        if model_entry is None:
            raise ValueError(f"Model not found: {model_identifier}")

        if model_entry.type == "yolo_pose_3rhub":
            instance_types = YoloPoseModel.load_instance_types(model_entry.folder)
        elif model_entry.type == "pose_estimation_model_group":
            sub_model_identifiers = model_entry.manifest["models"]
            instance_types = []
            for sub_model_identifier in sub_model_identifiers:
                sub_model_instance_types = get_instance_types_from_model(sub_model_identifier)
                instance_types.extend(sub_model_instance_types)
        else:
            raise ValueError(f"Model type not supported: {model_entry.type}")

    return instance_types


def run_track(args):
    import torch
    from py3r.pose.cli.model_repository import ModelDetector
    from py3r.pose.core.model.composite_pose_model import CompositePoseModel
    from py3r.pose.cli.prediction_job import PredictionJob
    from py3r.media.video.ffmpeg_video_file_source import FFmpegVideoFileSource

    if args.input_file:
        input_files = [args.input_file]
    elif args.input_folder:
        video_files = list(args.input_folder.glob("*.*"))
        input_files = [video_file for video_file in video_files if video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
    else:
        raise ValueError("No input source specified")

    if args.output_file:
        if len(input_files) > 1:
            raise ValueError("--output-file cannot be used with multiple input files. Use --output-folder instead.")
        output_files = [args.output_file]
    elif args.output_folder:
        output_files = []
        for input_file in input_files:
            output_file = args.output_folder / (input_file.stem + "_poses.csv")
            output_files.append(output_file)
    else:
        output_files = [None] * len(input_files)

    if args.vis_file:
        if len(input_files) > 1:
            raise ValueError("--vis-file cannot be used with multiple input files. Use --vis-folder instead.")
        if not args.vis_file.suffix.lower() == ".mp4":
            raise ValueError("Visualization file must have .mp4 extension")
        vis_files = [args.vis_file]
    elif args.vis_folder:
        vis_files = []
        for i, input_file in enumerate(input_files):
            if args.vis_first_only and i > 0:
                vis_files.append(None)
                continue
            vis_file = args.vis_folder / (input_file.stem + "_vis.mp4")
            vis_files.append(vis_file)
    else:
        vis_files = [None] * len(input_files)

    tracker = None
    if args.tracker == "fixed-instances":
        from py3r.pose.core.tracking.fixed_instances_tracker import FixedInstancesTracker
        tracker = FixedInstancesTracker(args.instances)

    for i, (input_file, output_file, vis_file) in enumerate(zip(input_files, output_files, vis_files)):
        source = FFmpegVideoFileSource(input_file, grayscale=args.grayscale, playback=False)

        if args.start_time is not None:
            start_frame = int(args.start_time * source.get_fps())
        elif args.start_frame is not None:
            start_frame = args.start_frame
        else:
            start_frame = 0

        if args.end_time is not None:
            end_frame = int(args.end_time * source.get_fps())
        elif args.end_frame is not None:
            end_frame = args.end_frame
        else:
            end_frame = None

        if args.command == "render":
            instance_types = None
            if args.model:
                instance_types = []
                for model_identifier in args.model:
                    instance_types.extend(get_instance_types_from_model(model_identifier))

            poses_source = PoseCSVReader(output_file, instance_types=instance_types)
            if instance_types is None:
                instance_types = poses_source.get_instance_types()

            pose_renderer = build_pose_renderer(args, instance_types)
            job = RenderJob()
            job.set_poses_source(poses_source)
            job.set_video_source(source)

            job.set_start_frame(start_frame)
            job.set_end_frame(end_frame)

            if args.original_speed:
                job.set_rate_limit(source.get_fps())

            job.set_visualization_file(vis_file)

            job.set_pose_renderer(pose_renderer)
            job.set_tracker(tracker)

            job.set_live_preview(args.live_preview)
            job.set_quiet(args.quiet)
        else:
            models = [build_model(model_identifier) for model_identifier in args.model]

            if len(models) == 1:
                model = models[0]
            else:
                model = CompositePoseModel(models)

            pose_renderer = build_pose_renderer(args, model.instance_types)

            job = PredictionJob(model)
            job.set_source(source)

            if args.original_speed:
                job.set_rate_limit(source.get_fps())

            job.set_start_frame(start_frame)
            job.set_end_frame(end_frame)

            job.set_output_file(output_file)
            job.set_visualization_file(vis_file)

            job.set_pose_renderer(pose_renderer)
            job.set_batch_size(args.batch_size)

            #job.set_label_filter(label_filter)
            job.set_tracker(tracker)

            job.set_live_preview(args.live_preview)
            job.set_no_progress(args.quiet)

        if len(input_files) > 1 and not args.quiet:
            print(f"Processing file {i + 1}/{len(input_files)}: {input_file}")
        job.run()

        torch.cuda.empty_cache()


def run_concat(input_files: List[Path], output_file: Path, reset_frame_index: bool = False):
    job = ConcatJob()

    pose_sources = [
        PoseCSVReader(input_file)
        for input_file in input_files
    ]
    job.set_pose_sources(pose_sources)

    job.set_output_file(output_file)
    job.set_reset_frame_index(reset_frame_index)
    job.run()


def run_merge(input_files: List[Path], output_file: Path):
    job = MergeJob()

    pose_sources = [
        PoseCSVReader(input_file)
        for input_file in input_files
    ]
    job.set_pose_sources(pose_sources)

    job.set_output_file(output_file)
    job.run()


def list_available_models():
    groups = available_models.filter_by_type("pose_estimation_model_group")
    groups = groups.filter(lambda e: all(available_models.get(*model.rsplit(";", 1)[0].split(":")) for model in e.manifest["models"]))

    print()
    if len(groups.models) > 0:
        print("--Available Presets--")
        for model_id, versions in groups.models.items():
            print(model_id)
        print()

    models = available_models.filter_by_type("yolo_pose_3rhub")
    if len(models.models) > 0:
        print("--Available Models--")
        for model_id, versions in models.models.items():
            print(model_id)
        print()

    if len(groups.models) == 0 and len(models.models) == 0:
        print("No available models found.")
    else:
        print("Use -m <model_name> or -m <model_name>:<version> to specify the model to use for tracking.")


def _track_arguments(parser):
    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument("input", nargs="?", type=Path, help="Path to input video file or folder")
    input.add_argument("--input-file", type=Path, help="Path to input folder containing video files")
    input.add_argument("--input-folder", type=Path, help="Path to input folder containing video files")

    output = parser.add_mutually_exclusive_group(required=False)
    output.add_argument('--output-file', type=Path, help="Path to output file")
    output.add_argument('--output-folder', type=Path, help="Path to output folder")
    output.add_argument('--no-output', action='store_true', help="No output will be saved")

    visualization = parser.add_mutually_exclusive_group(required=False)
    visualization.add_argument('--vis-file', type=Path, help="Path to visualization file")
    visualization.add_argument('--vis-folder', type=Path, help="Path to visualization folder")
    visualization.add_argument('--no-vis', action='store_true', help="No visualization will be saved")

    parser.add_argument("--model", "-m", type=str, nargs="+", help="Model identifier(s) (name:version) or path to model folder(s)", default=[])

    parser.add_argument("--batch-size", "-b", type=int, help="Batch size", default=32)

    parser.add_argument("--tracker", type=str, help="Tracker type", default=None, choices=["fixed-instances"])
    parser.add_argument("--instances", type=str, nargs="+", help="Instance types to track", default=[])

    start = parser.add_mutually_exclusive_group(required=False)
    start.add_argument("--start-frame", type=int, help="Start frame index")
    start.add_argument("--start-time", type=float, help="Start time in seconds")

    end = parser.add_mutually_exclusive_group(required=False)
    end.add_argument("--end-frame", type=int, help="End frame index")
    end.add_argument("--end-time", type=float, help="End time in seconds")

    parser.add_argument("--original-speed", "-p", action="store_true", help="Process at original video speed")
    parser.add_argument("--grayscale", "-g", action="store_true", help="Process video in grayscale")
    parser.add_argument("--live-preview", "-l", action="store_true", help="Show live preview during processing")
    parser.add_argument("--vis-first-only", action="store_true", help="Only visualize the first video")
    parser.add_argument("--quiet", "-q", action="store_true", help="Do not show progress bar")

    parser.add_argument("--no-skeleton", action="store_true", help="Do not show skeleton in visualization")
    parser.add_argument("--show-bounding-box", action="store_true", help="Show bounding box in visualization")
    parser.add_argument("--show-instance-id", action="store_true", help="Show instance ID in visualization")
    parser.add_argument("--show-class-name", action="store_true", help="Show class name in visualization")
    parser.add_argument("--show-confidence", action="store_true", help="Show confidence in visualization")
    parser.add_argument("--point-radius", type=int, help="Point radius in visualization", default=4)
    parser.add_argument("--skeleton-thickness", type=int, help="Skeleton thickness in visualization", default=2)
    parser.add_argument("--bounding-box-color", type=str, help="Bounding box color in visualization (6-digit hex code, e.g. #a1b2c3)", default="#0000ff")
    parser.add_argument("--skeleton-color", type=str, help="Skeleton color in visualization (6-digit hex code, e.g. #a1b2c3)", default="#000000")
    parser.add_argument("--class-name-color", type=str, help="Class name color in visualization (6-digit hex code, e.g. #a1b2c3)", default="#ffffff")

    parser.add_argument("--render", action="store_true", help="Use existing pose data to create visualization / live preview")

    return parser


def _concat_arguments(parser):
    parser.add_argument("input_files", nargs="+", type=Path, help="Paths to input pose files")
    parser.add_argument("--output-file", '-o', type=Path, required=True, help="Path to output pose file")
    parser.add_argument("--reset-frame-index", action="store_true", help="Reset frame index to 0 for each input file")


def _merge_arguments(parser):
    parser.add_argument("input_files", nargs="+", type=Path, help="Paths to input pose files")
    parser.add_argument("--output-file", '-o', type=Path, required=True, help="Path to output pose file")


def _stereo_track_arguments(parser):
    parser.add_argument("--input-file", type=Path, help="Path to input file")
    parser.add_argument("--output-file-left", type=Path, help="Path to output file for left camera")
    parser.add_argument("--output-file-right", type=Path, help="Path to output file for right camera")

    parser.add_argument("--model", "-m", type=str, nargs="+", help="Model identifier(s) (name:version) or path to model folder(s)", default=[])
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size", default=32)


def main():
    parser = argparse.ArgumentParser(description="Pose Estimation")
    subparsers = parser.add_subparsers(dest="command")
    track_parser = subparsers.add_parser("track", help="Track objects in video")
    render_parser = subparsers.add_parser("render", help="Render video with existing pose data")

    concat_parser = subparsers.add_parser("concat", help="Concatenate multiple pose results files")
    merge_parser = subparsers.add_parser("merge", help="Merge multiple pose results files")
    stereo_parser = subparsers.add_parser("stereo_track", help="Track objects in stereo video")

    _track_arguments(track_parser)
    _track_arguments(render_parser)
    _concat_arguments(concat_parser)
    _merge_arguments(merge_parser)

    args = parser.parse_args()

    if args.command in ["track", "render"]:
        if args.input:
            if args.input.is_file():
                args.input_file = args.input
            elif args.input.is_dir():
                args.input_folder = args.input
            else:
                parser.error(f"Input path does not exist or is not a valid file/folder: {args.input}")

        if not args.no_output and not args.output_file and not args.output_folder:
            args.output_folder = args.input_folder if args.input_folder else args.input_file.parent

        if not args.no_vis and not args.vis_file and not args.vis_folder:
            args.vis_folder = args.input_folder if args.input_folder else args.input_file.parent

        if args.input_folder and args.output_file:
            parser.error("--output-file cannot be used when the input is a folder. Use --output-folder instead.")

        if args.input_folder and args.vis_file:
            parser.error("--vis-file cannot be used when the input is a folder. Use --vis-folder instead.")

        if args.tracker == "fixed-instances":
            if not args.instances:
                parser.error("At least one instance type must be specified for fixed-instances tracker using --instances")

        if args.command == "track" and len(args.model) == 0:
            list_available_models()
            exit(1)

        for model in args.model:
            if ";" in model:
                model_identifier, intermittent_interval = model.rsplit(";", 1)
            else:
                model_identifier = model

            if not Path(model_identifier).is_dir():
                model_entry = available_models.get(*model_identifier.split(":", 1))

                if model_entry is None:
                    print(f"Model not found: {model_identifier}")
                    list_available_models()
                    exit(1)
        run_track(args)
    elif args.command == "concat":
        run_concat([Path(input_file) for input_file in args.input_files], args.output_file, args.reset_frame_index)
    elif args.command == "merge":
        run_merge([Path(input_file) for input_file in args.input_files], args.output_file)
    elif args.command == "stereo_track":
        run_stereo_track



if __name__ == "__main__":
    main()