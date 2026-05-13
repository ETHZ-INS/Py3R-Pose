import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Callable, Dict, Any, List, Union, Optional

import yaml


@dataclass
class ModelEntry:
    folder: Path
    manifest: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def version(self) -> str:
        return self.manifest.get("version", "0.0.0")

    @property
    def key(self) -> Tuple[str, str]:
        return self.id, self.version

    @property
    def name(self) -> str:
        return self.manifest.get("name", "")

    @property
    def task(self) -> str:
        return self.manifest.get("task", "")

    @property
    def type(self) -> str:
        return self.manifest.get("type", "")


class ModelList:
    def __init__(self, models: Union[Dict[str, Dict[str, ModelEntry]], List[ModelEntry]]):
        if isinstance(models, list):
            self.models = {}
            for model in models:
                if model.id not in self.models:
                    self.models[model.id] = {}
                self.models[model.id][model.version] = model
        else:
            self.models = models

    def get(self, model_id: str, version: str = "latest") -> Optional[ModelEntry]:
        versions = self.models.get(model_id)

        if not versions:
            return None

        if not version or version == "latest":
            latest_version = max(versions.keys(), key=lambda v: tuple(map(int, v.split('.'))))
            return versions[latest_version]

        return versions.get(version)

    def filter(self, model_filter: Callable[[ModelEntry], bool]) -> 'ModelList':
        filtered_models = {}
        for model_id, versions in self.models.items():
            for version, entry in versions.items():
                if model_filter(entry):
                    if model_id not in filtered_models:
                        filtered_models[model_id] = {}
                    filtered_models[model_id][version] = entry
        return ModelList(filtered_models)

    def filter_by_task(self, task: str) -> 'ModelList':
        return self.filter(lambda entry: entry.task == task)

    def filter_by_type(self, type_: str) -> 'ModelList':
        return self.filter(lambda entry: entry.type == type_)

    def __repr__(self) -> str:
        return repr(self.models)


class ModelDetector:
    def __init__(self, model_directories: Optional[List[Path]] = None):
        self.model_directories = model_directories or []

        # Add current working directory and the local appdata model directory
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            self.model_directories.append(Path(local_appdata) / "ETH3RHub" / "models")
        self.model_directories.append(Path.cwd() / "models")

    @staticmethod
    def _detect_manifest(folder: Path) -> Optional[Tuple[str, str, dict]]:
        manifest_file = folder / "manifest.yaml"
        if not manifest_file.exists():
            return None

        try:
            with manifest_file.open("r") as f:
                manifest = yaml.safe_load(f)
            model_id = manifest.get("id")
            if model_id is None:
                return None
            version = manifest.get("version", "0.0.0")
            return model_id, version, manifest
        except (FileNotFoundError, yaml.YAMLError):
            return None

    @staticmethod
    def _detect_bundles(folder: Path) -> List[Tuple[str, str, dict]]:
        bundles = []

        for bundle_file in folder.glob("*.yaml"):
            try:
                with bundle_file.open("r") as f:
                    manifest = yaml.safe_load(f)
                model_id = manifest.get("id")
                if model_id is None:
                    continue
                version = manifest.get("version", "0.0.0")
                bundles.append((model_id, version, manifest))
            except (FileNotFoundError, yaml.YAMLError):
                pass

        return bundles

    def _find_models_in_directory(self, directory: Path, models: dict = None) -> dict:
        models = models if models is not None else {}

        if not directory.exists() or not directory.is_dir():
            return models

        # noinspection PyBroadException
        try:
            manifest_data = self._detect_manifest(directory)

            if manifest_data is not None:
                model_id, version, manifest = manifest_data
                if model_id not in models:
                    models[model_id] = {}
                if version not in models[model_id]:
                    models[model_id][version] = ModelEntry(directory, manifest)

                # If folder contains a manifest, do not search subdirectories
                return models
        except Exception:
            pass

        # noinspection PyBroadException
        try:
            for subfolder in directory.iterdir():
                if subfolder.is_dir() and not subfolder.name.startswith('.'):
                    self._find_models_in_directory(subfolder, models)
        except Exception:
            pass

        # noinspection PyBroadException
        try:
            bundles = self._detect_bundles(directory)
            for model_id, version, manifest in bundles:
                if model_id not in models:
                    models[model_id] = {}
                if version not in models[model_id]:
                    models[model_id][version] = ModelEntry(directory, manifest)
        except Exception:
            pass

        return models

    def find_models(self) -> ModelList:
        models = {}
        for model_directory in self.model_directories:
            self._find_models_in_directory(model_directory, models)
        return ModelList(models)
