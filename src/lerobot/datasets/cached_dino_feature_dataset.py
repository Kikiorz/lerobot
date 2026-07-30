#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset wrapper for precomputed frozen-DINO feature maps.

The cache is a raw C-contiguous array addressed by the dataset's *absolute*
``index`` column. Keeping it outside the LeRobot dataset avoids duplicating
large video-derived tensors into parquet files. A typical manifest is::

    {
      "version": 1,
      "cache_path": "/proc/<cache-server-pid>/fd/<fd>",
      "shape": [num_frames, num_cameras, 768, 30, 40],
      "dtype": "float16",
      "camera_keys": ["observation.images.front", "..."],
      "ready": true
    }

The memory map is opened lazily in each DataLoader process. This is important
for ``/proc/<pid>/fd/<fd>`` cache-server paths: it neither serializes the map
nor opens it before worker processes have been created.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.utils.constants import OBS_DINO_FEATURES

from .lerobot_dataset import LeRobotDataset


class CachedDinoFeatureDataset(Dataset):
    """Wrap a :class:`LeRobotDataset` and substitute cached DINO features for videos.

    The underlying reader still supplies state/action columns and all requested
    delta-timestamp windows. Only video decoding is skipped. Cache rows are
    addressed by ``item['index']`` rather than the wrapper's relative index so
    episode-filtered datasets use the correct feature row.
    """

    def __init__(self, dataset: LeRobotDataset, manifest_path: str | Path):
        super().__init__()
        self.dataset = dataset
        self.manifest_path = Path(manifest_path).expanduser()
        manifest = self._load_manifest(self.manifest_path)

        self.cache_path = self._get_cache_path(manifest, self.manifest_path)
        self.cache_shape = self._get_shape(manifest)
        self.cache_dtype = self._get_dtype(manifest)
        self.cache_camera_keys = self._get_camera_keys(manifest)
        self._camera_indices = self._validate_dataset_compatibility()
        self._validate_cache_file()

        # These must stay process-local. A DataLoader can fork after a sample
        # was inspected in the parent process, so track PID as well as clearing
        # them during pickling for spawn-based workers.
        self._mmap: np.memmap | None = None
        self._mmap_pid: int | None = None

    @staticmethod
    def _load_manifest(manifest_path: Path) -> dict[str, Any]:
        try:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"DINO feature cache manifest does not exist: {manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"DINO feature cache manifest is not valid JSON: {manifest_path}") from exc

        if not isinstance(manifest, dict):
            raise ValueError(f"DINO feature cache manifest must contain a JSON object: {manifest_path}")
        if manifest.get("version") != 1:
            raise ValueError(
                f"Unsupported DINO feature cache manifest version {manifest.get('version')!r}; expected 1"
            )
        if manifest.get("ready") is not True:
            raise ValueError(
                "DINO feature cache manifest is not ready. Finish cache generation before starting training."
            )
        return manifest

    @staticmethod
    def _get_cache_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
        value = manifest.get("cache_path")
        if not isinstance(value, str) or not value:
            raise ValueError("DINO feature cache manifest field 'cache_path' must be a non-empty string")
        cache_path = Path(value).expanduser()
        # Do not resolve /proc/<pid>/fd/<fd>: resolving it can replace the
        # descriptor path with a deleted backing path and make later workers
        # unable to reopen the cache. Relative paths are manifest-relative.
        return cache_path if cache_path.is_absolute() else manifest_path.parent / cache_path

    @staticmethod
    def _get_shape(manifest: dict[str, Any]) -> tuple[int, ...]:
        value = manifest.get("shape")
        if not isinstance(value, list) or len(value) < 3:
            raise ValueError(
                "DINO feature cache manifest field 'shape' must be a list like "
                "[num_frames, num_cameras, ...feature_shape]"
            )
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in value):
            raise ValueError("DINO feature cache manifest field 'shape' must contain only positive integers")
        return tuple(value)

    @staticmethod
    def _get_dtype(manifest: dict[str, Any]) -> np.dtype:
        value = manifest.get("dtype")
        if not isinstance(value, str):
            raise ValueError("DINO feature cache manifest field 'dtype' must be a string")
        try:
            dtype = np.dtype(value)
        except TypeError as exc:
            raise ValueError(f"Unsupported DINO feature cache dtype: {value!r}") from exc
        if dtype != np.dtype(np.float16):
            raise ValueError(f"DINO feature cache must use float16, got {dtype}")
        return dtype

    def _get_camera_keys(self, manifest: dict[str, Any]) -> list[str]:
        value = manifest.get("camera_keys")
        if not isinstance(value, list) or not value or not all(isinstance(key, str) and key for key in value):
            raise ValueError("DINO feature cache manifest field 'camera_keys' must be a non-empty list of strings")
        if len(value) != len(set(value)):
            raise ValueError("DINO feature cache manifest field 'camera_keys' contains duplicate keys")
        if len(value) != self.cache_shape[1]:
            raise ValueError(
                "DINO feature cache camera dimension does not match 'camera_keys': "
                f"shape[1]={self.cache_shape[1]}, camera_keys={len(value)}"
            )
        return value

    def _validate_dataset_compatibility(self) -> tuple[int, ...]:
        if self.cache_shape[0] < self.dataset.meta.total_frames:
            raise ValueError(
                "DINO feature cache has too few rows for dataset absolute indices: "
                f"cache={self.cache_shape[0]}, dataset={self.dataset.meta.total_frames}"
            )

        dataset_camera_keys = list(self.dataset.meta.camera_keys)
        if set(self.cache_camera_keys) != set(dataset_camera_keys):
            raise ValueError(
                "DINO feature cache camera keys do not match the dataset. "
                f"cache={self.cache_camera_keys}, dataset={dataset_camera_keys}"
            )
        return tuple(self.cache_camera_keys.index(key) for key in dataset_camera_keys)

    def _validate_cache_file(self) -> None:
        try:
            size_bytes = os.stat(self.cache_path).st_size
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"DINO feature cache does not exist: {self.cache_path}") from exc
        except OSError as exc:
            raise OSError(f"Cannot access DINO feature cache {self.cache_path}: {exc}") from exc

        expected_bytes = math.prod(self.cache_shape) * self.cache_dtype.itemsize
        if size_bytes != expected_bytes:
            raise ValueError(
                "DINO feature cache byte size does not match its manifest: "
                f"expected {expected_bytes:,}, found {size_bytes:,} at {self.cache_path}"
            )

    def _get_mmap(self) -> np.memmap:
        pid = os.getpid()
        if self._mmap is None or self._mmap_pid != pid:
            try:
                # Copy-on-write makes the NumPy view writable for PyTorch
                # without ever modifying the shared raw cache on disk/RAM.
                self._mmap = np.memmap(
                    self.cache_path,
                    dtype=self.cache_dtype,
                    mode="c",
                    shape=self.cache_shape,
                    order="C",
                )
            except OSError as exc:
                raise OSError(f"Unable to memory-map DINO feature cache {self.cache_path}: {exc}") from exc
            self._mmap_pid = pid
        return self._mmap

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_mmap"] = None
        state["_mmap_pid"] = None
        return state

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        reader = self.dataset._ensure_reader()
        if reader.hf_dataset is None:
            reader.load_and_activate()
        item = reader.get_item(idx, decode_videos=False)

        absolute_index = int(item["index"].item())
        if absolute_index < 0 or absolute_index >= self.cache_shape[0]:
            raise IndexError(
                "DINO feature cache does not contain dataset absolute index "
                f"{absolute_index} (cache rows: {self.cache_shape[0]})"
            )

        features = self._get_mmap()[absolute_index]
        # Manifest camera ordering is explicit. Return the dataset's camera
        # order so policy construction and the cached tensor agree even if a
        # cache was produced with a different ordering.
        if self._camera_indices != tuple(range(len(self._camera_indices))):
            features = features[list(self._camera_indices)]
        item[OBS_DINO_FEATURES] = torch.from_numpy(features)
        return item

    # The training pipeline reads these attributes from the dataset facade.
    # Delegate all remaining read-only dataset metadata/helpers to the wrapped
    # LeRobotDataset rather than maintaining a second copy.
    @property
    def meta(self):
        return self.dataset.meta

    @property
    def repo_id(self):
        return self.dataset.repo_id

    @property
    def root(self):
        return self.dataset.root

    @property
    def episodes(self):
        return self.dataset.episodes

    @property
    def delta_timestamps(self):
        return self.dataset.delta_timestamps

    @property
    def fps(self):
        return self.dataset.fps

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    @property
    def features(self):
        return self.dataset.features

    @property
    def hf_dataset(self):
        return self.dataset.hf_dataset

    @property
    def reader(self):
        return self.dataset.reader

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dataset!r}, manifest_path={str(self.manifest_path)!r})"
