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
"""Tests for the precomputed DINO feature dataset wrapper."""

import json
import os

import numpy as np
import pytest
import torch

from lerobot.datasets.cached_dino_feature_dataset import CachedDinoFeatureDataset
from lerobot.utils.constants import ACTION, OBS_DINO_FEATURES


def _write_manifest(path, cache_path, shape, camera_keys, *, ready=True):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cache_path": str(cache_path),
                "shape": list(shape),
                "dtype": "float16",
                "camera_keys": camera_keys,
                "ready": ready,
            }
        ),
        encoding="utf-8",
    )


def test_cached_dino_features_skip_video_decode_and_keep_delta_queries(
    tmp_path, lerobot_dataset_factory, monkeypatch
):
    """The wrapper uses absolute indices, skips video decode, and retains deltas."""
    fps = 30
    dataset = lerobot_dataset_factory(
        root=tmp_path / "dataset",
        total_episodes=1,
        total_frames=8,
        use_videos=True,
        delta_timestamps={"state": [-1 / fps, 0.0], ACTION: [0.0, 1 / fps]},
    )
    shape = (dataset.meta.total_frames, len(dataset.meta.camera_keys), 2, 3, 4)
    cache = np.arange(np.prod(shape), dtype=np.float16).reshape(shape)
    cache_file = tmp_path / "dino_features.bin"
    cache.tofile(cache_file)
    manifest_path = tmp_path / "dino_features.json"

    # A cache server exposes its memfd in this form. Keeping the descriptor
    # open also verifies that the wrapper preserves the /proc path verbatim.
    with cache_file.open("rb") as cache_fd:
        proc_cache_path = f"/proc/self/fd/{cache_fd.fileno()}"
        _write_manifest(manifest_path, proc_cache_path, shape, list(dataset.meta.camera_keys))
        cached_dataset = CachedDinoFeatureDataset(dataset, manifest_path)
        assert cached_dataset._mmap is None

        def fail_if_decoded(*args, **kwargs):
            pytest.fail("video decoding must be skipped when using cached DINO features")

        monkeypatch.setattr(dataset.reader, "_query_videos", fail_if_decoded)
        item = cached_dataset[3]

    assert OBS_DINO_FEATURES in item
    assert item[OBS_DINO_FEATURES].dtype == torch.float16
    assert item[OBS_DINO_FEATURES].shape == torch.Size(shape[1:])
    torch.testing.assert_close(item[OBS_DINO_FEATURES], torch.from_numpy(cache[3]))
    # The underlying reader still handles non-visual temporal queries.
    assert item["state"].shape == torch.Size((2, 6))
    assert item[ACTION].shape == torch.Size((2, 6))
    assert "state_is_pad" in item
    assert f"{ACTION}_is_pad" in item


def test_cached_dino_features_reject_mismatched_raw_cache_size(tmp_path, lerobot_dataset_factory):
    """Fail before training if cache bytes cannot represent the declared shape."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "dataset",
        total_episodes=1,
        total_frames=4,
        use_videos=False,
    )
    shape = (dataset.meta.total_frames, len(dataset.meta.camera_keys), 2, 3, 4)
    cache_file = tmp_path / "truncated.bin"
    cache_file.write_bytes(b"too short")
    manifest_path = tmp_path / "dino_features.json"
    _write_manifest(manifest_path, cache_file, shape, list(dataset.meta.camera_keys))

    with pytest.raises(ValueError, match="byte size"):
        CachedDinoFeatureDataset(dataset, manifest_path)


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="requires Linux /proc descriptor paths")
def test_cached_dino_features_reject_not_ready_manifest(tmp_path, lerobot_dataset_factory):
    """A cache generator must publish ``ready`` only after all rows are written."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "dataset",
        total_episodes=1,
        total_frames=4,
        use_videos=False,
    )
    shape = (dataset.meta.total_frames, len(dataset.meta.camera_keys), 2, 3, 4)
    cache_file = tmp_path / "dino_features.bin"
    np.zeros(shape, dtype=np.float16).tofile(cache_file)
    manifest_path = tmp_path / "dino_features.json"
    _write_manifest(manifest_path, cache_file, shape, list(dataset.meta.camera_keys), ready=False)

    with pytest.raises(ValueError, match="not ready"):
        CachedDinoFeatureDataset(dataset, manifest_path)
