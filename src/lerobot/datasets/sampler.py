#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import BatchSampler, SequentialSampler

logger = logging.getLogger(__name__)


class CacheLocalityPairedBatchSampler(BatchSampler):
    """Yield contiguous batches for a memory-mapped feature cache.

    ``CachedDinoFeatureDataset`` retrieves one large feature row per dataset
    index. The ordinary random sampler is statistically useful, but on a cache
    larger than RAM it turns every batch into a collection of unrelated mmap
    page faults. This sampler keeps reads contiguous instead. With the default
    ``shuffle_blocks=False`` it yields ascending dataset indices and therefore
    makes no random cache reads at all.

    It deliberately exposes the normal :class:`~torch.utils.data.BatchSampler`
    attributes (``sampler``, ``batch_size``, and ``drop_last``). Accelerate can
    consequently wrap it in ``BatchSamplerShard`` unchanged. In the common
    two-rank, no-split configuration, each adjacent source-batch pair is
    contiguous: rank 0 receives the first batch and rank 1 receives the second.

    ``locality_batch_size`` is a contiguous block boundary. It must be a
    multiple of ``batch_size`` so that no partial batch is introduced between
    blocks. The current training option uses sequential blocks. ``shuffle_blocks``
    is retained for callers that want a shuffled *block* order while preserving
    contiguous rows inside every batch; it is off by default.
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        locality_batch_size: int,
        drop_last: bool = False,
        *,
        shuffle_blocks: bool = False,
        generator: torch.Generator | None = None,
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if (
            isinstance(locality_batch_size, bool)
            or not isinstance(locality_batch_size, int)
            or locality_batch_size <= 0
        ):
            raise ValueError(
                "locality_batch_size must be a positive integer, "
                f"got {locality_batch_size!r}"
            )
        if locality_batch_size % batch_size != 0:
            raise ValueError(
                "locality_batch_size must be a multiple of batch_size so each "
                f"cache-locality block contains complete batches, got "
                f"locality_batch_size={locality_batch_size}, batch_size={batch_size}"
            )

        super().__init__(SequentialSampler(data_source), batch_size=batch_size, drop_last=drop_last)
        self.locality_batch_size = locality_batch_size
        self.shuffle_blocks = shuffle_blocks
        self.generator = generator

    def _block_starts(self) -> list[int]:
        """Return full locality-block starts, leaving the partial tail last.

        Leaving the tail in its natural final position preserves PyTorch's
        usual ``drop_last`` semantics and lets Accelerate handle its final
        even-batch padding exactly as it does for the standard DataLoader.
        """

        num_full_blocks = len(self.sampler) // self.locality_batch_size
        starts = list(range(0, num_full_blocks * self.locality_batch_size, self.locality_batch_size))
        if self.shuffle_blocks and len(starts) > 1:
            order = torch.randperm(len(starts), generator=self.generator).tolist()
            starts = [starts[i] for i in order]
        return starts

    def _yield_block(self, start: int, stop: int) -> Iterator[list[int]]:
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            if batch_stop - batch_start == self.batch_size or not self.drop_last:
                yield list(range(batch_start, batch_stop))

    def __iter__(self) -> Iterator[list[int]]:
        num_samples = len(self.sampler)
        full_block_stop = (num_samples // self.locality_batch_size) * self.locality_batch_size

        for block_start in self._block_starts():
            yield from self._yield_block(block_start, block_start + self.locality_batch_size)

        # The partial cache-locality block is emitted last, even when full
        # blocks are shuffled. This is the only location where a final partial
        # batch can occur.
        if full_block_stop < num_samples:
            yield from self._yield_block(full_block_stop, num_samples)


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
