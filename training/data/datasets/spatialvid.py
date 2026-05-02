import os.path as osp
import numpy as np
import cv2
import numpy as np
import json
import os
import sys
import pandas as pd
from decord import VideoReader
import gc
from contextlib import contextmanager
from collections.abc import Mapping, Sequence

from tqdm import tqdm
from ..base_dataset import BaseDataset


def _pose_vec_xyzw_to_c2w(pose_vec):
    pose_vec = np.asarray(pose_vec, dtype=np.float32)
    translation = pose_vec[:3]
    x, y, z, w = pose_vec[3:7]
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    return c2w


def _intrinsics_vec_to_matrix(intr_vec, width, height):
    fx, fy, cx, cy = np.asarray(intr_vec, dtype=np.float32)
    intr = np.eye(3, dtype=np.float32)
    intr[0, 0] = fx * width
    intr[1, 1] = fy * height
    intr[0, 2] = cx * width
    intr[1, 2] = cy * height
    return intr


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


class SpatialVID(BaseDataset):
    CONTEXT_SAMPLING_STRATEGIES = (
        "uniform",
        "uniform_first",
        "random_first",
        "window_with_first",
        "prefix",
        "first_plus_sparse",
        "first_plus_recent_window",
        "mixed",
    )

    def __init__(
        self,
        ROOT,
        video_ids=None,
        video_paths=None,
        use_camera_annotations=False,
        continuous_target_frames=False,
        force_first_context=True,
        timestamp_unit="seconds",
        context_sampling_strategy="uniform",
        context_sampling_weights=None,
        variants_per_scene=1,
        *args,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video_ids = self._normalize_filter_values(video_ids)
        self.video_paths = self._normalize_filter_values(video_paths)
        self.use_camera_annotations = use_camera_annotations
        self.continuous_target_frames = bool(continuous_target_frames)
        self.force_first_context = bool(force_first_context)
        self.context_sampling_strategy = str(context_sampling_strategy)
        self.context_sampling_weights = context_sampling_weights
        self.variants_per_scene = max(1, int(variants_per_scene))
        if self.context_sampling_strategy not in self.CONTEXT_SAMPLING_STRATEGIES:
            raise ValueError(
                f"Unsupported context_sampling_strategy={self.context_sampling_strategy!r}. "
                f"Expected one of {self.CONTEXT_SAMPLING_STRATEGIES}."
            )
        if timestamp_unit not in {"seconds", "frames"}:
            raise ValueError(f"Unsupported timestamp_unit={timestamp_unit!r}. Expected 'seconds' or 'frames'.")
        self.timestamp_unit = timestamp_unit
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    @staticmethod
    def _normalize_filter_values(values):
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        return {str(value) for value in values}

    def _load_data(self):
        metadata = pd.read_csv(osp.join(self.ROOT, "data/train/SpatialVID_HQ_metadata.csv"))
        if self.continuous_target_frames:
            min_clip_length = int(self.num_views)
        else:
            min_anno_length = (self.num_views - 1) * self.min_interval + 1
            annotation_interval = (0.2 * metadata["fps"]).astype(int)
            min_clip_length = annotation_interval * (min_anno_length - 1) + 1
        self.scenes = metadata[metadata["num frames"] >= min_clip_length]
        if self.video_ids is not None:
            self.scenes = self.scenes[self.scenes["id"].astype(str).isin(self.video_ids)]
        if self.video_paths is not None:
            self.scenes = self.scenes[self.scenes["video path"].astype(str).isin(self.video_paths)]
        if len(self.scenes) == 0:
            filters = []
            if self.video_ids is not None:
                filters.append(f"video_ids={sorted(self.video_ids)}")
            if self.video_paths is not None:
                filters.append(f"video_paths={sorted(self.video_paths)}")
            filter_text = ", ".join(filters) if filters else "no filters"
            raise ValueError(f"SpatialVID found no scenes after filtering ({filter_text}).")

    def __len__(self):
        return len(self.scenes) * self.variants_per_scene

    def _scene_variant_index(self, idx):
        idx = int(idx)
        if self.variants_per_scene <= 1:
            return idx, 0
        return idx // self.variants_per_scene, idx % self.variants_per_scene

    def _strategy_for_sample(self, rng):
        strategy = self.context_sampling_strategy
        if strategy != "mixed":
            self._last_context_strategy = strategy
            return strategy
        if self.context_sampling_weights is None:
            choices = (
                "uniform_first",
                "random_first",
                "window_with_first",
                "prefix",
                "first_plus_sparse",
                "first_plus_recent_window",
            )
            weights = np.ones(len(choices), dtype=np.float64)
        elif isinstance(self.context_sampling_weights, Mapping):
            choices = tuple(str(key) for key in self.context_sampling_weights.keys())
            weights = np.array([float(value) for value in self.context_sampling_weights.values()], dtype=np.float64)
        elif isinstance(self.context_sampling_weights, Sequence) and not isinstance(self.context_sampling_weights, str):
            choices = tuple(str(item[0]) for item in self.context_sampling_weights)
            weights = np.array([float(item[1]) for item in self.context_sampling_weights], dtype=np.float64)
        else:
            raise ValueError("context_sampling_weights must be a mapping or a sequence of (strategy, weight) pairs.")
        for choice in choices:
            if choice == "mixed" or choice not in self.CONTEXT_SAMPLING_STRATEGIES:
                raise ValueError(f"Invalid mixed context sampling choice: {choice!r}.")
        if len(choices) == 0 or weights.sum() <= 0:
            raise ValueError("context_sampling_weights must contain at least one positive weight.")
        weights = weights / weights.sum()
        strategy = str(rng.choice(choices, p=weights))
        self._last_context_strategy = strategy
        return strategy

    def _context_local_indices(self, num_context_views):
        num_context_views = int(np.clip(num_context_views, 1, self.num_views))
        if num_context_views >= self.num_views:
            self._last_context_strategy = "all"
            return np.arange(self.num_views, dtype=np.int64)
        if self.force_first_context:
            return self._context_local_indices_with_first(num_context_views)
        self._last_context_strategy = "uniform_no_first"
        indices = np.linspace(0, self.num_views - 1, num_context_views, dtype=int)
        return np.unique(indices).astype(np.int64)

    def _context_local_indices_with_first(self, num_context_views):
        if num_context_views <= 1:
            return np.array([0], dtype=np.int64)
        strategy = self._strategy_for_sample(self._rng)
        count = num_context_views - 1
        candidates = np.arange(1, self.num_views, dtype=np.int64)
        if strategy in {"uniform", "uniform_first"}:
            indices = np.linspace(0, self.num_views - 1, num_context_views, dtype=int)
        elif strategy == "random_first":
            rest = self._rng.choice(candidates, size=count, replace=False)
            indices = np.concatenate(([0], np.sort(rest)))
        elif strategy == "window_with_first":
            window = min(count, self.num_views - 1)
            start = int(self._rng.integers(1, self.num_views - window + 1))
            indices = np.concatenate(([0], np.arange(start, start + window, dtype=np.int64)))
        elif strategy == "prefix":
            indices = np.arange(num_context_views, dtype=np.int64)
        elif strategy == "first_plus_sparse":
            rest = np.linspace(1, self.num_views - 1, count, dtype=int)
            indices = np.concatenate(([0], rest))
        elif strategy == "first_plus_recent_window":
            window = min(count, self.num_views - 1)
            max_start = self.num_views - window
            min_start = max(1, self.num_views - max(window * 3, window))
            start = int(self._rng.integers(min_start, max_start + 1))
            indices = np.concatenate(([0], np.arange(start, start + window, dtype=np.int64)))
        else:
            raise ValueError(f"Unsupported sampled context strategy: {strategy!r}")
        indices = np.unique(indices).astype(np.int64)
        if len(indices) < num_context_views:
            missing = num_context_views - len(indices)
            pool = np.setdiff1d(candidates, indices, assume_unique=False)
            if len(pool) > 0:
                extra = self._rng.choice(pool, size=min(missing, len(pool)), replace=False)
                indices = np.unique(np.concatenate((indices, extra))).astype(np.int64)
        return np.sort(indices[:num_context_views]).astype(np.int64)

    def _timestamp(self, sample_index, time_index, v, first_sample_index, first_time_index, reverse, fps):
        if self.timestamp_unit == "seconds":
            delta = sample_index[v] - first_sample_index
            if reverse:
                delta = -delta
            return np.float32(delta / max(float(fps), 1e-6))
        delta = time_index - first_time_index if not reverse else first_time_index - time_index
        return np.float32(delta)

    def _get_views(self, idx, rng, num_context_views):
        scene_idx, variant_id = self._scene_variant_index(idx)
        scene_info = self.scenes.iloc[scene_idx]
        video_path = osp.join(self.ROOT, "SpatialVid/HQ", scene_info["video path"])
        annotation_dir = osp.join(self.ROOT, "SpatialVid/HQ", scene_info["annotation path"])
        poses_path = osp.join(annotation_dir, "poses.npy")
        intrinsics_path = osp.join(annotation_dir, "intrinsics.npy")
        has_camera_annotations = self.use_camera_annotations and osp.exists(poses_path) and osp.exists(intrinsics_path)
        poses = np.load(poses_path).astype(np.float32) if has_camera_annotations else None
        intrinsics = np.load(intrinsics_path).astype(np.float32) if has_camera_annotations else None
        if has_camera_annotations:
            annotation_length = min(len(poses), len(intrinsics))
            poses = poses[:annotation_length]
            intrinsics = intrinsics[:annotation_length]

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            video_length = len(video_reader)
            fps = float(scene_info["fps"])
            reverse = False
            if self.continuous_target_frames:
                clip_length = int(self.num_views)
                start = rng.integers(0, max(video_length - clip_length, 0) + 1)
                sample_index = np.arange(start, start + clip_length, dtype=np.int64)
                if has_camera_annotations:
                    video_lookup = np.linspace(0, video_length - 1, annotation_length)
                    sample_anno_index = np.abs(video_lookup[None, :] - sample_index[:, None]).argmin(axis=1)
                else:
                    sample_anno_index = None
            elif has_camera_annotations:
                sample_anno_index, reverse = self.sample_from_video(
                    annotation_length, self.num_views, self.min_interval, self.max_interval, rng
                )
                video_lookup = np.linspace(0, video_length - 1, annotation_length, dtype=int)
                sample_index = video_lookup[sample_anno_index]
            else:
                sample_index, reverse = self.sample_from_video(
                    video_length, self.num_views, self.min_interval, self.max_interval, rng
                )
                sample_anno_index = None
            images = video_reader.get_batch(sample_index).asnumpy()

        with open(osp.join(annotation_dir, "caption.json"), 'r') as f:
            captions = json.load(f)
            text_prompt = captions["SceneDescription"]

        context_local_indices = set(self._context_local_indices(num_context_views).tolist())
        context_strategy = getattr(self, "_last_context_strategy", self.context_sampling_strategy)
        first_sample_index = sample_index[0]
        first_time_index = sample_anno_index[0] if has_camera_annotations else sample_index[0]
        context_views = []
        target_views = []
        for v, rgb_image in enumerate(images):
            time_index = sample_anno_index[v] if has_camera_annotations else sample_index[v]
            timestamp = self._timestamp(sample_index, time_index, v, first_sample_index, first_time_index, reverse, fps)
            original_height, original_width = rgb_image.shape[:2]
            camera_pose = None
            camera_intr = None
            if has_camera_annotations:
                anno_index = sample_anno_index[v]
                camera_pose = _pose_vec_xyzw_to_c2w(poses[anno_index])
                camera_intr = _intrinsics_vec_to_matrix(intrinsics[anno_index], original_width, original_height)
            rgb_image, _, camera_intr = self._crop_resize_if_necessary(
                rgb_image, (self.width, self.height), rng=rng, info=(idx, v), intrinsics=camera_intr,
            )
            view = dict(
                img=rgb_image,
                dataset="SpatialVID",
                video_name=scene_info["id"],
                image_name=f"frame_{sample_index[v]:06d}",
                is_static=False,
                timestamp=timestamp,
                prompt=text_prompt,
                scene_idx=scene_idx,
                variant_id=variant_id,
                context_strategy=context_strategy,
            )
            if has_camera_annotations:
                view["camera_poses"] = camera_pose.astype(np.float32)
                view["camera_intrs"] = camera_intr.astype(np.float32)
            if v in context_local_indices:
                view["is_target"] = False
                context_views.append(
                    view
                )
            else:
                view["is_target"] = True
                target_views.append(
                    view
                )
        views = context_views + target_views
        return views
