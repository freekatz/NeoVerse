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
import hashlib
import random
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path

from tqdm import tqdm
from ..base_dataset import BaseDataset
from ..temporal_trajectory import (
    DEFAULT_TRAJECTORY_PROFILE,
    MAX_CONDITION_FRAMES,
    TRAJECTORY_PROFILE_FORWARD_PAUSE,
    normalize_trajectory_profile,
    sample_training_trajectory,
)


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
        temporal_augmentation=False,
        temporal_trajectory_profile=TRAJECTORY_PROFILE_FORWARD_PAUSE,
        temporal_order="trajectory",
        temporal_max_condition_frames=MAX_CONDITION_FRAMES,
        variants_per_scene=1,
        trajectories_per_clip=None,
        temporal_variant_profile_weights=None,
        fixed_clips_per_scene=0,
        camera_cache_dir=None,
        camera_cache_required=False,
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
        self.temporal_augmentation = bool(temporal_augmentation)
        self.temporal_trajectory_profile = normalize_trajectory_profile(temporal_trajectory_profile or DEFAULT_TRAJECTORY_PROFILE)
        self.temporal_order = str(temporal_order or "trajectory").strip().lower()
        self.temporal_max_condition_frames = max(1, int(temporal_max_condition_frames))
        self.trajectories_per_clip = self._optional_positive_int(trajectories_per_clip)
        variant_count = self.trajectories_per_clip if self.trajectories_per_clip is not None else variants_per_scene
        self.variants_per_scene = max(1, int(variant_count))
        self.temporal_variant_profile_weights = self._normalize_profile_weights(temporal_variant_profile_weights)
        self.temporal_variant_profile_schedule = self._build_profile_schedule(
            self.variants_per_scene,
            self.temporal_variant_profile_weights,
        )
        self.fixed_clips_per_scene = max(0, int(fixed_clips_per_scene or 0))
        self.camera_cache_dir = str(camera_cache_dir) if camera_cache_dir not in (None, "") else None
        self.camera_cache_required = bool(camera_cache_required)
        if self.context_sampling_strategy not in self.CONTEXT_SAMPLING_STRATEGIES:
            raise ValueError(
                f"Unsupported context_sampling_strategy={self.context_sampling_strategy!r}. "
                f"Expected one of {self.CONTEXT_SAMPLING_STRATEGIES}."
            )
        if self.temporal_order not in {"trajectory", "sorted"}:
            raise ValueError("temporal_order must be 'trajectory' or 'sorted'.")
        if timestamp_unit not in {"seconds", "frames"}:
            raise ValueError(f"Unsupported timestamp_unit={timestamp_unit!r}. Expected 'seconds' or 'frames'.")
        self.timestamp_unit = timestamp_unit
        super().__init__(*args, **kwargs)
        if self.camera_cache_dir is not None and self.aug_crop:
            raise ValueError("camera_cache_dir requires aug_crop=False so cached intrinsics match loaded images.")
        self.loaded_data = self._load_data()

    @staticmethod
    def _normalize_filter_values(values):
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        return {str(value) for value in values}

    @staticmethod
    def _optional_positive_int(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip() in ("", "none", "None", "null", "Null"):
            return None
        return max(1, int(value))

    @staticmethod
    def _normalize_profile_weights(weights):
        if weights is None:
            return None
        if isinstance(weights, str) and weights.strip() in ("", "none", "None", "null", "Null"):
            return None
        if isinstance(weights, str):
            items = []
            for part in weights.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        "temporal_variant_profile_weights string must use "
                        "'profile:weight,profile:weight' format."
                    )
                profile, weight = part.split(":", 1)
                items.append((profile.strip(), float(weight.strip())))
        elif isinstance(weights, Mapping):
            items = list(weights.items())
        elif isinstance(weights, Sequence):
            items = list(weights)
        else:
            raise ValueError("temporal_variant_profile_weights must be a string, mapping, sequence, or None.")

        normalized = []
        for profile, weight in items:
            profile = normalize_trajectory_profile(profile)
            weight = float(weight)
            if weight <= 0:
                continue
            normalized.append((profile, weight))
        return normalized or None

    @staticmethod
    def _build_profile_schedule(count, weights):
        count = max(1, int(count))
        if weights is None:
            return None

        total_weight = sum(weight for _, weight in weights)
        raw_counts = [weight / total_weight * count for _, weight in weights]
        counts = [int(np.floor(value)) for value in raw_counts]
        remainder = count - sum(counts)
        fractions = sorted(
            range(len(raw_counts)),
            key=lambda idx: (raw_counts[idx] - counts[idx], raw_counts[idx]),
            reverse=True,
        )
        for idx in fractions[:remainder]:
            counts[idx] += 1

        schedule = []
        current = [0] * len(counts)
        for _ in range(count):
            best_idx = None
            best_score = None
            for idx, quota in enumerate(counts):
                if quota <= 0:
                    continue
                current[idx] += quota
                if best_idx is None or current[idx] > best_score:
                    best_idx = idx
                    best_score = current[idx]
            current[best_idx] -= count
            schedule.append(weights[best_idx][0])
        return tuple(schedule)

    @staticmethod
    def _fixed_clip_starts(video_length, clip_length, count):
        video_length = int(video_length)
        clip_length = int(clip_length)
        count = max(0, int(count))
        max_start = max(video_length - clip_length, 0)
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        if count == 1:
            return np.asarray([0], dtype=np.int64)
        if max_start + 1 <= count:
            return np.arange(max_start + 1, dtype=np.int64)
        starts = np.rint(np.linspace(0, max_start, count)).astype(np.int64)
        return np.unique(np.clip(starts, 0, max_start)).astype(np.int64)

    def _load_data(self):
        metadata = pd.read_csv(osp.join(self.ROOT, "data/train/SpatialVID_HQ_metadata.csv"))
        if self.continuous_target_frames or self.temporal_augmentation:
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
        self.scenes = self.scenes.reset_index(drop=True)
        self.fixed_clip_index = []
        if self.fixed_clips_per_scene > 0:
            clip_length = int(self.num_views) if (self.continuous_target_frames or self.temporal_augmentation) else None
            if clip_length is None:
                raise ValueError("fixed_clips_per_scene currently requires continuous_target_frames=True or temporal_augmentation=True.")
            for scene_idx, scene_info in self.scenes.iterrows():
                starts = self._fixed_clip_starts(
                    int(scene_info["num frames"]),
                    clip_length,
                    self.fixed_clips_per_scene,
                )
                for clip_id, start in enumerate(starts.tolist()):
                    self.fixed_clip_index.append(
                        {
                            "scene_idx": int(scene_idx),
                            "clip_id": int(clip_id),
                            "clip_start": int(start),
                        }
                    )
            if len(self.fixed_clip_index) == 0:
                raise ValueError("SpatialVID fixed clip table is empty.")

    def __len__(self):
        if self.fixed_clips_per_scene > 0:
            return len(self.fixed_clip_index) * self.variants_per_scene
        return len(self.scenes) * self.variants_per_scene

    def _scene_variant_index(self, idx):
        idx = int(idx)
        if self.fixed_clips_per_scene > 0:
            base_idx = idx // self.variants_per_scene
            variant_id = idx % self.variants_per_scene
            clip_info = self.fixed_clip_index[base_idx]
            return (
                int(clip_info["scene_idx"]),
                int(variant_id),
                int(clip_info["clip_id"]),
                int(clip_info["clip_start"]),
            )
        if self.variants_per_scene <= 1:
            return idx, 0, None, None
        return idx // self.variants_per_scene, idx % self.variants_per_scene, None, None

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

    def _profile_for_variant(self, variant_id):
        if self.temporal_variant_profile_schedule is None:
            return self.temporal_trajectory_profile
        return self.temporal_variant_profile_schedule[int(variant_id) % len(self.temporal_variant_profile_schedule)]

    def _trajectory_seed(self, scene_id, clip_start, variant_id, profile, rng):
        if self.temporal_variant_profile_schedule is None and self.trajectories_per_clip is None:
            return int(rng.integers(0, np.iinfo(np.uint32).max))
        payload = "|".join(
            [
                str(self.seed),
                str(scene_id),
                str(clip_start),
                str(variant_id),
                str(profile),
                str(self.num_views),
                str(self.temporal_max_condition_frames),
            ]
        )
        return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)

    def _target_trajectory_indices(self, fps, rng, num_context_views, scene_id=None, clip_start=None, variant_id=0):
        profile = self._profile_for_variant(variant_id)
        py_seed = self._trajectory_seed(scene_id, clip_start, variant_id, profile, rng)
        py_rng = random.Random(py_seed)
        trajectory = sample_training_trajectory(
            num_frames=self.num_views,
            rng=py_rng,
            max_condition_frames=min(
                self.temporal_max_condition_frames,
                max(1, int(num_context_views)),
                int(self.num_views),
            ),
            fps=fps,
            profile=profile,
        )
        offsets = np.rint(np.asarray(trajectory.temporal_coords, dtype=np.float32) * float(fps)).astype(np.int64)
        offsets = np.clip(offsets, 0, max(int(self.num_views) - 1, 0))
        return offsets.astype(np.int64), trajectory

    @staticmethod
    def camera_cache_path_for(cache_dir, scene_id, clip_start, num_views, height, width):
        scene_name = str(scene_id).replace("/", "_")
        filename = f"clip_start{int(clip_start):08d}_frames{int(num_views):03d}_{int(height)}x{int(width)}.npz"
        return str(Path(cache_dir) / scene_name / filename)

    def _camera_cache_path(self, scene_info, clip_start):
        if self.camera_cache_dir is None or clip_start is None:
            return None
        return self.camera_cache_path_for(
            self.camera_cache_dir,
            scene_info["id"],
            int(clip_start),
            self.num_views,
            self.height,
            self.width,
        )

    def _load_camera_cache(self, scene_info, clip_start):
        path = self._camera_cache_path(scene_info, clip_start)
        if path is None:
            return None, None, None
        if not osp.exists(path):
            if self.camera_cache_required:
                raise FileNotFoundError(f"Missing SpatialVID camera cache: {path}")
            return None, None, path
        data = np.load(path)
        poses = data["camera_poses"].astype(np.float32)
        intrs = data["camera_intrs"].astype(np.float32)
        if poses.shape != (self.num_views, 4, 4):
            raise ValueError(f"Invalid camera_poses shape in {path}: {poses.shape}")
        if intrs.shape != (self.num_views, 3, 3):
            raise ValueError(f"Invalid camera_intrs shape in {path}: {intrs.shape}")
        return poses, intrs, path

    def _get_views(self, idx, rng, num_context_views):
        scene_idx, variant_id, fixed_clip_id, fixed_clip_start = self._scene_variant_index(idx)
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
            target_trajectory_indices = np.arange(self.num_views, dtype=np.int64)
            temporal_trajectory = None
            if self.temporal_augmentation:
                target_trajectory_indices, temporal_trajectory = self._target_trajectory_indices(
                    fps,
                    rng,
                    num_context_views,
                    scene_id=scene_info["id"],
                    clip_start=fixed_clip_start,
                    variant_id=variant_id,
                )
                clip_length = int(self.num_views)
                start = int(fixed_clip_start) if fixed_clip_start is not None else rng.integers(0, max(video_length - clip_length, 0) + 1)
                sample_index = np.arange(start, start + clip_length, dtype=np.int64)
                if has_camera_annotations:
                    video_lookup = np.linspace(0, video_length - 1, annotation_length)
                    sample_anno_index = np.abs(video_lookup[None, :] - sample_index[:, None]).argmin(axis=1)
                else:
                    sample_anno_index = None
            elif self.continuous_target_frames:
                clip_length = int(self.num_views)
                start = int(fixed_clip_start) if fixed_clip_start is not None else rng.integers(0, max(video_length - clip_length, 0) + 1)
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

        cached_poses, cached_intrs, camera_cache_path = self._load_camera_cache(scene_info, fixed_clip_start)
        has_cached_cameras = cached_poses is not None and cached_intrs is not None

        with open(osp.join(annotation_dir, "caption.json"), 'r') as f:
            captions = json.load(f)
            text_prompt = captions["SceneDescription"]

        context_local_indices_array = self._context_local_indices(num_context_views)
        context_local_indices = set(context_local_indices_array.tolist())
        context_strategy = getattr(self, "_last_context_strategy", self.context_sampling_strategy)
        first_sample_index = sample_index[0]
        first_time_index = sample_anno_index[0] if has_camera_annotations else sample_index[0]
        context_views = []
        target_views = []
        for v, rgb_image in enumerate(images):
            time_index = sample_anno_index[v] if has_camera_annotations else sample_index[v]
            timestamp = self._timestamp(sample_index, time_index, v, first_sample_index, first_time_index, reverse, fps)
            trajectory_type = temporal_trajectory.trajectory_type if temporal_trajectory is not None else "none"
            trajectory_profile = (
                self._profile_for_variant(variant_id)
                if self.temporal_augmentation
                else self.temporal_trajectory_profile
            )
            original_height, original_width = rgb_image.shape[:2]
            camera_pose = None
            camera_intr = None
            if has_cached_cameras:
                camera_pose = cached_poses[v]
                camera_intr = cached_intrs[v]
            elif has_camera_annotations:
                anno_index = sample_anno_index[v]
                camera_pose = _pose_vec_xyzw_to_c2w(poses[anno_index])
                camera_intr = _intrinsics_vec_to_matrix(intrinsics[anno_index], original_width, original_height)
            rgb_image, _, resized_camera_intr = self._crop_resize_if_necessary(
                rgb_image,
                (self.width, self.height),
                rng=rng,
                info=(idx, v),
                intrinsics=camera_intr if has_camera_annotations and not has_cached_cameras else None,
            )
            if has_camera_annotations and not has_cached_cameras:
                camera_intr = resized_camera_intr
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
                clip_id=int(fixed_clip_id) if fixed_clip_id is not None else int(variant_id),
                clip_start=int(sample_index[0]),
                frame_index=int(sample_index[v]),
                context_strategy=context_strategy,
                trajectory_index=int(v),
                target_trajectory_index=int(target_trajectory_indices[v]),
                temporal_augmentation=self.temporal_augmentation,
                temporal_order=self.temporal_order,
                temporal_trajectory_profile=trajectory_profile,
                temporal_trajectory_type=trajectory_type,
                allow_nonmonotonic_timestamps=False,
            )
            if camera_cache_path is not None:
                view["camera_cache_path"] = camera_cache_path
            if has_cached_cameras or has_camera_annotations:
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
