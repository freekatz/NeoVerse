from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


WAN_LATENT_TEMPORAL_STRIDE = 4
MAX_CONDITION_FRAMES = 8

TRAJECTORY_PROFILE_MIXED_FZR = "mixed_fzr"
TRAJECTORY_PROFILE_FORWARD_PAUSE = "forward_pause"
TRAJECTORY_PROFILE_BACKWARD = "backward"
DEFAULT_TRAJECTORY_PROFILE = TRAJECTORY_PROFILE_FORWARD_PAUSE
TRAJECTORY_PROFILES = (
    TRAJECTORY_PROFILE_FORWARD_PAUSE,
    TRAJECTORY_PROFILE_MIXED_FZR,
    TRAJECTORY_PROFILE_BACKWARD,
)
TRAJECTORY_PROFILE_ALIASES = {
    "natural_v2": TRAJECTORY_PROFILE_FORWARD_PAUSE,
    "natural": TRAJECTORY_PROFILE_FORWARD_PAUSE,
    "legacy": TRAJECTORY_PROFILE_MIXED_FZR,
    "mixed": TRAJECTORY_PROFILE_MIXED_FZR,
    "reverse": TRAJECTORY_PROFILE_BACKWARD,
}
TRAJECTORY_PROFILE_CHOICES = TRAJECTORY_PROFILES + tuple(TRAJECTORY_PROFILE_ALIASES)


def pixel_frame_to_latent_index(
    frame_idx: int,
    stride: int = WAN_LATENT_TEMPORAL_STRIDE,
) -> int:
    frame_idx = int(frame_idx)
    stride = max(int(stride), 1)
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got {frame_idx}")
    return (frame_idx + stride - 1) // stride


@dataclass
class Unit:
    start_frame: int
    length: int
    unit_type: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    camera: object | None = None

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.length

    def generate_coords(self, step: float) -> list[float]:
        if self.length == 0:
            return []
        if self.length == 1:
            return [self.t_start]

        if self.unit_type == "F":
            delta = self.t_end - self.t_start
            return [self.t_start + delta * i / (self.length - 1) for i in range(self.length)]
        if self.unit_type == "R":
            delta = self.t_start - self.t_end
            return [self.t_start - delta * i / (self.length - 1) for i in range(self.length)]
        return [self.t_start] * self.length


@dataclass
class Trajectory:
    units: list[Unit] = field(default_factory=list)
    backward: bool = False
    num_frames: int = 81

    @property
    def condition_frame_indices(self) -> list[int]:
        return [unit.start_frame for unit in self.units]

    @property
    def condition_latent_indices(self) -> list[int]:
        return sorted(set(pixel_frame_to_latent_index(idx) for idx in self.condition_frame_indices))

    @property
    def type_string(self) -> str:
        parts = [f"{unit.unit_type}:{unit.length}" for unit in self.units]
        direction = "backward" if self.backward else "forward"
        return ",".join(parts) + "_" + direction

    def generate_coords(self, fps: float = 24.0) -> list[float]:
        step = 1.0 / max(float(fps), 1e-8)
        coords: list[float] = []
        for unit in self.units:
            coords.extend(unit.generate_coords(step))

        max_time = (self.num_frames - 1) / max(float(fps), 1e-8)
        coords = [round(max(0.0, min(max_time, coord)), 8) for coord in coords]
        if self.backward:
            coords = [round(max_time - coord, 8) for coord in coords]
        return coords

    def to_result(self, fps: float = 24.0) -> "TrajectoryResult":
        return TrajectoryResult(
            temporal_coords=self.generate_coords(fps=fps),
            condition_frame_indices=self.condition_frame_indices,
            condition_latent_indices=self.condition_latent_indices,
            trajectory_type=self.type_string,
            fps=float(fps),
        )


@dataclass
class TrajectoryResult:
    temporal_coords: list[float]
    condition_frame_indices: list[int]
    condition_latent_indices: list[int]
    trajectory_type: str = "unknown"
    fps: float = 24.0


def create_units_from_positions(positions: list[int], num_frames: int) -> list[Unit]:
    units: list[Unit] = []
    for idx, position in enumerate(positions):
        if idx < len(positions) - 1:
            length = positions[idx + 1] - position
        else:
            length = num_frames - position
        units.append(Unit(start_frame=position, length=length))
    return units


def normalize_trajectory_profile(profile: str) -> str:
    profile = str(profile or DEFAULT_TRAJECTORY_PROFILE).strip().lower()
    profile = TRAJECTORY_PROFILE_ALIASES.get(profile, profile)
    if profile not in TRAJECTORY_PROFILES:
        raise ValueError(
            f"Unknown trajectory profile {profile!r}. "
            f"Choose from: {', '.join(TRAJECTORY_PROFILE_CHOICES)}"
        )
    return profile


def _weighted_choice(rng: random.Random, weighted_values: list[tuple[str, float]]) -> str:
    values = [value for value, _ in weighted_values]
    weights = [weight for _, weight in weighted_values]
    return rng.choices(values, weights=weights, k=1)[0]


def _sample_condition_count(max_k: int, rng: random.Random, profile: str) -> int:
    if profile == TRAJECTORY_PROFILE_MIXED_FZR:
        choices = list(range(1, max_k + 1))
        weights = [1.0 / (2**i) for i in range(max_k)]
        return rng.choices(choices, weights=weights, k=1)[0]

    min_k = 1 if max_k == 1 else 2
    choices = list(range(min_k, max_k + 1))
    base_weights = {1: 0.15, 2: 1.0, 3: 0.75, 4: 0.4}
    weights = [base_weights.get(k, 0.4 / (2 ** max(0, k - 4))) for k in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def _sample_backward_flag(rng: random.Random, profile: str) -> bool:
    if profile == TRAJECTORY_PROFILE_MIXED_FZR:
        return rng.random() < 0.5
    return False


def assign_unit_types(units: list[Unit], rng: random.Random, num_frames: int, fps: float = 24.0) -> None:
    t = 0.0
    step = 1.0 / max(float(fps), 1e-8)
    max_t = (num_frames - 1) / max(float(fps), 1e-8)

    for unit in units:
        candidates = ["Z"]
        if t < max_t - 1e-9:
            candidates.append("F")
        if t > 1e-9:
            candidates.append("R")

        unit.unit_type = rng.choice(candidates)
        unit.t_start = t

        if unit.unit_type == "F":
            delta = min(unit.length * step, max_t - t)
            unit.t_end = t + delta
            t = unit.t_end
        elif unit.unit_type == "R":
            delta = min(unit.length * step, t)
            unit.t_end = t - delta
            t = unit.t_end
        else:
            unit.t_end = t


def assign_unit_types_forward_pause(
    units: list[Unit],
    rng: random.Random,
    num_frames: int,
    fps: float = 24.0,
) -> None:
    t = 0.0
    step = 1.0 / max(float(fps), 1e-8)
    max_t = (num_frames - 1) / max(float(fps), 1e-8)
    has_forward_unit = False

    for unit in units:
        at_start = t <= 1e-9
        at_end = t >= max_t - 1e-9
        if at_start:
            weighted_candidates = [("F", 0.9), ("Z", 0.1)]
        elif at_end:
            weighted_candidates = [("Z", 1.0)]
        else:
            weighted_candidates = [("F", 0.8), ("Z", 0.2)]

        unit.unit_type = _weighted_choice(rng, weighted_candidates)
        has_forward_unit = has_forward_unit or unit.unit_type == "F"
        unit.t_start = t

        if unit.unit_type == "F":
            delta = min(unit.length * step, max_t - t)
            unit.t_end = t + delta
            t = unit.t_end
        else:
            unit.t_end = t

    if not has_forward_unit and units:
        units[0].unit_type = "F"
        compute_unit_coords(units, num_frames, fps=fps)


def compute_unit_coords(units: list[Unit], num_frames: int, fps: float = 24.0) -> None:
    t = 0.0
    step = 1.0 / max(float(fps), 1e-8)
    max_t = (num_frames - 1) / max(float(fps), 1e-8)

    for idx, unit in enumerate(units):
        if idx == 0 and unit.unit_type == "R" and t <= 1e-9:
            t = max_t
        unit.t_start = t
        if unit.unit_type == "F":
            delta = min(unit.length * step, max_t - t)
            unit.t_end = t + delta
            t = unit.t_end
        elif unit.unit_type == "R":
            delta = min(unit.length * step, t)
            unit.t_end = t - delta
            t = unit.t_end
        else:
            unit.t_end = t


def sample_training_trajectory(
    num_frames: int = 81,
    rng: random.Random | None = None,
    max_condition_frames: int = MAX_CONDITION_FRAMES,
    fps: float = 24.0,
    profile: str = DEFAULT_TRAJECTORY_PROFILE,
) -> TrajectoryResult:
    if rng is None:
        rng = random.Random()
    profile = normalize_trajectory_profile(profile)
    if profile == TRAJECTORY_PROFILE_BACKWARD:
        return build_inference_trajectory(
            num_frames=int(num_frames),
            units=f"F:{int(num_frames)}",
            backward=True,
            condition_unit_indices=[0],
            fps=fps,
        )

    max_k = min(max(1, int(max_condition_frames)), int(num_frames))
    k = _sample_condition_count(max_k, rng, profile=profile)

    if k == 1:
        positions = [0]
    else:
        positions = [0] + sorted(rng.sample(range(1, int(num_frames)), k - 1))

    units = create_units_from_positions(positions, int(num_frames))
    backward = _sample_backward_flag(rng, profile=profile)
    if profile == TRAJECTORY_PROFILE_MIXED_FZR:
        assign_unit_types(units, rng, int(num_frames), fps=fps)
    else:
        assign_unit_types_forward_pause(units, rng, int(num_frames), fps=fps)
    return Trajectory(units=units, backward=backward, num_frames=int(num_frames)).to_result(fps=fps)


assign_unit_types_natural_v2 = assign_unit_types_forward_pause


def build_inference_trajectory(
    num_frames: int = 81,
    units: str = "F",
    backward: bool = False,
    condition_unit_indices: list[int] | None = None,
    fps: float = 24.0,
) -> TrajectoryResult:
    units_str = units.strip()
    if units_str.startswith("["):
        coords = [float(coord) for coord in json.loads(units_str)]
        if len(coords) != num_frames:
            raise ValueError(f"JSON length {len(coords)} != num_frames {num_frames}")
        max_time = (num_frames - 1) / max(float(fps), 1e-8)
        if backward:
            coords = [round(max_time - coord, 8) for coord in coords]
        return TrajectoryResult(
            temporal_coords=coords,
            condition_frame_indices=[0],
            condition_latent_indices=[0],
            trajectory_type="explicit",
            fps=float(fps),
        )

    unit_specs: list[tuple[str, int]] = []
    for part in units_str.split(","):
        part = part.strip()
        if ":" in part:
            unit_type, length_str = part.split(":", 1)
            unit_type = unit_type.strip().upper()
            length = int(length_str.strip())
        else:
            unit_type = part.upper()
            length = int(num_frames)
        if unit_type not in {"F", "R", "Z"}:
            raise ValueError(f"Unknown unit type {unit_type!r}. Valid values: F, R, Z")
        unit_specs.append((unit_type, length))

    total = sum(length for _, length in unit_specs)
    if total != num_frames:
        raise ValueError(f"Unit lengths sum to {total}, expected {num_frames}")

    unit_list: list[Unit] = []
    start = 0
    for unit_type, length in unit_specs:
        unit_list.append(Unit(start_frame=start, length=length, unit_type=unit_type))
        start += length
    compute_unit_coords(unit_list, num_frames, fps=fps)

    if condition_unit_indices is None:
        condition_unit_indices = [0]
    for idx in condition_unit_indices:
        if idx < 0 or idx >= len(unit_list):
            raise ValueError(f"condition_unit_indices contains {idx}, but there are {len(unit_list)} units")

    trajectory = Trajectory(units=unit_list, backward=bool(backward), num_frames=int(num_frames))
    result = trajectory.to_result(fps=fps)
    result.condition_frame_indices = sorted(set(unit_list[idx].start_frame for idx in condition_unit_indices))
    result.condition_latent_indices = sorted(set(pixel_frame_to_latent_index(idx) for idx in result.condition_frame_indices))
    return result
