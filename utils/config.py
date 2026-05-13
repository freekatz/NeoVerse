from omegaconf import OmegaConf

from utils.data.spatialvid import SpatialVID

NULL_CONFIG_STRINGS = {"", "none", "None", "null", "Null"}


def is_null_config_value(value):
    return value is None or (isinstance(value, str) and value.strip() in NULL_CONFIG_STRINGS)


def resolved_config_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def config_value(cfg, key, default=None):
    return resolved_config_value(cfg.get(key, default))


def none_if_null(value):
    value = resolved_config_value(value)
    return None if is_null_config_value(value) else value


def config_bool(cfg, key, default=False):
    value = config_value(cfg, key, default)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def config_int(cfg, key, default):
    value = config_value(cfg, key, default)
    if is_null_config_value(value):
        value = default
    return int(value)


def config_optional_int(cfg, key, default=None):
    value = config_value(cfg, key, default)
    if is_null_config_value(value):
        return default
    return int(value)


def normalize_filter_values(value):
    value = resolved_config_value(value)
    if is_null_config_value(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if is_null_config_value(text):
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        items = [item.strip().strip("'\"") for item in text.split(",")]
        return [item for item in items if item] or None
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            normalized = normalize_filter_values(item)
            if normalized is None:
                continue
            if isinstance(normalized, list):
                items.extend(normalized)
            else:
                items.append(normalized)
        return items or None
    return [str(value)]


def build_spatialvid_dataset(cfg):
    return SpatialVID(
        split=None,
        ROOT=str(config_value(cfg, "data_root", "data/SpatialVID")),
        video_ids=normalize_filter_values(config_value(cfg, "video_ids", None)),
        video_paths=normalize_filter_values(config_value(cfg, "video_paths", None)),
        use_camera_annotations=config_bool(cfg, "use_camera_annotations", False),
        continuous_target_frames=config_bool(cfg, "continuous_target_frames", True),
        force_first_context=config_bool(cfg, "force_first_context", True),
        timestamp_unit=str(config_value(cfg, "timestamp_unit", "seconds")),
        temporal_augmentation=config_bool(cfg, "temporal_augmentation", False),
        temporal_trajectory_profile=str(config_value(cfg, "temporal_trajectory_profile", "forward_pause")),
        temporal_order=str(config_value(cfg, "temporal_order", "trajectory")),
        temporal_max_condition_frames=config_int(cfg, "temporal_max_condition_frames", 8),
        context_sampling_strategy=str(config_value(cfg, "context_sampling_strategy", "uniform")),
        context_sampling_weights=config_value(cfg, "context_sampling_weights", None),
        variants_per_scene=config_int(cfg, "variants_per_scene", 1),
        trajectories_per_clip=config_optional_int(cfg, "trajectories_per_clip", None),
        temporal_variant_profile_weights=none_if_null(config_value(cfg, "temporal_variant_profile_weights", None)),
        fixed_clips_per_scene=config_int(cfg, "fixed_clips_per_scene", 0),
        camera_cache_dir=none_if_null(config_value(cfg, "camera_cache_dir", None)),
        camera_cache_required=config_bool(cfg, "camera_cache_required", False),
        min_interval=1,
        max_interval=1,
        height=config_int(cfg, "height", 336),
        width=config_int(cfg, "width", 560),
        num_views=config_int(cfg, "num_views", 81),
        min_num_context_views=config_int(cfg, "min_num_context_views", 10),
        max_num_context_views=config_int(cfg, "max_num_context_views", 20),
        seed=config_int(cfg, "dataset_seed", config_int(cfg, "seed", 0)),
    )
