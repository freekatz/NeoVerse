import argparse
import copy
import gc
import io
import json
import os
import sys
import threading
import time
import imageio
import torch
import numpy as np
import uvicorn

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

for proxy_env in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(proxy_env, None)

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, homo_matrix_inverse
from diffsynth.utils.app import extract_point_cloud, build_scene_glb
from diffsynth.auxiliary_models.worldmirror.models.utils.rotation import quat_to_rotmat, rotmat_to_quat

parser = argparse.ArgumentParser()
parser.add_argument("--reconstructor_path", type=str,
                    default="models/NeoVerse/reconstructor.ckpt",
                    help="Path to reconstructor checkpoint")
parser.add_argument("--low_vram", action="store_true",
                    help="Enable low-VRAM mode with model offloading")
parser.add_argument("--device", type=str, default=None,
                    help="Torch device to use, for example cuda, cuda:1, or cpu")
parser.add_argument("--server_name", type=str, default="0.0.0.0",
                    help="Host/IP for the Gradio server to bind to")
parser.add_argument("--server_port", type=int, default=7860,
                    help="Port for the Gradio server")
parser.add_argument("--share", action="store_true",
                    help="Enable Gradio share link (requires frpc download)")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Global model
# ---------------------------------------------------------------------------
OUTPUT_ROOT = "outputs/gradio"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
WORKFLOW_ROOT = os.path.join(OUTPUT_ROOT, "workflow")
os.makedirs(WORKFLOW_ROOT, exist_ok=True)
GLB_PATH = os.path.join(OUTPUT_ROOT, "scene.glb")
PREVIEW_PATH = os.path.join(OUTPUT_ROOT, "preview.mp4")
MASK_PATH = os.path.join(OUTPUT_ROOT, "mask.mp4")
OUTPUT_PATH = os.path.join(OUTPUT_ROOT, "output.mp4")
COMPARE_PATH = os.path.join(OUTPUT_ROOT, "render_vs_generated.mp4")
JSON_PATH = os.path.join(OUTPUT_ROOT, "trajectory.json")
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
if str(device).startswith("cuda:") and torch.cuda.is_available():
    torch.cuda.set_device(torch.device(device))
VIEWER_LOCK = threading.RLock()
LATEST_SCENE = {
    "state": None,
    "meta": None,
    "source": None,
    "updated_at": None,
}
SCENE_REGISTRY = {
    "records": {},
    "order": [],
    "active_id": None,
}
PIPELINE_MODEL_NAMES = ("text_encoder", "dit", "vae", "control_branch", "reconstructor")
RENDER_TARGET_KEYS = (
    "target_rgb",
    "target_depth",
    "target_mask",
    "target_poses",
    "target_intrs",
    "target_timestamps",
)
SCENE_TENSOR_KEYS = (
    "input_intrs",
    "input_cam2world",
    "input_timestamps",
)
WORKFLOW_START_POLICIES = {"previous_end", "initial", "reconstruction_first"}
print(f"Loading NeoVerse pipeline (reconstructor: {args.reconstructor_path})...")
pipe = WanVideoNeoVersePipeline.from_pretrained(
    local_model_path="models",
    reconstructor_path=args.reconstructor_path,
    lora_path="models/NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    lora_alpha=1.0,
    device=device,
    torch_dtype=torch.bfloat16,
    enable_vram_management=args.low_vram,
)
print("Pipeline loaded.")


def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _module_current_device(module):
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return torch.device("cpu")


def _set_model_phase(active_model_names=()):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    active = set(active_model_names or ())
    if pipe.vram_management_enabled:
        pipe.load_models_to_device(list(active))
        _cuda_cleanup()
        return

    for name in PIPELINE_MODEL_NAMES:
        model = getattr(pipe, name, None)
        if model is None:
            continue
        target_device = torch.device(device if name in active else "cpu")
        if _module_current_device(model) != target_device:
            model.to(target_device)
    _cuda_cleanup()


def _move_tensor_to_device(value, target_device):
    if torch.is_tensor(value):
        return value.to(device=target_device)
    return value


def _move_gaussian_to_device(gaussian, target_device):
    for attr_name, attr_value in list(gaussian.__dict__.items()):
        if torch.is_tensor(attr_value):
            setattr(gaussian, attr_name, attr_value.to(device=target_device))
    return gaussian


def _move_state_gaussians_to_device(state, target_device):
    if state is None or "gaussians" not in state:
        return state
    for batch in state["gaussians"]:
        for gaussian in batch:
            _move_gaussian_to_device(gaussian, target_device)
    return state


def _offload_registered_scene_gaussians_to_cpu():
    with VIEWER_LOCK:
        states = [record["state"] for record in SCENE_REGISTRY["records"].values()]
    seen = set()
    for scene_state in states:
        if scene_state is None:
            continue
        state_id = id(scene_state)
        if state_id in seen:
            continue
        seen.add(state_id)
        _move_state_gaussians_to_device(scene_state, "cpu")
    _cuda_cleanup()


def _move_state_tensors_to_device(state, target_device, keys):
    if state is None:
        return state
    for key in keys:
        if key in state:
            state[key] = _move_tensor_to_device(state[key], target_device)
    return state


def _ensure_scene_state_on_device(state):
    _move_state_tensors_to_device(state, device, SCENE_TENSOR_KEYS)
    _move_state_gaussians_to_device(state, device)
    return state


def _ensure_render_targets_on_device(state):
    _move_state_tensors_to_device(state, device, RENDER_TARGET_KEYS)
    return state


def _clear_transient_render_state(state):
    if state is None:
        return state
    for key in RENDER_TARGET_KEYS:
        state.pop(key, None)
    return state


def _raise_cuda_oom(stage, exc):
    try:
        _set_model_phase(())
    finally:
        _cuda_cleanup()
    raise gr.Error(
        f"CUDA out of memory while {stage}. The workflow cleared cached tensors; "
        "try fewer iterations, fewer frames, or restart with --low_vram if this repeats."
    ) from exc


def _export_scene_to(scene, output_path):
    """Export a trimesh.Scene and return the path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(file_obj=output_path)
    return output_path


def _export_scene(scene):
    """Export a trimesh.Scene to the fixed GLB path and return it."""
    return _export_scene_to(scene, GLB_PATH)


def _safe_normalize(vec, eps=1e-6):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec * 0.0
    return vec / norm


def _rotation_matrix_from_axis_angle(axis, angle_deg):
    axis = _safe_normalize(np.asarray(axis, dtype=np.float32))
    angle = np.deg2rad(angle_deg)
    x, y, z = axis
    cross = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float32)
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _build_orbit_camera_pose(center, yaw_deg, pitch_deg, roll_deg, radius):
    center = np.asarray(center, dtype=np.float32)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    radius = max(float(radius), 1e-3)

    offset = np.array([
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw) * np.cos(pitch),
    ], dtype=np.float32) * radius
    camera_pos = center + offset

    forward = _safe_normalize(center - camera_pos)
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(forward, up_world)) > 0.98:
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = _safe_normalize(np.cross(up_world, forward))
    up = _safe_normalize(np.cross(forward, right))

    if abs(float(roll_deg)) > 1e-4:
        roll_rot = _rotation_matrix_from_axis_angle(forward, roll_deg)
        right = _safe_normalize(roll_rot @ right)
        up = _safe_normalize(roll_rot @ up)

    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = np.stack([right, up, forward], axis=1)
    cam2world[:3, 3] = camera_pos
    return cam2world


def _to_rgb_image(frame):
    image = (frame.detach().clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().numpy()
    return Image.fromarray(image)


def _to_scalar_image(frame, scale=None):
    frame = frame.detach().float().cpu()
    if scale is None:
        valid = frame[frame > 0]
        scale = float(valid.max().item()) if valid.numel() > 0 else 1.0
    scale = max(scale, 1e-6)
    image = (frame.clamp(0, scale) * (255.0 / scale)).to(dtype=torch.uint8).numpy()
    return Image.fromarray(image)


def _frame_to_rgb_pil(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if torch.is_tensor(frame):
        tensor = frame.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            array = tensor.float().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        elif tensor.ndim == 3 and tensor.shape[0] == 3:
            array = tensor.permute(1, 2, 0).float().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        elif tensor.ndim == 2:
            gray = tensor.float().clamp(0, 1).mul(255).to(torch.uint8).numpy()
            array = np.stack([gray] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported tensor frame shape: {tuple(tensor.shape)}")
        return Image.fromarray(array).convert("RGB")

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255 if array.max() > 1.0 else 1.0)
        if array.max() <= 1.0:
            array = (array * 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _target_rgb_to_pil_frames(target_rgb):
    return [_frame_to_rgb_pil(target_rgb[0, frame_idx]) for frame_idx in range(target_rgb.shape[1])]


def _build_side_by_side_frames(left_frames, right_frames, gap=8, background=(10, 12, 16)):
    if not left_frames or not right_frames:
        raise ValueError("Both left and right frame sequences are required for comparison video.")

    num_frames = max(len(left_frames), len(right_frames))
    output_frames = []
    for frame_idx in range(num_frames):
        left = _frame_to_rgb_pil(left_frames[min(frame_idx, len(left_frames) - 1)])
        right = _frame_to_rgb_pil(right_frames[min(frame_idx, len(right_frames) - 1)])
        frame_height = max(left.height, right.height)
        frame_width = left.width + gap + right.width
        canvas = Image.new("RGB", (frame_width, frame_height), color=background)
        canvas.paste(left, (0, (frame_height - left.height) // 2))
        canvas.paste(right, (left.width + gap, (frame_height - right.height) // 2))
        output_frames.append(canvas)
    return output_frames


def _concat_video_files(video_paths, output_path, fps=16):
    paths = [path for path in video_paths if path and os.path.exists(path)]
    if not paths:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=9)
    reference_size = None
    try:
        for path in paths:
            reader = imageio.get_reader(path)
            try:
                for frame in reader:
                    frame = np.asarray(frame)
                    if frame.ndim == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    if frame.ndim == 3 and frame.shape[-1] == 4:
                        frame = frame[..., :3]
                    if reference_size is None:
                        reference_size = (frame.shape[1], frame.shape[0])
                    elif (frame.shape[1], frame.shape[0]) != reference_size:
                        frame = np.asarray(Image.fromarray(frame).resize(reference_size, Image.BICUBIC))
                    writer.append_data(frame)
            finally:
                reader.close()
    finally:
        writer.close()
    return output_path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _build_source_views(pil_images, scene_type):
    static_flag = scene_type == "Static scene"
    frame_num = len(pil_images)
    views = {
        "img": torch.stack([F.to_tensor(img)[None] for img in pil_images], dim=1).to(device),
        "is_target": torch.zeros((1, frame_num), dtype=torch.bool, device=device),
    }
    if static_flag:
        views["is_static"] = torch.ones((1, frame_num), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, frame_num), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, frame_num), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, frame_num, dtype=torch.int64, device=device).unsqueeze(0)
    return views


def _load_media_from_paths(paths, scene_type):
    if not paths:
        raise ValueError("Please provide at least one local video or image path.")

    static = scene_type == "Static scene"
    video_path = None
    image_paths = []
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"Input path does not exist: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTS:
            video_path = path
            break
        image_paths.append(path)

    if video_path is not None:
        pil_images = load_video(
            video_path,
            81,
            resolution=(560, 336),
            resize_mode="center_crop",
            static_scene=static,
        )
    elif image_paths:
        pil_images = load_video(
            image_paths,
            81,
            resolution=(560, 336),
            resize_mode="center_crop",
            static_scene=static,
        )
    else:
        raise ValueError("No supported video or image inputs were found.")
    return {"images": pil_images, "scene_type": scene_type}


def _camera_pose_to_orbit(center, cam2world):
    camera_pos = np.asarray(cam2world[:3, 3], dtype=np.float32)
    offset = camera_pos - np.asarray(center, dtype=np.float32)
    radius = float(np.linalg.norm(offset))
    if radius < 1e-6:
        return 0.0, -10.0, 1.5
    yaw = float(np.rad2deg(np.arctan2(offset[0], offset[2])))
    pitch = float(np.rad2deg(np.arcsin(np.clip(offset[1] / radius, -1.0, 1.0))))
    return yaw, pitch, radius


def _compute_scene_meta(state):
    timestamps = state["input_timestamps"].detach().cpu().float().numpy()
    input_cam2world = state["input_cam2world"].detach().cpu().numpy()
    scene_center = np.asarray(state["scene_center"], dtype=np.float32)
    camera_positions = input_cam2world[:, :3, 3]
    camera_distances = np.linalg.norm(camera_positions - scene_center[None], axis=1)
    if state["points"].shape[0] > 0:
        point_distances = np.linalg.norm(state["points"] - scene_center[None], axis=1)
        scene_scale = float(np.quantile(point_distances, 0.9))
    else:
        scene_scale = float(np.quantile(camera_distances, 0.9)) if len(camera_distances) > 0 else 1.0
    scene_scale = max(scene_scale, 0.25)

    default_yaw, default_pitch, default_radius = _camera_pose_to_orbit(scene_center, input_cam2world[0])
    default_radius = max(default_radius, scene_scale * 1.2, 0.5)

    return {
        "width": int(state["width"]),
        "height": int(state["height"]),
        "scene_center": [float(v) for v in scene_center.tolist()],
        "scene_scale": float(scene_scale),
        "time_min": float(timestamps.min()) if len(timestamps) > 0 else 0.0,
        "time_max": float(timestamps.max()) if len(timestamps) > 0 else 0.0,
        "num_frames": int(len(timestamps)),
        "static_scene": bool(state.get("scene_type", "General scene") == "Static scene"),
        "default_yaw": float(default_yaw),
        "default_pitch": float(default_pitch),
        "default_roll": 0.0,
        "default_radius": float(default_radius),
        "min_radius": float(max(default_radius * 0.12, scene_scale * 0.05, 0.1)),
        "max_radius": float(max(default_radius * 3.5, scene_scale * 4.0, 2.0)),
        "default_focal_scale": 1.0,
        "source_label": state.get("source_label", "local input"),
    }


def _store_latest_scene(state, source):
    meta = _compute_scene_meta(state)
    state["viewer_meta"] = meta
    with VIEWER_LOCK:
        LATEST_SCENE["state"] = state
        LATEST_SCENE["meta"] = meta
        LATEST_SCENE["source"] = source
        LATEST_SCENE["updated_at"] = time.time()
        record = _register_scene_unlocked(
            scene_id="latest",
            label=f"Latest: {meta.get('source_label', source)}",
            state=state,
            meta=meta,
            source=source,
            make_active=True,
        )
        LATEST_SCENE["meta"] = record["meta"]
    return record["meta"]


def _public_scene_record(record):
    meta = dict(record["meta"] or {})
    meta["scene_id"] = record["scene_id"]
    meta["scene_label"] = record["label"]
    return {
        "scene_id": record["scene_id"],
        "label": record["label"],
        "source": record.get("source"),
        "run_id": record.get("run_id"),
        "iteration": record.get("iteration"),
        "generated_video": record.get("generated_video"),
        "comparison_video": record.get("comparison_video"),
        "glb_path": record.get("glb_path"),
        "updated_at": record.get("updated_at"),
        "meta": meta,
    }


def _scene_choices_from_records(records):
    return [(record["label"], record["scene_id"]) for record in records]


def _ordered_scene_records_unlocked(run_id=None):
    records = []
    for scene_id in SCENE_REGISTRY["order"]:
        record = SCENE_REGISTRY["records"].get(scene_id)
        if record is None:
            continue
        if run_id is not None and record.get("run_id") != run_id:
            continue
        records.append(record)
    return records


def _register_scene_unlocked(
    scene_id,
    label,
    state,
    meta=None,
    source="workflow",
    glb_path=None,
    generated_video=None,
    comparison_video=None,
    run_id=None,
    iteration=None,
    make_active=False,
):
    if meta is None:
        meta = _compute_scene_meta(state)
    meta = dict(meta)
    meta["scene_id"] = scene_id
    meta["scene_label"] = label
    state["viewer_meta"] = meta

    record = {
        "scene_id": scene_id,
        "label": label,
        "state": state,
        "meta": meta,
        "source": source,
        "glb_path": glb_path,
        "generated_video": generated_video,
        "comparison_video": comparison_video,
        "run_id": run_id,
        "iteration": iteration,
        "updated_at": time.time(),
    }
    if scene_id not in SCENE_REGISTRY["records"]:
        SCENE_REGISTRY["order"].append(scene_id)
    SCENE_REGISTRY["records"][scene_id] = record
    if make_active:
        SCENE_REGISTRY["active_id"] = scene_id
    return record


def _register_scene(*args, **kwargs):
    with VIEWER_LOCK:
        return _register_scene_unlocked(*args, **kwargs)


def _get_scene_record_unlocked(scene_id=None):
    if scene_id:
        record = SCENE_REGISTRY["records"].get(scene_id)
        if record is None:
            raise KeyError(scene_id)
        return record
    active_id = SCENE_REGISTRY.get("active_id")
    if active_id and active_id in SCENE_REGISTRY["records"]:
        return SCENE_REGISTRY["records"][active_id]
    if LATEST_SCENE["state"] is not None:
        return {
            "scene_id": "latest",
            "label": "Latest scene",
            "state": LATEST_SCENE["state"],
            "meta": LATEST_SCENE["meta"],
            "source": LATEST_SCENE["source"],
            "glb_path": None,
            "generated_video": None,
            "comparison_video": None,
            "run_id": None,
            "iteration": None,
            "updated_at": LATEST_SCENE["updated_at"],
        }
    return None


@torch.no_grad()
def _render_view_image(
    state,
    viewer_mode="Orbit Viewer",
    time_value=0.0,
    yaw=0.0,
    pitch=-10.0,
    roll=0.0,
    radius=1.5,
    center_x=0.0,
    center_y=0.0,
    center_z=0.0,
    focal_scale=1.0,
    modality="rgb",
    mask_threshold=0.95,
    resolution_scale=1.0,
    include_scene_glb=False,
):
    if state is None or "gaussians" not in state:
        raise ValueError("Run reconstruction first.")

    _ensure_scene_state_on_device(state)
    base_h, base_w = state["height"], state["width"]
    resolution_scale = float(np.clip(resolution_scale, 0.25, 1.0))
    render_h = max(64, int(round(base_h * resolution_scale)))
    render_w = max(64, int(round(base_w * resolution_scale)))

    input_cam2world = state["input_cam2world"]
    input_intrs = state["input_intrs"]
    input_timestamps = state["input_timestamps"]
    static_flag = state.get("scene_type", "General scene") == "Static scene"

    timestamps_np = input_timestamps.detach().cpu().float().numpy()
    if len(timestamps_np) == 0:
        raise ValueError("No timestamps are available for rendering.")
    requested_time = 0.0 if static_flag else float(np.clip(time_value, timestamps_np.min(), timestamps_np.max()))
    frame_idx = 0 if static_flag else int(np.argmin(np.abs(timestamps_np - requested_time)))
    resolved_time = float(timestamps_np[0] if static_flag else requested_time)
    render_timestamp = torch.tensor(
        resolved_time,
        device=device,
        dtype=torch.float32,
    )

    intrs = input_intrs[0 if static_flag else frame_idx].clone().to(device=device)
    intrs[:2] *= resolution_scale
    intrs[0, 0] *= float(focal_scale)
    intrs[1, 1] *= float(focal_scale)

    viewer_mode_norm = viewer_mode.lower().replace("_", " ")
    if viewer_mode_norm in {"input camera", "input"}:
        cam2world = input_cam2world[0 if static_flag else frame_idx].to(device=device)
        extra_cam2worlds = None
        resolved_mode = "input"
    else:
        orbit_center = state["scene_center"] + np.array([center_x, center_y, center_z], dtype=np.float32)
        cam2world_np = _build_orbit_camera_pose(orbit_center, yaw, pitch, roll, radius)
        cam2world = torch.from_numpy(cam2world_np).to(device=device, dtype=input_cam2world.dtype)
        extra_cam2worlds = [cam2world_np]
        resolved_mode = "orbit"

    world2cam = homo_matrix_inverse(cam2world.unsqueeze(0))[0]
    render_rgb, render_depth, render_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        state["gaussians"],
        render_viewmats=[world2cam.unsqueeze(0)],
        render_Ks=[intrs.unsqueeze(0)],
        render_timestamps=[render_timestamp.unsqueeze(0)],
        sh_degree=0,
        width=render_w,
        height=render_h,
    )

    modality = modality.lower()
    if modality == "rgb":
        output_image = _to_rgb_image(render_rgb[0, 0])
    elif modality == "depth":
        output_image = _to_scalar_image(render_depth[0, 0, ..., 0])
    elif modality == "alpha":
        output_image = _to_scalar_image(render_alpha[0, 0, ..., 0], scale=1.0)
    elif modality == "mask":
        output_image = _to_scalar_image((render_alpha[0, 0, ..., 0] > float(mask_threshold)).float(), scale=1.0)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    glb_path = None
    if include_scene_glb:
        scene = build_scene_glb(
            state["points"],
            state["colors"],
            state["frame_indices"],
            input_cam2world.detach().cpu().numpy(),
            selected_idx=frame_idx,
            extra_cam2worlds=extra_cam2worlds,
            extra_camera_colors=[(255, 255, 255)] if extra_cam2worlds is not None else None,
        )
        glb_path = _export_scene(scene)

    camera_pos = cam2world[:3, 3].detach().cpu().numpy()
    meta = {
        "mode": resolved_mode,
        "frame_index": int(frame_idx),
        "timestamp": resolved_time,
        "requested_time": float(requested_time),
        "camera_position": [float(v) for v in camera_pos.tolist()],
        "render_width": int(render_w),
        "render_height": int(render_h),
        "modality": modality,
    }
    status = (
        f"mode={resolved_mode} | frame={meta['frame_index']} | "
        f"time={meta['timestamp']:.2f} | camera=({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}) | "
        f"render={render_w}x{render_h}"
    )
    return output_image, status, meta, glb_path


# ---------------------------------------------------------------------------
# 1. Upload handler
# ---------------------------------------------------------------------------
def _get_example_videos(config_path="examples/gallery.json"):
    """Scan directory for video/image files and return metadata list.

    If an ``examples.json`` exists in *directory*, it is used as the
    authoritative source (preserving order and per-example parameters).
    Files present on disk but absent from the JSON are appended with
    default parameters.
    """
    if not os.path.exists(config_path):
        return []
    _DEFAULTS = {
        "scene_type": "General scene",
        "camera_motion": "static",
        "angle": 0,
        "distance": 0,
        "orbit_radius": 0,
        "mode": "relative",
        "zoom_ratio": 1.0,
        "alpha_threshold": 1.0,
        "use_first_frame": True,
        "traj_file": None,
    }

    examples = []
    if os.path.exists(config_path):
        with open(config_path) as f:
            entries = json.load(f)
        for entry in entries:
            fpath = entry["file"]
            if not os.path.exists(fpath):
                continue
            ex = {**_DEFAULTS, **entry}
            examples.append(ex)
    return examples


def handle_upload(files, scene_type):
    """Load user media into a list of PIL images stored in gr.State."""
    if not files:
        return gr.update(), None, gr.update(interactive=False)
    try:
        state = _load_media_from_paths(list(files), scene_type)
    except ValueError:
        return gr.update(), None, gr.update(interactive=False)
    state["source_label"] = os.path.basename(files[0]) if files else "uploaded input"
    return state, state["images"], gr.update(interactive=True)


# ---------------------------------------------------------------------------
# 2. Reconstruction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_reconstruction_core(state, source="gradio", store_latest=True, export_glb=True):
    if state is None or "images" not in state:
        raise ValueError("Please upload a video or images first.")

    _set_model_phase(("reconstructor",))
    pil_images = state["images"]
    views = _build_source_views(pil_images, state.get("scene_type", "General scene"))

    try:
        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)
    finally:
        _set_model_phase(())

    gaussians = predictions["splats"]
    input_intrs = predictions["rendered_intrinsics"][0]
    input_cam2world = predictions["rendered_extrinsics"][0]
    input_timestamps = predictions["rendered_timestamps"][0]
    points, colors, frame_indices = extract_point_cloud(predictions)

    state.pop("source_views", None)
    state["gaussians"] = gaussians
    state["input_intrs"] = input_intrs
    state["input_cam2world"] = input_cam2world
    state["input_timestamps"] = input_timestamps
    state["points"] = points
    state["colors"] = colors
    state["frame_indices"] = frame_indices
    state["height"] = pil_images[0].size[1]
    state["width"] = pil_images[0].size[0]
    if points.shape[0] > 0:
        state["scene_center"] = np.median(points, axis=0).astype(np.float32)
    else:
        state["scene_center"] = input_cam2world[:, :3, 3].detach().cpu().numpy().mean(axis=0).astype(np.float32)

    glb_path = None
    if export_glb:
        scene = build_scene_glb(points, colors, frame_indices, input_cam2world.detach().cpu().numpy())
        glb_path = _export_scene(scene)
    meta = _store_latest_scene(state, source=source) if store_latest else _compute_scene_meta(state)
    return state, glb_path, meta


@torch.no_grad()
def reconstruct(state):
    """Run the reconstructor and return 3D scene."""
    try:
        state, glb_path, _ = _run_reconstruction_core(state, source="gradio")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return state, glb_path, gr.update(interactive=True), gr.update(interactive=True)


# ---------------------------------------------------------------------------
# 3. Time trajectory helpers
# ---------------------------------------------------------------------------
DEFAULT_TRAJECTORY_FRAMES = 81
TIME_PRESETS = [
    "linear",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "fast_forward",
    "hyperlapse",
    "slow_motion",
    "freeze_start",
    "freeze_end",
    "reverse",
    "reverse_bullet",
    "bullet_time",
    "boomerang",
    "yo_yo",
    "stutter",
    "time_warp",
    "custom",
]

TIME_PRESET_DESCRIPTIONS = {
    "linear": "Uniform playback speed.",
    "ease_in": "Starts cautiously and accelerates into the shot.",
    "ease_out": "Hits hard early and gently settles at the end.",
    "ease_in_out": "Slow-fast-slow cinematic timing.",
    "fast_forward": "Reach the end early, then hold.",
    "hyperlapse": "Very aggressive rush to the end frame.",
    "slow_motion": "Only traverses part of the source clip by the end.",
    "freeze_start": "Hold the opening moment, then launch forward.",
    "freeze_end": "Rush through the scene and land on the ending frame.",
    "reverse": "Play the source clip backward.",
    "reverse_bullet": "Reverse playback with a held middle moment.",
    "bullet_time": "Move, freeze on a moment, then continue.",
    "boomerang": "Go to the end, then return to the start.",
    "yo_yo": "Forward, snap part-way back, then surge forward again.",
    "stutter": "Advance in bursts with repeated temporal holds.",
    "time_warp": "Fast jump, rewind pocket, then recover to the end.",
    "custom": "User-defined time control.",
}

TIME_CURVE_EDITOR_HTML = """
<div class="neo-time-editor-shell">
  <style>
    .neo-time-editor-shell {
      border: 1px solid var(--border-color-primary, rgba(15, 23, 42, 0.12));
      border-radius: 14px;
      background: var(--block-background-fill, rgba(255, 255, 255, 0.96));
      padding: 12px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
      font-family: inherit;
    }
    .neo-time-editor-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }
    .neo-time-editor-help {
      font-size: 12px;
      line-height: 1.45;
      color: var(--body-text-color-subdued, #667085);
      margin: 0;
      flex: 1 1 auto;
    }
    .neo-time-editor-badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      flex: 0 0 auto;
    }
    .neo-time-editor-badge {
      display: inline-flex;
      align-items: center;
      padding: 3px 9px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.05);
      color: var(--body-text-color, #1f2937);
      font-size: 12px;
      font-weight: 500;
      white-space: nowrap;
    }
    .neo-time-editor-stage {
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(15, 23, 42, 0.02), rgba(15, 23, 42, 0.01)),
        var(--panel-background-fill, #f8fafc);
      border: 1px solid rgba(15, 23, 42, 0.08);
    }
    .neo-time-editor-svg {
      display: block;
      width: 100%;
      height: 260px;
      touch-action: none;
      cursor: crosshair;
    }
    .neo-time-editor-foot {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-top: 8px;
    }
    .neo-time-editor-status {
      font-size: 12px;
      color: var(--body-text-color-subdued, #667085);
      min-height: 18px;
      flex: 1 1 auto;
    }
    .neo-time-editor-reset {
      border: 1px solid rgba(15, 23, 42, 0.12);
      border-radius: 10px;
      padding: 7px 11px;
      font-size: 12px;
      font-weight: 500;
      background: var(--button-secondary-background-fill, rgba(255, 255, 255, 0.92));
      color: var(--body-text-color, #1f2937);
      cursor: pointer;
    }
    .neo-time-editor-reset:hover {
      background: rgba(15, 23, 42, 0.04);
    }
    @media (max-width: 760px) {
      .neo-time-editor-head,
      .neo-time-editor-foot {
        flex-direction: column;
        align-items: flex-start;
      }
      .neo-time-editor-badges {
        justify-content: flex-start;
      }
    }
  </style>
  <div class="neo-time-editor-head">
    <p class="neo-time-editor-help">Drag points to retime. Double-click to add one. Right-click an inner point to delete it.</p>
    <div class="neo-time-editor-badges">
      <span class="neo-time-editor-badge" data-role="frames">frames 81</span>
      <span class="neo-time-editor-badge" data-role="points">points 2</span>
    </div>
  </div>
  <div class="neo-time-editor-stage">
    <svg class="neo-time-editor-svg" preserveAspectRatio="none"></svg>
  </div>
  <div class="neo-time-editor-foot">
    <div class="neo-time-editor-status" data-role="status">Linked to Time Keyframes JSON.</div>
    <button type="button" class="neo-time-editor-reset" data-role="reset">Reload From JSON</button>
  </div>
</div>
"""

TIME_CURVE_EDITOR_JS = """
const root = element.querySelector('.neo-time-editor-shell') || element;
if (!root) {
  return;
}
if (element.__neoTimeEditorBooted) {
  return;
}
element.__neoTimeEditorBooted = true;
const svg = root.querySelector('.neo-time-editor-svg');
const statusNode = root.querySelector('[data-role="status"]');
const framesBadge = root.querySelector('[data-role="frames"]');
const pointsBadge = root.querySelector('[data-role="points"]');
const resetButton = root.querySelector('[data-role="reset"]');
const margin = { left: 18, right: 18, top: 16, bottom: 16 };
const svgWidth = 720;
const svgHeight = 260;
const state = {
  points: [],
  numFrames: 81,
  yMax: 80,
  dragIndex: null,
  hoverIndex: null,
  lastText: null,
  intervalId: null,
  resizeObserver: null,
};

function clamp(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function formatNumber(value) {
  return Number.parseFloat(value.toFixed(2)).toString();
}

function setStatus(message) {
  if (statusNode) {
    statusNode.textContent = message;
  }
}

function getTextarea() {
  return document.querySelector('#time-keyframes-input textarea');
}

function setNativeValue(node, value) {
  const prototype = node.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
  if (descriptor && descriptor.set) {
    descriptor.set.call(node, value);
  } else {
    node.value = value;
  }
}

function parseTextareaValue(text) {
  const trimmed = (text || '').trim();
  if (!trimmed) {
    return null;
  }
  let parsed;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    setStatus('Editor paused: invalid JSON in Time Keyframes.');
    return null;
  }
  if (!Array.isArray(parsed) || parsed.length === 0) {
    setStatus('Editor paused: expected a non-empty JSON list.');
    return null;
  }
  const points = [];
  for (const item of parsed) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const entries = Object.entries(item);
    if (entries.length !== 1) {
      continue;
    }
    const [frameKey, timeValue] = entries[0];
    const frame = Number.parseInt(frameKey, 10);
    const sourceTime = Number(timeValue);
    if (!Number.isFinite(frame) || !Number.isFinite(sourceTime)) {
      continue;
    }
    points.push({ frame, time: sourceTime });
  }
  if (points.length === 0) {
    setStatus('Editor paused: no valid points were found.');
    return null;
  }
  points.sort((a, b) => a.frame - b.frame);
  const deduped = [];
  for (const point of points) {
    if (deduped.length && deduped[deduped.length - 1].frame === point.frame) {
      deduped[deduped.length - 1] = point;
    } else {
      deduped.push(point);
    }
  }
  const maxFrame = Math.max(1, deduped[deduped.length - 1].frame);
  const yMax = Math.max(
    1,
    maxFrame,
    ...deduped.map((point) => point.time),
  );
  return {
    points: deduped,
    numFrames: maxFrame + 1,
    yMax,
  };
}

function serializePoints(points) {
  const payload = points.map((point) => ({
    [String(point.frame)]: Number(point.time.toFixed(2)),
  }));
  return JSON.stringify(payload, null, 2);
}

function innerWidth() {
  return svgWidth - margin.left - margin.right;
}

function innerHeight() {
  return svgHeight - margin.top - margin.bottom;
}

function lastFrame() {
  return Math.max(1, state.numFrames - 1);
}

function xToSvg(frame) {
  return margin.left + (frame / lastFrame()) * innerWidth();
}

function yToSvg(sourceTime) {
  return margin.top + (1 - sourceTime / state.yMax) * innerHeight();
}

function svgToPoint(clientX, clientY) {
  const rect = svg.getBoundingClientRect();
  const localX = clamp(clientX - rect.left, margin.left, rect.width - margin.right);
  const localY = clamp(clientY - rect.top, margin.top, rect.height - margin.bottom);
  const xNorm = clamp((localX - margin.left) / Math.max(1, rect.width - margin.left - margin.right), 0, 1);
  const yNorm = clamp(1 - (localY - margin.top) / Math.max(1, rect.height - margin.top - margin.bottom), 0, 1);
  return {
    frame: xNorm * lastFrame(),
    time: yNorm * state.yMax,
  };
}

function findPointIndex(clientX, clientY) {
  const rect = svg.getBoundingClientRect();
  const localX = clientX - rect.left;
  const localY = clientY - rect.top;
  let bestIndex = -1;
  let bestDistance = Infinity;
  state.points.forEach((point, index) => {
    const dx = localX - xToSvg(point.frame) * rect.width / svgWidth;
    const dy = localY - yToSvg(point.time) * rect.height / svgHeight;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestDistance <= 16 ? bestIndex : -1;
}

function syncToTextbox(triggerServer) {
  const textarea = getTextarea();
  if (!textarea) {
    setStatus('Waiting for Time Keyframes JSON input.');
    return;
  }
  const text = serializePoints(state.points);
  if (textarea.value === text && state.lastText === text) {
    return;
  }
  setNativeValue(textarea, text);
  textarea.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
  if (triggerServer) {
    textarea.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
  }
  state.lastText = text;
  setStatus('Curve updated from drag editor.');
}

function drawLine(x1, y1, x2, y2, color, width, dashArray) {
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1);
  line.setAttribute('y1', y1);
  line.setAttribute('x2', x2);
  line.setAttribute('y2', y2);
  line.setAttribute('stroke', color);
  line.setAttribute('stroke-width', width);
  if (dashArray) {
    line.setAttribute('stroke-dasharray', dashArray);
  }
  svg.appendChild(line);
}

function render() {
  svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
  svg.innerHTML = '';
  const background = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  background.setAttribute('x', '0');
  background.setAttribute('y', '0');
  background.setAttribute('width', String(svgWidth));
  background.setAttribute('height', String(svgHeight));
  background.setAttribute('fill', 'rgba(248, 250, 252, 0.0)');
  svg.appendChild(background);

  for (let i = 0; i <= 4; i += 1) {
    const x = margin.left + (i / 4) * innerWidth();
    drawLine(x, margin.top, x, svgHeight - margin.bottom, 'rgba(148, 163, 184, 0.24)', 1, '4 6');
  }
  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + (i / 4) * innerHeight();
    drawLine(margin.left, y, svgWidth - margin.right, y, 'rgba(148, 163, 184, 0.18)', 1, '4 6');
  }

  const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  polyline.setAttribute(
    'points',
    state.points.map((point) => `${xToSvg(point.frame)},${yToSvg(point.time)}`).join(' '),
  );
  polyline.setAttribute('fill', 'none');
  polyline.setAttribute('stroke', '#d95f02');
  polyline.setAttribute('stroke-width', '3');
  polyline.setAttribute('stroke-linecap', 'round');
  polyline.setAttribute('stroke-linejoin', 'round');
  svg.appendChild(polyline);

  state.points.forEach((point, index) => {
    const x = xToSvg(point.frame);
    const y = yToSvg(point.time);
    const halo = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    halo.setAttribute('cx', x);
    halo.setAttribute('cy', y);
    halo.setAttribute('r', index === state.hoverIndex || index === state.dragIndex ? '10' : '8');
    halo.setAttribute('fill', 'rgba(59, 130, 246, 0.16)');
    svg.appendChild(halo);

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', x);
    circle.setAttribute('cy', y);
    circle.setAttribute('r', '5');
    circle.setAttribute('fill', index === 0 || index === state.points.length - 1 ? '#0f172a' : '#1d4ed8');
    circle.setAttribute('stroke', '#ffffff');
    circle.setAttribute('stroke-width', '2');
    svg.appendChild(circle);
  });

  if (framesBadge) {
    framesBadge.textContent = `frames ${state.numFrames}`;
  }
  if (pointsBadge) {
    pointsBadge.textContent = `points ${state.points.length}`;
  }
  const activeIndex = state.dragIndex !== null ? state.dragIndex : state.hoverIndex;
  if (activeIndex !== null && activeIndex >= 0 && activeIndex < state.points.length) {
    const point = state.points[activeIndex];
    setStatus(`frame ${point.frame} -> source ${formatNumber(point.time)} | double-click to add | right-click to delete`);
  } else {
    setStatus('Drag to retime. Double-click to add a point. Right-click an inner point to delete it.');
  }
}

function loadFromTextarea(force) {
  const textarea = getTextarea();
  if (!textarea) {
    setStatus('Waiting for Time Keyframes JSON input.');
    return;
  }
  const text = textarea.value || '';
  if (!force && text === state.lastText) {
    return;
  }
  const parsed = parseTextareaValue(text);
  if (!parsed) {
    return;
  }
  state.points = parsed.points;
  state.numFrames = parsed.numFrames;
  state.yMax = parsed.yMax;
  state.lastText = text;
  render();
}

function updateDraggedPoint(clientX, clientY) {
  if (state.dragIndex === null) {
    return;
  }
  const candidate = svgToPoint(clientX, clientY);
  const index = state.dragIndex;
  let frame = Math.round(candidate.frame);
  if (index === 0) {
    frame = 0;
  } else if (index === state.points.length - 1) {
    frame = lastFrame();
  } else {
    const previousFrame = state.points[index - 1].frame + 1;
    const nextFrame = state.points[index + 1].frame - 1;
    frame = clamp(frame, previousFrame, nextFrame);
  }
  const time = Number(clamp(candidate.time, 0, state.yMax).toFixed(2));
  state.points[index] = { frame, time };
  render();
}

function addPoint(clientX, clientY) {
  const candidate = svgToPoint(clientX, clientY);
  const frame = Math.round(candidate.frame);
  if (frame <= 0 || frame >= lastFrame()) {
    return;
  }
  if (state.points.some((point) => point.frame === frame)) {
    return;
  }
  state.points.push({
    frame,
    time: Number(clamp(candidate.time, 0, state.yMax).toFixed(2)),
  });
  state.points.sort((a, b) => a.frame - b.frame);
  render();
  syncToTextbox(true);
}

function removePoint(clientX, clientY) {
  const index = findPointIndex(clientX, clientY);
  if (index <= 0 || index >= state.points.length - 1) {
    return;
  }
  state.points.splice(index, 1);
  state.hoverIndex = null;
  render();
  syncToTextbox(true);
}

function handlePointerMove(event) {
  if (state.dragIndex !== null) {
    updateDraggedPoint(event.clientX, event.clientY);
    return;
  }
  const nextHover = findPointIndex(event.clientX, event.clientY);
  if (nextHover !== state.hoverIndex) {
    state.hoverIndex = nextHover;
    render();
  }
}

function handlePointerUp(event) {
  if (state.dragIndex !== null) {
    updateDraggedPoint(event.clientX, event.clientY);
    state.dragIndex = null;
    syncToTextbox(true);
  }
}

svg.addEventListener('pointerdown', (event) => {
  const index = findPointIndex(event.clientX, event.clientY);
  if (index === -1) {
    return;
  }
  state.dragIndex = index;
  state.hoverIndex = index;
  svg.setPointerCapture(event.pointerId);
  event.preventDefault();
  render();
});

svg.addEventListener('pointermove', handlePointerMove);
svg.addEventListener('pointerup', handlePointerUp);
svg.addEventListener('pointercancel', handlePointerUp);
svg.addEventListener('dblclick', (event) => {
  event.preventDefault();
  addPoint(event.clientX, event.clientY);
});
svg.addEventListener('contextmenu', (event) => {
  const index = findPointIndex(event.clientX, event.clientY);
  if (index > 0 && index < state.points.length - 1) {
    event.preventDefault();
    removePoint(event.clientX, event.clientY);
  }
});

resetButton.addEventListener('click', () => {
  loadFromTextarea(true);
});

state.intervalId = window.setInterval(() => {
  loadFromTextarea(false);
}, 250);

if (window.ResizeObserver) {
  state.resizeObserver = new ResizeObserver(() => render());
  state.resizeObserver.observe(root);
}

loadFromTextarea(true);
"""


def _get_num_frames_from_state(state, fallback=DEFAULT_TRAJECTORY_FRAMES):
    if state is None:
        return fallback
    if "input_timestamps" in state:
        return int(state["input_timestamps"].shape[0])
    if "input_cam2world" in state:
        return int(state["input_cam2world"].shape[0])
    if "target_rgb" in state:
        return int(state["target_rgb"].shape[1])
    return fallback


def _time_keyframes_to_text(time_keyframes):
    return json.dumps(time_keyframes, indent=2)


def _dense_curve_to_time_keyframes(time_curve):
    curve = np.asarray(time_curve, dtype=np.float32)
    if curve.size == 0:
        return []
    if curve.size == 1:
        return [{0: float(curve[0])}]

    frame_indices = [0]
    previous_delta = float(curve[1] - curve[0])
    for idx in range(1, curve.size - 1):
        delta = float(curve[idx + 1] - curve[idx])
        if not np.isclose(delta, previous_delta, atol=1e-4, rtol=1e-4):
            frame_indices.append(idx)
        previous_delta = delta
    frame_indices.append(curve.size - 1)
    frame_indices = sorted(set(frame_indices))
    return [{int(idx): float(curve[idx])} for idx in frame_indices]


def _pairs_to_time_keyframes(pairs, num_frames):
    last_idx = max(num_frames - 1, 0)
    time_keyframes = []
    for frame_idx, time_value in pairs:
        clamped_idx = int(np.clip(frame_idx, 0, last_idx))
        time_keyframes.append({clamped_idx: float(time_value)})

    if num_frames > 1:
        normalized = {}
        for keyframe in time_keyframes:
            frame_idx, time_value = next(iter(keyframe.items()))
            normalized[frame_idx] = time_value
        if 0 not in normalized:
            normalized[0] = 0.0
        if last_idx not in normalized:
            normalized[last_idx] = float(last_idx)
        time_keyframes = [{frame_idx: normalized[frame_idx]} for frame_idx in sorted(normalized.keys())]
    return CameraTrajectory._normalize_time_keyframes(time_keyframes, num_frames)


def _fractional_pairs_to_time_keyframes(pairs, num_frames):
    last_idx = max(num_frames - 1, 0)
    last_time = float(last_idx)
    scaled_pairs = [
        (int(round(frame_ratio * last_idx)), float(time_ratio * last_time))
        for frame_ratio, time_ratio in pairs
    ]
    return _pairs_to_time_keyframes(scaled_pairs, num_frames)


def _build_time_preset_keyframes(preset, num_frames):
    preset_pairs = {
        "linear": [
            (0.0, 0.0),
            (1.0, 1.0),
        ],
        "ease_in": [
            (0.0, 0.0),
            (0.14, 0.03),
            (0.3, 0.12),
            (0.48, 0.3),
            (0.68, 0.56),
            (0.84, 0.8),
            (1.0, 1.0),
        ],
        "ease_out": [
            (0.0, 0.0),
            (0.16, 0.2),
            (0.32, 0.44),
            (0.52, 0.7),
            (0.7, 0.88),
            (0.86, 0.97),
            (1.0, 1.0),
        ],
        "ease_in_out": [
            (0.0, 0.0),
            (0.16, 0.05),
            (0.32, 0.18),
            (0.5, 0.5),
            (0.68, 0.82),
            (0.84, 0.95),
            (1.0, 1.0),
        ],
        "fast_forward": [
            (0.0, 0.0),
            (0.35, 1.0),
            (1.0, 1.0),
        ],
        "hyperlapse": [
            (0.0, 0.0),
            (0.12, 0.48),
            (0.24, 0.8),
            (0.36, 1.0),
            (1.0, 1.0),
        ],
        "slow_motion": [
            (0.0, 0.0),
            (1.0, 0.45),
        ],
        "freeze_start": [
            (0.0, 0.0),
            (0.22, 0.0),
            (0.46, 0.24),
            (0.7, 0.68),
            (1.0, 1.0),
        ],
        "freeze_end": [
            (0.0, 0.0),
            (0.18, 0.34),
            (0.42, 0.72),
            (0.68, 1.0),
            (1.0, 1.0),
        ],
        "reverse": [
            (0.0, 1.0),
            (1.0, 0.0),
        ],
        "reverse_bullet": [
            (0.0, 1.0),
            (0.28, 0.5),
            (0.72, 0.5),
            (1.0, 0.0),
        ],
        "bullet_time": [
            (0.0, 0.0),
            (0.28, 0.5),
            (0.72, 0.5),
            (1.0, 1.0),
        ],
        "boomerang": [
            (0.0, 0.0),
            (0.5, 1.0),
            (1.0, 0.0),
        ],
        "yo_yo": [
            (0.0, 0.0),
            (0.34, 1.0),
            (0.66, 0.18),
            (1.0, 1.0),
        ],
        "stutter": [
            (0.0, 0.0),
            (0.12, 0.12),
            (0.18, 0.12),
            (0.34, 0.38),
            (0.42, 0.38),
            (0.6, 0.66),
            (0.7, 0.66),
            (0.86, 0.92),
            (1.0, 1.0),
        ],
        "time_warp": [
            (0.0, 0.0),
            (0.16, 0.34),
            (0.28, 0.82),
            (0.42, 0.54),
            (0.62, 0.72),
            (0.8, 0.96),
            (1.0, 1.0),
        ],
    }
    return _fractional_pairs_to_time_keyframes(preset_pairs.get(preset, preset_pairs["linear"]), num_frames)


def _parse_time_keyframes_text(time_keyframes_text, num_frames):
    raw_text = (time_keyframes_text or "").strip()
    if raw_text == "":
        return _build_time_preset_keyframes("linear", num_frames)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid time keyframes JSON: {exc}") from exc
    try:
        return CameraTrajectory._normalize_time_keyframes(parsed, num_frames)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _extract_time_keyframes_from_data(data, num_frames):
    if "time_keyframes" in data:
        return CameraTrajectory._normalize_time_keyframes(data["time_keyframes"], num_frames)
    if "time_curve" in data:
        time_curve = CameraTrajectory._normalize_time_curve(data["time_curve"], num_frames)
        return _dense_curve_to_time_keyframes(time_curve)
    return _build_time_preset_keyframes("linear", num_frames)


def _build_time_curve_summary(time_curve):
    curve = np.asarray(time_curve, dtype=np.float32)
    if curve.size == 0:
        return "No source-time curve."
    deltas = np.diff(curve)
    if deltas.size == 0:
        direction = "static"
        hold_segments = 0
    elif np.all(deltas >= -1e-4):
        direction = "forward"
        hold_segments = int(np.count_nonzero(np.isclose(deltas, 0.0, atol=1e-4)))
    elif np.all(deltas <= 1e-4):
        direction = "reverse"
        hold_segments = int(np.count_nonzero(np.isclose(deltas, 0.0, atol=1e-4)))
    else:
        direction = "mixed"
        hold_segments = int(np.count_nonzero(np.isclose(deltas, 0.0, atol=1e-4)))
    backward_segments = int(np.count_nonzero(deltas < -1e-4))
    return (
        f"mode=absolute_source_time | clamp=clamp | frames={curve.size} | "
        f"source={curve[0]:.2f}->{curve[-1]:.2f} | min/max={curve.min():.2f}/{curve.max():.2f} | "
        f"direction={direction} | hold_segments={hold_segments} | backward_segments={backward_segments}"
    )


def _build_time_controls(num_frames, time_keyframes=None, preset="linear"):
    keyframes = time_keyframes or _build_time_preset_keyframes(preset, num_frames)
    time_curve = CameraTrajectory._build_time_curve_from_keyframes(keyframes, num_frames)
    description = TIME_PRESET_DESCRIPTIONS.get(preset, TIME_PRESET_DESCRIPTIONS["custom"])
    summary = f"preset={preset} | {description} | {_build_time_curve_summary(time_curve)}"
    return preset, _time_keyframes_to_text(keyframes), summary


def _build_time_controls_from_json(data, fallback_num_frames=DEFAULT_TRAJECTORY_FRAMES):
    num_frames = int(data.get("num_frames", fallback_num_frames))
    if "time_keyframes" in data or "time_curve" in data:
        keyframes = _extract_time_keyframes_from_data(data, num_frames)
        return _build_time_controls(num_frames, keyframes, preset="custom")
    return _build_time_controls(num_frames, preset="linear")


def _apply_time_preset(state, time_preset, current_text):
    num_frames = _get_num_frames_from_state(state)
    if time_preset == "custom":
        if (current_text or "").strip():
            keyframes = _parse_time_keyframes_text(current_text, num_frames)
            _, time_text, time_summary = _build_time_controls(num_frames, keyframes, preset="custom")
            return "custom", time_text, time_summary
        time_preset = "linear"
    keyframes = _build_time_preset_keyframes(time_preset, num_frames)
    _, time_text, time_summary = _build_time_controls(num_frames, keyframes, preset=time_preset)
    return time_preset, time_text, time_summary


def _refresh_time_controls(state, time_keyframes_text):
    num_frames = _get_num_frames_from_state(state)
    if (time_keyframes_text or "").strip() == "":
        return _apply_time_preset(state, "linear", "")
    keyframes = _parse_time_keyframes_text(time_keyframes_text, num_frames)
    _, time_text, time_summary = _build_time_controls(num_frames, keyframes, preset="custom")
    return "custom", time_text, time_summary


def _resolve_time_payload(num_frames, time_preset, time_keyframes_text):
    if (time_keyframes_text or "").strip():
        keyframes = _parse_time_keyframes_text(time_keyframes_text, num_frames)
    else:
        effective_preset = time_preset if time_preset in TIME_PRESETS and time_preset != "custom" else "linear"
        keyframes = _build_time_preset_keyframes(effective_preset, num_frames)
    time_curve = CameraTrajectory._build_time_curve_from_keyframes(keyframes, num_frames)
    payload = {
        "time_control_mode": "absolute_source_time",
        "time_clamp": "clamp",
        "time_keyframes": keyframes,
        "time_curve": [float(t) for t in time_curve.tolist()],
    }
    return payload, keyframes, time_curve


def _write_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _build_camera_params_json(
    state,
    t_type,
    angle,
    distance,
    orbit_radius,
    mode,
    zoom_ratio,
    use_first_frame,
    time_preset,
    time_keyframes_text,
    num_frames=None,
    name="gradio_traj",
):
    num_frames = int(num_frames or _get_num_frames_from_state(state))
    num_frames = max(num_frames, 2)
    last_frame = max(num_frames - 1, 0)
    time_payload, _, _ = _resolve_time_payload(num_frames, time_preset, time_keyframes_text)
    json_data = {
        "name": name,
        "mode": mode,
        "num_frames": num_frames,
        "zoom_ratio": float(zoom_ratio),
        "use_first_frame": bool(use_first_frame),
        "keyframes": [
            {"0": [{"static": {}}]},
            {
                str(last_frame): [{
                    t_type: {
                        "angle": int(angle),
                        "distance": float(distance),
                        "orbit_radius": float(orbit_radius),
                    }
                }]
            },
        ],
    }
    json_data.update(time_payload)
    return json_data


def _prepare_uploaded_trajectory_json(
    state,
    t_file,
    mode,
    zoom_ratio,
    use_first_frame,
    time_preset,
    time_keyframes_text,
):
    if t_file is None:
        raise gr.Error("Upload a trajectory JSON first.")
    with open(t_file, "r") as f:
        json_data = json.load(f)
    json_data["mode"] = mode
    json_data["zoom_ratio"] = zoom_ratio
    json_data["use_first_frame"] = use_first_frame
    json_data.setdefault("num_frames", _get_num_frames_from_state(state))
    time_payload, _, _ = _resolve_time_payload(int(json_data["num_frames"]), time_preset, time_keyframes_text)
    json_data.update(time_payload)
    return json_data


def _resample_camera_sequence(cam2world, num_frames):
    if cam2world.shape[0] == num_frames:
        return cam2world
    matrices = cam2world.detach().cpu().numpy()
    frame_indices = np.arange(matrices.shape[0], dtype=np.int32)
    resampled = CameraTrajectory._interpolate_sparse_matrices(frame_indices, matrices, num_frames)
    return torch.from_numpy(resampled).to(device=cam2world.device, dtype=cam2world.dtype)


def _resample_intrinsics(intrs, num_frames):
    if intrs.shape[0] == num_frames:
        return intrs
    source = intrs.detach().cpu().numpy()
    src_axis = np.linspace(0.0, 1.0, source.shape[0], dtype=np.float32)
    dst_axis = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    flattened = source.reshape(source.shape[0], -1)
    interpolated = np.stack(
        [np.interp(dst_axis, src_axis, flattened[:, idx]) for idx in range(flattened.shape[1])],
        axis=-1,
    )
    interpolated = interpolated.reshape(num_frames, *source.shape[1:]).astype(np.float32)
    return torch.from_numpy(interpolated).to(device=intrs.device, dtype=intrs.dtype)


def _poses_are_aligned(reference_pose, candidate_pose, translation_atol=1e-4, rotation_atol_rad=1e-3):
    translation_error = torch.linalg.vector_norm(reference_pose[:3, 3] - candidate_pose[:3, 3]).item()
    rotation_delta = reference_pose[:3, :3].transpose(0, 1) @ candidate_pose[:3, :3]
    cosine = torch.clamp((torch.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = float(torch.arccos(cosine).item())
    return translation_error <= translation_atol and rotation_error <= rotation_atol_rad


def _should_copy_first_frame(use_first_frame, render_timestamps, input_timestamps, target_cam2world, input_cam2world):
    if not use_first_frame:
        return False
    if render_timestamps.numel() == 0 or input_timestamps.numel() == 0:
        return False
    if abs(float(render_timestamps[0].item()) - float(input_timestamps[0].item())) > 1e-4:
        return False
    return _poses_are_aligned(input_cam2world[0], target_cam2world[0])


def _mask_tensor_to_pil_frames(target_mask):
    frames = []
    for i in range(target_mask.shape[1]):
        mask_f = (target_mask[0, i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        if mask_f.ndim == 3 and mask_f.shape[2] == 1:
            mask_f = np.repeat(mask_f, 3, axis=2)
        elif mask_f.ndim == 2:
            mask_f = np.stack([mask_f] * 3, axis=-1)
        frames.append(Image.fromarray(mask_f))
    return frames


@torch.no_grad()
def _render_trajectory_core(
    state,
    json_data,
    alpha_threshold,
    trajectory_path,
    preview_path=None,
    mask_path=None,
    glb_path=None,
    relative_anchor_pose=None,
):
    if state is None or "gaussians" not in state:
        raise gr.Error("Run reconstruction first.")

    _ensure_scene_state_on_device(state)
    _write_json_file(trajectory_path, json_data)
    cam_traj = CameraTrajectory.from_json(trajectory_path)

    static_flag = state.get("scene_type", "General scene") == "Static scene"
    input_cam2world = state["input_cam2world"]
    target_cam2world = cam_traj.c2w.to(device)
    num_frames = target_cam2world.shape[0]
    if cam_traj.mode == "relative":
        if relative_anchor_pose is not None:
            anchor_pose = relative_anchor_pose.to(device=target_cam2world.device, dtype=target_cam2world.dtype)
            target_cam2world = anchor_pose.unsqueeze(0) @ target_cam2world
        elif static_flag:
            target_cam2world = input_cam2world[0:1] @ target_cam2world
        else:
            source_cam2world = _resample_camera_sequence(input_cam2world, num_frames)
            target_cam2world = source_cam2world @ target_cam2world

    gaussians = state["gaussians"]
    input_intrs = state["input_intrs"]
    timestamps = state["input_timestamps"]
    H, W = state["height"], state["width"]

    if static_flag:
        K_render = input_intrs[:1].repeat(num_frames, 1, 1)
        render_timestamps = timestamps[:1].repeat(num_frames)
    else:
        K_render = _resample_intrinsics(input_intrs, num_frames)
        render_timestamps = cam_traj.time_curve.to(device=device, dtype=timestamps.dtype)
        render_timestamps = render_timestamps.clamp(
            min=float(timestamps.min().item()),
            max=float(timestamps.max().item()),
        )

    ratio = torch.linspace(1, cam_traj.zoom_ratio, K_render.shape[0], device=K_render.device, dtype=K_render.dtype)
    K_zoomed = K_render.clone()
    K_zoomed[:, 0, 0] *= ratio
    K_zoomed[:, 1, 1] *= ratio

    target_world2cam = homo_matrix_inverse(target_cam2world)
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[target_world2cam],
        render_Ks=[K_zoomed],
        render_timestamps=[render_timestamps],
        sh_degree=0, width=W, height=H,
    )

    target_mask = (target_alpha > alpha_threshold).float()
    if _should_copy_first_frame(cam_traj.use_first_frame, render_timestamps, timestamps, target_cam2world, input_cam2world):
        first_frame_rgb = F.to_tensor(state["images"][0]).permute(1, 2, 0).to(device)
        target_rgb[0, 0] = first_frame_rgb
        target_mask[0, 0] = 1.0

    state["target_rgb"] = target_rgb
    state["target_depth"] = target_depth
    state["target_mask"] = target_mask
    state["target_poses"] = target_cam2world.unsqueeze(0)
    state["target_intrs"] = K_zoomed.unsqueeze(0)
    state["target_timestamps"] = render_timestamps.unsqueeze(0)

    if preview_path is not None:
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        save_video(_target_rgb_to_pil_frames(target_rgb), preview_path, fps=16)
    if mask_path is not None:
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        save_video(_mask_tensor_to_pil_frames(target_mask), mask_path, fps=16)
    if glb_path is not None:
        scene = build_scene_glb(
            state["points"],
            state["colors"],
            state["frame_indices"],
            target_cam2world.detach().cpu().numpy(),
        )
        _export_scene_to(scene, glb_path)

    return {
        "state": state,
        "cam_traj": cam_traj,
        "target_cam2world": target_cam2world,
        "target_intrs": K_zoomed,
        "target_timestamps": render_timestamps,
        "preview_path": preview_path,
        "mask_path": mask_path,
        "glb_path": glb_path,
        "trajectory_path": trajectory_path,
    }


@torch.no_grad()
def _generate_video_core(state, prompt, negative_prompt, seed, output_path, compare_path=None):
    if state is None or "target_rgb" not in state:
        raise gr.Error("Run Render Preview first.")

    _ensure_render_targets_on_device(state)
    _set_model_phase(("text_encoder", "dit", "vae", "control_branch"))
    H, W = state["height"], state["width"]
    num_frames = int(state["target_rgb"].shape[1])
    wrapped_data = {
        "source_views": None,
        "target_rgb": state["target_rgb"],
        "target_depth": state["target_depth"],
        "target_mask": state["target_mask"],
        "target_poses": state["target_poses"],
        "target_intrs": state["target_intrs"],
    }

    try:
        generated_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed), rand_device=device,
            height=H, width=W, num_frames=num_frames,
            cfg_scale=1.0, num_inference_steps=4, tiled=False,
            **wrapped_data,
        )
    finally:
        _set_model_phase(())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_video(generated_frames, output_path, fps=16)
    if compare_path is not None:
        os.makedirs(os.path.dirname(compare_path), exist_ok=True)
        comparison_frames = _build_side_by_side_frames(
            _target_rgb_to_pil_frames(state["target_rgb"]),
            generated_frames,
        )
        save_video(comparison_frames, compare_path, fps=16)
    _cuda_cleanup()
    return generated_frames, output_path, compare_path


# ---------------------------------------------------------------------------
# 3. Build trajectory
# ---------------------------------------------------------------------------
def build_trajectory(state, t_type, mode, angle, distance, orbit_radius, zoom_ratio, use_first_frame,
                     time_preset, time_keyframes_text):
    """Build camera trajectory from UI rows, visualize, and export JSON."""
    if state is None or "gaussians" not in state:
        raise gr.Error("Run reconstruction first.")
    num_frames = _get_num_frames_from_state(state)
    time_payload, _, _ = _resolve_time_payload(num_frames, time_preset, time_keyframes_text)

    json_data = {
        "name": "gradio_traj",
        "mode": mode,
        "num_frames": num_frames,
        "zoom_ratio": zoom_ratio,
        "use_first_frame": use_first_frame,
        "keyframes": [
            {
                "0": [{"static": {}}]
            },
            {
                str(num_frames - 1): [{t_type: {"angle": int(angle), "distance": float(distance), "orbit_radius": float(orbit_radius)}}]
            }
        ]
    }
    json_data.update(time_payload)
    with open(JSON_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    cam_traj = CameraTrajectory.from_json(JSON_PATH)
    return cam_traj


def upload_trajectory(state, t_file):
    """Load trajectory JSON, build trajectory."""
    if state is None or "gaussians" not in state:
        raise gr.Error("Upload a trajectory JSON after reconstruction.")
    if t_file is None:
        raise gr.Error("Upload a trajectory JSON first.")

    cam_traj = CameraTrajectory.from_json(t_file)
    return cam_traj


def handle_traj_upload(t_file):
    """Parse uploaded trajectory JSON and extract shared parameters."""
    if t_file is None:
        return (gr.update(),) * 6
    with open(t_file, "r") as f:
        data = json.load(f)
    mode = data.get("mode", "relative")
    zoom_ratio = data.get("zoom_ratio", 1.0)
    use_first_frame = data.get("use_first_frame", True)
    time_preset, time_text, time_summary = _build_time_controls_from_json(data)
    return mode, zoom_ratio, use_first_frame, time_preset, time_text, time_summary


# ---------------------------------------------------------------------------
# 4. Render preview
# ---------------------------------------------------------------------------
@torch.no_grad()
def preview(state, selected_tab, t_file, t_type, angle, distance, orbit_r,
            mode, zoom, use_ff, alpha_threshold, time_preset, time_keyframes_text):
    """Build trajectory then render preview.

    The active tab determines the trajectory source:
    *TAB_TRAJ_FILE* uses the uploaded JSON; *TAB_CAMERA_PARAMS* uses sliders.
    """
    if state is None or "gaussians" not in state:
        raise gr.Error("Run reconstruction first.")
    if selected_tab == TAB_TRAJ_FILE:
        json_data = _prepare_uploaded_trajectory_json(
            state,
            t_file,
            mode,
            zoom,
            use_ff,
            time_preset,
            time_keyframes_text,
        )
    else:
        json_data = _build_camera_params_json(
            state,
            t_type,
            angle,
            distance,
            orbit_r,
            mode,
            zoom,
            use_ff,
            time_preset,
            time_keyframes_text,
        )
    result = _render_trajectory_core(
        state,
        json_data,
        alpha_threshold,
        trajectory_path=JSON_PATH,
        preview_path=PREVIEW_PATH,
        mask_path=MASK_PATH,
        glb_path=GLB_PATH,
    )
    return result["state"], result["glb_path"], PREVIEW_PATH, MASK_PATH, gr.update(interactive=True), JSON_PATH, gr.update(value=None)


# ---------------------------------------------------------------------------
# 4.5 Dynamic viewer
# ---------------------------------------------------------------------------
@torch.no_grad()
def render_dynamic_view(state, viewer_mode, time_index, yaw, pitch, roll, radius,
                        center_x, center_y, center_z, focal_scale):
    """Render a freely movable camera view at a selectable timestamp."""
    if state is None or "gaussians" not in state:
        raise gr.Error("Run reconstruction first.")

    H, W = state["height"], state["width"]
    input_cam2world = state["input_cam2world"]
    input_intrs = state["input_intrs"]
    input_timestamps = state["input_timestamps"]
    static_flag = state.get("scene_type", "General scene") == "Static scene"

    frame_idx = 0 if static_flag else int(np.clip(time_index, 0, len(input_timestamps) - 1))
    timestamp = input_timestamps[frame_idx].to(device=device)
    intrs = input_intrs[0 if static_flag else frame_idx].clone().to(device=device)
    intrs[0, 0] *= float(focal_scale)
    intrs[1, 1] *= float(focal_scale)

    if viewer_mode == "Input Camera":
        cam2world = input_cam2world[0 if static_flag else frame_idx].to(device=device)
        extra_cam2worlds = None
    else:
        orbit_center = state["scene_center"] + np.array([center_x, center_y, center_z], dtype=np.float32)
        cam2world_np = _build_orbit_camera_pose(orbit_center, yaw, pitch, roll, radius)
        cam2world = torch.from_numpy(cam2world_np).to(device=device, dtype=input_cam2world.dtype)
        extra_cam2worlds = [cam2world_np]

    world2cam = homo_matrix_inverse(cam2world.unsqueeze(0))[0]
    render_rgb, render_depth, render_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        state["gaussians"],
        render_viewmats=[world2cam.unsqueeze(0)],
        render_Ks=[intrs.unsqueeze(0)],
        render_timestamps=[timestamp.unsqueeze(0)],
        sh_degree=0,
        width=W,
        height=H,
    )

    rgb_image = _to_rgb_image(render_rgb[0, 0])
    depth_image = _to_scalar_image(render_depth[0, 0, ..., 0])
    alpha_image = _to_scalar_image(render_alpha[0, 0, ..., 0], scale=1.0)

    scene = build_scene_glb(
        state["points"],
        state["colors"],
        state["frame_indices"],
        input_cam2world.cpu().numpy(),
        selected_idx=frame_idx,
        extra_cam2worlds=extra_cam2worlds,
        extra_camera_colors=[(255, 255, 255)] if extra_cam2worlds is not None else None,
    )
    glb_path = _export_scene(scene)

    camera_pos = cam2world[:3, 3].detach().cpu().numpy()
    status = (
        f"mode={viewer_mode} | frame={frame_idx} | timestamp={int(timestamp.item())} | "
        f"camera=({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})"
    )
    return glb_path, rgb_image, depth_image, alpha_image, status


# ---------------------------------------------------------------------------
# 5. Generate final video
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_final(state, prompt, negative_prompt, seed):
    """Run diffusion generation using rendered conditioning."""
    _generate_video_core(state, prompt, negative_prompt, seed, OUTPUT_PATH, COMPARE_PATH)
    return OUTPUT_PATH, COMPARE_PATH


# ---------------------------------------------------------------------------
# 6. Multi-round workflow
# ---------------------------------------------------------------------------
def _gaussian_count(gaussians):
    total = 0
    for batch in gaussians:
        for gs in batch:
            total += int(gs.means.shape[0])
    return total


def _transform_points_np(points, transform):
    if points.shape[0] == 0:
        return points.copy()
    transform_np = transform.detach().cpu().float().numpy()
    return (points @ transform_np[:3, :3].T + transform_np[:3, 3]).astype(np.float32)


def _transform_gaussian(gs, transform):
    transformed = copy.copy(gs)
    for name, value in gs.__dict__.items():
        if torch.is_tensor(value):
            setattr(transformed, name, value.detach().clone())
        else:
            setattr(transformed, name, copy.deepcopy(value))

    R = transform[:3, :3].to(device=transformed.means.device, dtype=transformed.means.dtype)
    t = transform[:3, 3].to(device=transformed.means.device, dtype=transformed.means.dtype)
    transformed.means = transformed.means @ R.transpose(0, 1) + t

    if getattr(transformed, "rotations", None) is not None and transformed.rotations.numel() > 0:
        rotation_dtype = transformed.rotations.dtype
        rotation_device = transformed.rotations.device
        R_float = R.to(device=rotation_device, dtype=torch.float32)
        # Gaussians store quaternions as WXYZ, while the local utility uses XYZW.
        rotations_xyzw = transformed.rotations[..., [1, 2, 3, 0]].to(dtype=torch.float32)
        rotmats = quat_to_rotmat(rotations_xyzw)
        transformed_xyzw = rotmat_to_quat(R_float.unsqueeze(0) @ rotmats)
        transformed.rotations = transformed_xyzw[..., [3, 0, 1, 2]].to(
            device=rotation_device,
            dtype=rotation_dtype,
        )

    for velocity_name in ("forward_vel", "backward_vel"):
        velocity = getattr(transformed, velocity_name, None)
        if torch.is_tensor(velocity):
            R_vel = R.to(device=velocity.device, dtype=velocity.dtype)
            setattr(transformed, velocity_name, velocity @ R_vel.transpose(0, 1))
    return transformed


def _transform_reconstructed_state(state, transform):
    aligned = dict(state)
    transform = transform.to(device=state["input_cam2world"].device, dtype=state["input_cam2world"].dtype)

    aligned["input_cam2world"] = transform.unsqueeze(0) @ state["input_cam2world"]
    aligned["gaussians"] = [
        [_transform_gaussian(gs, transform) for gs in batch]
        for batch in state["gaussians"]
    ]
    aligned["points"] = _transform_points_np(state["points"], transform)
    aligned["colors"] = state["colors"].copy()
    aligned["frame_indices"] = state["frame_indices"].copy()
    aligned["scene_center"] = _transform_points_np(
        np.asarray(state["scene_center"], dtype=np.float32)[None],
        transform,
    )[0]
    aligned.pop("viewer_meta", None)
    return aligned


def _align_state_to_target_anchor(state, target_anchor_pose):
    new_anchor_pose = state["input_cam2world"][0].to(device=target_anchor_pose.device, dtype=target_anchor_pose.dtype)
    transform = target_anchor_pose @ homo_matrix_inverse(new_anchor_pose.unsqueeze(0))[0]
    return _transform_reconstructed_state(state, transform), transform


def _merge_arrays(first, second, dtype=None):
    if first.shape[0] == 0:
        merged = second.copy()
    elif second.shape[0] == 0:
        merged = first.copy()
    else:
        merged = np.concatenate([first, second], axis=0)
    if dtype is not None:
        merged = merged.astype(dtype)
    return merged


def _fuse_reconstructed_state(fused_state, aligned_state):
    fused = dict(aligned_state)
    fused.pop("source_views", None)
    _clear_transient_render_state(fused)
    base_gaussians = list(fused_state["gaussians"][0])
    new_gaussians = list(aligned_state["gaussians"][0])
    fused["gaussians"] = [base_gaussians + new_gaussians]

    base_frame_indices = fused_state["frame_indices"]
    new_frame_indices = aligned_state["frame_indices"]
    frame_offset = int(base_frame_indices.max() + 1) if base_frame_indices.shape[0] > 0 else 0
    fused["points"] = _merge_arrays(fused_state["points"], aligned_state["points"], dtype=np.float32)
    fused["colors"] = _merge_arrays(fused_state["colors"], aligned_state["colors"], dtype=np.uint8)
    fused["frame_indices"] = _merge_arrays(
        base_frame_indices,
        new_frame_indices + frame_offset,
        dtype=np.int32,
    )

    old_history = fused_state.get("camera_history_cam2world")
    if old_history is None:
        old_history = fused_state["input_cam2world"].detach().cpu().numpy()
    new_history = aligned_state["input_cam2world"].detach().cpu().numpy()
    fused["camera_history_cam2world"] = np.concatenate([old_history, new_history], axis=0)
    fused["source_label"] = "workflow fused scene"
    fused["scene_type"] = "General scene"
    if fused["points"].shape[0] > 0:
        fused["scene_center"] = np.median(fused["points"], axis=0).astype(np.float32)
    else:
        fused["scene_center"] = fused["input_cam2world"][:, :3, 3].detach().cpu().numpy().mean(axis=0).astype(np.float32)
    fused.pop("viewer_meta", None)
    return fused


def _export_fused_scene(state, output_path):
    cam2world = state.get("camera_history_cam2world")
    if cam2world is None:
        cam2world = state["input_cam2world"].detach().cpu().numpy()
    scene = build_scene_glb(
        state["points"],
        state["colors"],
        state["frame_indices"],
        np.asarray(cam2world, dtype=np.float32),
        vis_frame_num=21,
    )
    return _export_scene_to(scene, output_path)


def _workflow_media_state_from_first_frame(state):
    if state is None or "images" not in state or not state["images"]:
        raise gr.Error("Upload a video or image first.")
    return {
        "images": [state["images"][0].copy()],
        "scene_type": "Static scene",
        "source_label": f"{state.get('source_label', 'input')} first frame",
    }


def _workflow_initial_state(state, first_frame_only):
    if state is None or "images" not in state:
        raise gr.Error("Upload a video or image first.")
    if first_frame_only:
        workflow_state = _workflow_media_state_from_first_frame(state)
        workflow_state, _, _ = _run_reconstruction_core(
            workflow_state,
            source="workflow_initial",
            store_latest=False,
            export_glb=False,
        )
        return workflow_state
    if "gaussians" in state:
        workflow_state = dict(state)
    else:
        workflow_state = dict(state)
        workflow_state, _, _ = _run_reconstruction_core(
            workflow_state,
            source="workflow_initial",
            store_latest=False,
            export_glb=False,
        )
    workflow_state["camera_history_cam2world"] = workflow_state["input_cam2world"].detach().cpu().numpy()
    return workflow_state


def _normalize_iteration_plan(raw_iterations, iteration_count):
    if not isinstance(raw_iterations, list) or len(raw_iterations) == 0:
        raise gr.Error("Workflow plan must be a non-empty JSON list or an object with an 'iterations' list.")

    requested_count = len(raw_iterations) if iteration_count is None else int(iteration_count)
    count = int(np.clip(requested_count, 1, 20))
    iterations = [copy.deepcopy(item) for item in raw_iterations[:count]]
    while len(iterations) < count:
        next_item = copy.deepcopy(iterations[-1])
        next_item["seed"] = int(next_item.get("seed", 42)) + 1
        iterations.append(next_item)

    normalized = []
    for idx, item in enumerate(iterations):
        if not isinstance(item, dict):
            raise gr.Error(f"Workflow iteration {idx + 1} must be a JSON object.")
        motion = item.get("camera_motion", item.get("trajectory", "orbit_left"))
        if isinstance(motion, dict):
            motion = "orbit_left"
        if motion not in CameraTrajectory.VALID_TRAJECTORY_TYPES:
            motion = "orbit_left"
        mode = item.get("mode", "relative")
        if mode not in ("relative", "global"):
            mode = "relative"
        time_preset = item.get("time_preset", "linear")
        if time_preset not in TIME_PRESETS:
            time_preset = "linear"
        start_policy = item.get("start_policy", "previous_end")
        if start_policy not in WORKFLOW_START_POLICIES:
            start_policy = "previous_end"
        normalized.append({
            **item,
            "prompt": str(item.get("prompt", "A smooth video with complete scene content.")),
            "negative_prompt": str(item.get("negative_prompt", "")),
            "seed": int(item.get("seed", 42 + idx)),
            "num_frames": max(2, int(item.get("num_frames", DEFAULT_TRAJECTORY_FRAMES))),
            "camera_motion": motion,
            "angle": float(item.get("angle", 15)),
            "distance": float(item.get("distance", 0.1)),
            "orbit_radius": float(item.get("orbit_radius", 1.0)),
            "mode": mode,
            "zoom_ratio": float(item.get("zoom_ratio", 1.0)),
            "use_first_frame": bool(item.get("use_first_frame", True)),
            "alpha_threshold": float(item.get("alpha_threshold", 1.0)),
            "time_preset": time_preset,
            "start_policy": start_policy,
        })
    return normalized


def _parse_workflow_plan(plan_text, iteration_count):
    try:
        parsed = json.loads((plan_text or "").strip())
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid workflow plan JSON: {exc}") from exc
    raw_iterations = parsed.get("iterations") if isinstance(parsed, dict) else parsed
    return _normalize_iteration_plan(raw_iterations, iteration_count)


def _iteration_time_keyframes_text(iteration):
    if "time_keyframes" not in iteration:
        return ""
    value = iteration["time_keyframes"]
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _trajectory_json_from_iteration(state, iteration, index):
    if "keyframes" in iteration or isinstance(iteration.get("trajectory"), dict):
        json_data = {
            "name": iteration.get("name", f"workflow_iter_{index + 1:02d}"),
            "mode": iteration["mode"],
            "num_frames": iteration["num_frames"],
            "zoom_ratio": iteration["zoom_ratio"],
            "use_first_frame": iteration["use_first_frame"],
        }
        if "keyframes" in iteration:
            json_data["keyframes"] = iteration["keyframes"]
        else:
            json_data["trajectory"] = iteration["trajectory"]
        time_payload, _, _ = _resolve_time_payload(
            iteration["num_frames"],
            iteration["time_preset"],
            _iteration_time_keyframes_text(iteration),
        )
        json_data.update(time_payload)
        return json_data

    return _build_camera_params_json(
        state,
        iteration["camera_motion"],
        iteration["angle"],
        iteration["distance"],
        iteration["orbit_radius"],
        iteration["mode"],
        iteration["zoom_ratio"],
        iteration["use_first_frame"],
        iteration["time_preset"],
        _iteration_time_keyframes_text(iteration),
        num_frames=iteration["num_frames"],
        name=iteration.get("name", f"workflow_iter_{index + 1:02d}"),
    )


def _write_resolved_trajectory_json(render_result, output_path):
    target_cam2world = render_result["target_cam2world"].detach().cpu().float().numpy()
    frame_indices = list(range(target_cam2world.shape[0]))
    payload = {
        "name": "resolved_world_trajectory",
        "mode": "global",
        "num_frames": int(target_cam2world.shape[0]),
        "trajectory": {
            "frame_indices": frame_indices,
            "frame_matrices": target_cam2world.tolist(),
        },
        "time_curve": render_result["target_timestamps"].detach().cpu().float().tolist(),
    }
    return _write_json_file(output_path, payload)


WORKFLOW_PLAN_PRESET_LIBRARY = {
    "World expansion sweep": [
        {"camera_motion": "pull_out", "angle": 0, "distance": 0.22, "orbit_radius": 1.0, "time_preset": "ease_out", "prompt_addition": "Reveal a wider view of the surrounding world."},
        {"camera_motion": "orbit_left", "angle": 45, "distance": 0.0, "orbit_radius": 1.1, "time_preset": "ease_in_out", "prompt_addition": "Reveal coherent side details and preserve the scene layout."},
        {"camera_motion": "orbit_left", "angle": 105, "distance": 0.0, "orbit_radius": 1.2, "time_preset": "linear", "prompt_addition": "Turn toward the back side and complete unseen geometry naturally."},
        {"camera_motion": "push_in", "angle": 0, "distance": 0.18, "orbit_radius": 1.0, "time_preset": "ease_in", "prompt_addition": "Move forward into newly revealed space with stable details."},
        {"camera_motion": "move_right", "angle": 0, "distance": 0.18, "orbit_radius": 1.0, "time_preset": "linear", "prompt_addition": "Translate laterally and extend the world beyond the current border."},
        {"camera_motion": "orbit_right", "angle": 60, "distance": 0.0, "orbit_radius": 1.15, "time_preset": "ease_in_out", "prompt_addition": "Revisit the scene from a diagonal view and keep all additions consistent."},
    ],
    "Full orbit survey": [
        {"camera_motion": "orbit_left", "angle": 45, "distance": 0.0, "orbit_radius": 1.1, "time_preset": "linear", "prompt_addition": "Rotate left and expose the nearby side of the scene."},
        {"camera_motion": "orbit_left", "angle": 90, "distance": 0.0, "orbit_radius": 1.2, "time_preset": "linear", "prompt_addition": "Continue around the subject and complete the side geometry."},
        {"camera_motion": "orbit_left", "angle": 135, "distance": 0.0, "orbit_radius": 1.25, "time_preset": "ease_in_out", "prompt_addition": "Reach the back-side view with coherent lighting and structure."},
        {"camera_motion": "orbit_left", "angle": 180, "distance": 0.0, "orbit_radius": 1.3, "time_preset": "ease_out", "prompt_addition": "Show the opposite side and fill all previously unseen regions."},
        {"camera_motion": "pull_out", "angle": 0, "distance": 0.18, "orbit_radius": 1.0, "time_preset": "linear", "prompt_addition": "Pull back to stabilize the complete reconstructed world."},
        {"camera_motion": "orbit_right", "angle": 45, "distance": 0.0, "orbit_radius": 1.15, "time_preset": "ease_in_out", "prompt_addition": "Return toward a three-quarter view while preserving all fused details."},
    ],
    "Forward exploration": [
        {"camera_motion": "push_in", "angle": 0, "distance": 0.18, "orbit_radius": 1.0, "time_preset": "ease_in", "prompt_addition": "Move forward and reveal the next part of the world."},
        {"camera_motion": "move_right", "angle": 0, "distance": 0.16, "orbit_radius": 1.0, "time_preset": "linear", "prompt_addition": "Shift right while extending the visible environment."},
        {"camera_motion": "pan_left", "angle": 25, "distance": 0.0, "orbit_radius": 1.0, "time_preset": "ease_in_out", "prompt_addition": "Turn the camera left to inspect newly created space."},
        {"camera_motion": "push_in", "angle": 0, "distance": 0.16, "orbit_radius": 1.0, "time_preset": "linear", "prompt_addition": "Advance again and maintain temporal stability."},
        {"camera_motion": "move_left", "angle": 0, "distance": 0.14, "orbit_radius": 1.0, "time_preset": "ease_out", "prompt_addition": "Shift left to cover the opposite side of the path."},
        {"camera_motion": "pull_out", "angle": 0, "distance": 0.2, "orbit_radius": 1.0, "time_preset": "ease_in_out", "prompt_addition": "Pull back and show the expanded route as a coherent scene."},
    ],
    "Vertical and scale pass": [
        {"camera_motion": "boom_up", "angle": 0, "distance": 0.16, "orbit_radius": 1.0, "time_preset": "ease_out", "prompt_addition": "Move upward and reveal overhead structure."},
        {"camera_motion": "tilt_down", "angle": 25, "distance": 0.0, "orbit_radius": 1.0, "time_preset": "linear", "prompt_addition": "Tilt downward to connect upper and lower geometry."},
        {"camera_motion": "orbit_right", "angle": 55, "distance": 0.0, "orbit_radius": 1.15, "time_preset": "ease_in_out", "prompt_addition": "Orbit right from the elevated view and complete side content."},
        {"camera_motion": "boom_down", "angle": 0, "distance": 0.14, "orbit_radius": 1.0, "time_preset": "linear", "prompt_addition": "Descend into the reconstructed space with stable scale."},
        {"camera_motion": "tilt_up", "angle": 20, "distance": 0.0, "orbit_radius": 1.0, "time_preset": "ease_out", "prompt_addition": "Tilt upward and preserve continuity with the overhead pass."},
        {"camera_motion": "pull_out", "angle": 0, "distance": 0.18, "orbit_radius": 1.0, "time_preset": "ease_in_out", "prompt_addition": "Pull out to inspect the full vertical reconstruction."},
    ],
}
WORKFLOW_PLAN_PRESET_CHOICES = ["World expansion sweep", "Full orbit survey", "Forward exploration", "Vertical and scale pass", "Use current controls"]
WORKFLOW_PLAN_PRESET_DESCRIPTIONS = {
    "World expansion sweep": "默认 6 步：拉远 -> 左绕到侧面 -> 继续转向背面 -> 前进 -> 右移扩展边界 -> 右绕回斜侧视角。",
    "Full orbit survey": "默认 6 步：围绕主体持续左绕 45/90/135/180 度，补全背面，再拉远并右绕回三分之四视角。",
    "Forward exploration": "默认 6 步：前进探索 -> 右移 -> 左转检查新空间 -> 再前进 -> 左移覆盖另一侧 -> 拉远查看扩展路线。",
    "Vertical and scale pass": "默认 6 步：上升 -> 下俯 -> 高位右绕 -> 下降 -> 上仰 -> 拉远检查垂直方向和尺度。",
    "Use current controls": "按当前 Trajectory 控件里的相机运动、距离、角度和时间曲线重复 N 次，适合手动精调单一策略。",
}


def workflow_preset_summary(plan_preset):
    base = WORKFLOW_PLAN_PRESET_DESCRIPTIONS.get(plan_preset, WORKFLOW_PLAN_PRESET_DESCRIPTIONS["World expansion sweep"])
    return f"{base} 每步默认从上一轮相机终点继续。"


def _workflow_time_keyframes_for_preset(time_preset, fallback_text=""):
    if time_preset == "custom" and (fallback_text or "").strip():
        return _parse_time_keyframes_text(fallback_text, DEFAULT_TRAJECTORY_FRAMES)
    effective_preset = time_preset if time_preset in TIME_PRESETS and time_preset != "custom" else "linear"
    return _build_time_preset_keyframes(effective_preset, DEFAULT_TRAJECTORY_FRAMES)


def _compose_workflow_plan_text(
    plan_preset,
    iteration_count,
    prompt_text,
    negative_prompt,
    seed_value,
    current_step,
    fallback_time_keyframes_text="",
):
    requested_count = 6 if iteration_count is None else int(iteration_count)
    count = int(np.clip(requested_count, 1, 20))
    if plan_preset == "Use current controls":
        steps = [current_step] * count
    else:
        preset_steps = WORKFLOW_PLAN_PRESET_LIBRARY.get(plan_preset, WORKFLOW_PLAN_PRESET_LIBRARY["World expansion sweep"])
        steps = [preset_steps[idx % len(preset_steps)] for idx in range(count)]

    plan = []
    for idx, step in enumerate(steps):
        step_time_preset = step.get("time_preset", current_step.get("time_preset", "linear"))
        prompt_addition = step.get("prompt_addition", "")
        prompt_full = prompt_text if not prompt_addition else f"{prompt_text} {prompt_addition}"
        plan.append({
            "name": f"workflow_iter_{idx + 1:02d}",
            "prompt": prompt_full,
            "negative_prompt": negative_prompt,
            "seed": int(seed_value) + idx,
            "num_frames": DEFAULT_TRAJECTORY_FRAMES,
            "camera_motion": step.get("camera_motion", current_step["camera_motion"]),
            "angle": float(step.get("angle", current_step["angle"])),
            "distance": float(step.get("distance", current_step["distance"])),
            "orbit_radius": float(step.get("orbit_radius", current_step["orbit_radius"])),
            "mode": step.get("mode", current_step["mode"]),
            "zoom_ratio": float(step.get("zoom_ratio", current_step["zoom_ratio"])),
            "use_first_frame": bool(step.get("use_first_frame", current_step["use_first_frame"])),
            "alpha_threshold": float(step.get("alpha_threshold", current_step["alpha_threshold"])),
            "time_preset": step_time_preset,
            "start_policy": step.get("start_policy", "previous_end"),
            "time_keyframes": _workflow_time_keyframes_for_preset(step_time_preset, fallback_time_keyframes_text),
        })
    return json.dumps({"preset": plan_preset, "iterations": plan}, indent=2)


def build_workflow_plan_text(
    plan_preset,
    iteration_count,
    prompt_text,
    negative_prompt,
    seed_value,
    t_type,
    angle,
    distance,
    orbit_radius,
    mode,
    zoom_ratio,
    use_first_frame,
    alpha_threshold,
    time_preset,
    time_keyframes_text,
):
    motion = t_type if t_type in CameraTrajectory.VALID_TRAJECTORY_TYPES and t_type != "static" else "orbit_left"
    current_step = {
        "camera_motion": motion,
        "angle": float(angle),
        "distance": float(distance),
        "orbit_radius": float(orbit_radius),
        "mode": mode,
        "zoom_ratio": float(zoom_ratio),
        "use_first_frame": bool(use_first_frame),
        "alpha_threshold": float(alpha_threshold),
        "time_preset": time_preset,
    }
    return _compose_workflow_plan_text(
        plan_preset,
        iteration_count,
        prompt_text,
        negative_prompt,
        seed_value,
        current_step,
        fallback_time_keyframes_text=time_keyframes_text,
    )


@torch.no_grad()
def run_workflow(state, iteration_count, first_frame_only, plan_text):
    iterations = _parse_workflow_plan(plan_text, iteration_count)
    _offload_registered_scene_gaussians_to_cpu()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(WORKFLOW_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    status_lines = [f"run_id={run_id}", "initializing scene"]
    try:
        fused_state = _workflow_initial_state(state, first_frame_only)
    except torch.cuda.OutOfMemoryError as exc:
        _raise_cuda_oom("initializing the workflow scene", exc)
    initial_anchor_pose = fused_state["input_cam2world"][0].detach().cpu().clone()
    workflow_anchor_pose = initial_anchor_pose.clone()
    initial_scene_path = _export_fused_scene(fused_state, os.path.join(run_dir, "initial_scene.glb"))
    initial_scene_id = f"{run_id}_initial"
    _register_scene(
        scene_id=initial_scene_id,
        label=f"{run_id} | Initial",
        state=fused_state,
        source="workflow_initial",
        glb_path=initial_scene_path,
        run_id=run_id,
        iteration=0,
        make_active=True,
    )
    manifest = {
        "run_id": run_id,
        "initial_scene_id": initial_scene_id,
        "initial_scene": initial_scene_path,
        "first_frame_only": bool(first_frame_only),
        "iterations": [],
    }

    last_output_path = None
    last_compare_path = None
    generated_paths = []
    compare_paths = []
    final_scene_path = initial_scene_path

    for idx, iteration in enumerate(iterations):
        iter_dir = os.path.join(run_dir, f"iter_{idx + 1:02d}")
        os.makedirs(iter_dir, exist_ok=True)
        start_policy = iteration.get("start_policy", "previous_end")
        if start_policy == "initial":
            relative_anchor_pose = initial_anchor_pose
        elif start_policy == "reconstruction_first":
            relative_anchor_pose = None
        else:
            start_policy = "previous_end"
            relative_anchor_pose = workflow_anchor_pose
        status_lines.append(
            f"iter {idx + 1}/{len(iterations)}: render target trajectory "
            f"(start_policy={start_policy})"
        )

        trajectory_json = _trajectory_json_from_iteration(fused_state, iteration, idx)
        try:
            render_result = _render_trajectory_core(
                fused_state,
                trajectory_json,
                iteration["alpha_threshold"],
                trajectory_path=os.path.join(iter_dir, "trajectory.json"),
                preview_path=os.path.join(iter_dir, "preview.mp4"),
                mask_path=os.path.join(iter_dir, "mask.mp4"),
                glb_path=os.path.join(iter_dir, "target_trajectory.glb"),
                relative_anchor_pose=relative_anchor_pose,
            )
        except torch.cuda.OutOfMemoryError as exc:
            _raise_cuda_oom(f"rendering workflow iteration {idx + 1}", exc)
        target_anchor_pose = render_result["target_cam2world"][0].detach().cpu().clone()
        target_end_pose = render_result["target_cam2world"][-1].detach().cpu().clone()
        resolved_trajectory_path = _write_resolved_trajectory_json(
            render_result,
            os.path.join(iter_dir, "target_trajectory_resolved.json"),
        )
        _offload_registered_scene_gaussians_to_cpu()

        status_lines.append(f"iter {idx + 1}/{len(iterations)}: generate video")
        try:
            generated_frames, output_path, compare_path = _generate_video_core(
                fused_state,
                iteration["prompt"],
                iteration["negative_prompt"],
                iteration["seed"],
                output_path=os.path.join(iter_dir, "generated.mp4"),
                compare_path=os.path.join(iter_dir, "render_vs_generated.mp4"),
            )
        except torch.cuda.OutOfMemoryError as exc:
            _clear_transient_render_state(fused_state)
            _raise_cuda_oom(f"generating video for workflow iteration {idx + 1}", exc)
        last_output_path = output_path
        last_compare_path = compare_path
        generated_paths.append(output_path)
        compare_paths.append(compare_path)
        _clear_transient_render_state(fused_state)
        _cuda_cleanup()

        status_lines.append(f"iter {idx + 1}/{len(iterations)}: reconstruct generated video")
        reconstructed_state = {
            "images": generated_frames,
            "scene_type": "General scene",
            "source_label": os.path.basename(output_path),
        }
        try:
            reconstructed_state, _, _ = _run_reconstruction_core(
                reconstructed_state,
                source=f"workflow_iter_{idx + 1:02d}",
                store_latest=False,
                export_glb=False,
            )
        except torch.cuda.OutOfMemoryError as exc:
            _raise_cuda_oom(f"reconstructing generated video for workflow iteration {idx + 1}", exc)

        aligned_state, align_transform = _align_state_to_target_anchor(reconstructed_state, target_anchor_pose)
        fused_state = _fuse_reconstructed_state(fused_state, aligned_state)
        final_scene_path = _export_fused_scene(fused_state, os.path.join(iter_dir, "fused_scene.glb"))
        scene_id = f"{run_id}_iter_{idx + 1:02d}"
        _register_scene(
            scene_id=scene_id,
            label=f"{run_id} | After iter {idx + 1}",
            state=fused_state,
            source=f"workflow_iter_{idx + 1:02d}",
            glb_path=final_scene_path,
            generated_video=output_path,
            comparison_video=compare_path,
            run_id=run_id,
            iteration=idx + 1,
            make_active=True,
        )

        iteration_record = {
            "index": idx + 1,
            "scene_id": scene_id,
            "prompt": iteration["prompt"],
            "seed": iteration["seed"],
            "trajectory": render_result["trajectory_path"],
            "resolved_trajectory": resolved_trajectory_path,
            "preview": render_result["preview_path"],
            "mask": render_result["mask_path"],
            "generated": output_path,
            "comparison": compare_path,
            "fused_scene": final_scene_path,
            "start_policy": start_policy,
            "align_transform": align_transform.detach().cpu().float().tolist(),
            "trajectory_start_pose": target_anchor_pose.float().tolist(),
            "trajectory_end_pose": target_end_pose.float().tolist(),
            "fused_gaussians": _gaussian_count(fused_state["gaussians"]),
            "fused_points": int(fused_state["points"].shape[0]),
        }
        manifest["iterations"].append(iteration_record)
        workflow_anchor_pose = target_end_pose
        status_lines.append(
            f"iter {idx + 1}/{len(iterations)}: fused "
            f"{iteration_record['fused_gaussians']} gaussians, {iteration_record['fused_points']} points"
        )
        _offload_registered_scene_gaussians_to_cpu()

    manifest["final_scene"] = final_scene_path
    all_generated_path = _concat_video_files(
        generated_paths,
        os.path.join(run_dir, "generated_all_iterations.mp4"),
        fps=16,
    )
    all_compare_path = _concat_video_files(
        compare_paths,
        os.path.join(run_dir, "render_vs_generated_all_iterations.mp4"),
        fps=16,
    )
    manifest["final_video"] = all_generated_path or last_output_path
    manifest["final_comparison"] = all_compare_path or last_compare_path
    manifest["last_iteration_video"] = last_output_path
    manifest["last_iteration_comparison"] = last_compare_path
    manifest["all_generated"] = all_generated_path
    manifest["all_comparison"] = all_compare_path
    _store_latest_scene(fused_state, source="workflow")
    all_scene_id = f"{run_id}_all"
    _register_scene(
        scene_id=all_scene_id,
        label=f"{run_id} | All iterations",
        state=fused_state,
        source="workflow_all_iterations",
        glb_path=final_scene_path,
        generated_video=manifest["final_video"],
        comparison_video=manifest["final_comparison"],
        run_id=run_id,
        iteration="all",
        make_active=True,
    )
    manifest["all_scene_id"] = all_scene_id
    manifest_path = _write_json_file(os.path.join(run_dir, "manifest.json"), manifest)
    with VIEWER_LOCK:
        SCENE_REGISTRY["active_id"] = all_scene_id
    _export_fused_scene(fused_state, GLB_PATH)

    status_lines.append(f"done: {run_dir}")
    with VIEWER_LOCK:
        result_records = _ordered_scene_records_unlocked(run_id=run_id)
    result_choices = _scene_choices_from_records(result_records)
    result_value = all_scene_id
    return (
        fused_state,
        final_scene_path,
        manifest["final_video"],
        manifest["final_comparison"],
        manifest_path,
        "\n".join(status_lines),
        gr.update(choices=result_choices, value=result_value, interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def select_workflow_result(scene_id):
    if not scene_id:
        raise gr.Error("Select a workflow result first.")
    with VIEWER_LOCK:
        record = SCENE_REGISTRY["records"].get(scene_id)
        if record is None:
            raise gr.Error(f"Workflow result is not available in memory: {scene_id}")
        SCENE_REGISTRY["active_id"] = scene_id
        LATEST_SCENE["state"] = record["state"]
        LATEST_SCENE["meta"] = record["meta"]
        LATEST_SCENE["source"] = record["source"]
        LATEST_SCENE["updated_at"] = time.time()

    status = (
        f"selected={record['label']} | "
        f"scene_id={record['scene_id']} | "
        f"video={record.get('generated_video') or 'none'}"
    )
    return (
        record["state"],
        record.get("glb_path"),
        record.get("generated_video"),
        record.get("comparison_video"),
        status,
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

VALID_TYPES = sorted(CameraTrajectory.VALID_TRAJECTORY_TYPES)
TAB_CAMERA_PARAMS = "tab_camera_params"
TAB_TRAJ_FILE = "tab_traj_file"
DEFAULT_TIME_PRESET, DEFAULT_TIME_TEXT, DEFAULT_TIME_SUMMARY = _build_time_controls(
    DEFAULT_TRAJECTORY_FRAMES,
    preset="linear",
)
DEFAULT_WORKFLOW_PROMPT = (
    "A smooth video with complete scene content. "
    "Inpaint any missing regions or margins naturally to match the surrounding scene."
)
DEFAULT_WORKFLOW_CURRENT_STEP = {
    "camera_motion": "orbit_left",
    "angle": 15,
    "distance": 0.1,
    "orbit_radius": 1.0,
    "mode": "relative",
    "zoom_ratio": 1.0,
    "use_first_frame": True,
    "alpha_threshold": 1.0,
    "time_preset": "linear",
}
DEFAULT_WORKFLOW_PLAN_TEXT = _compose_workflow_plan_text(
    "World expansion sweep",
    6,
    DEFAULT_WORKFLOW_PROMPT,
    "",
    42,
    DEFAULT_WORKFLOW_CURRENT_STEP,
)

with gr.Blocks(title="NeoVerse Interactive Demo") as demo:
    gr.HTML(
    """
    <div style="text-align: center;">
    <h1>
        <strong style="background: linear-gradient(to right, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">NeoVerse</strong>
        <span>: Enhancing 4D World Model with in-the-wild Monocular Videos</span>
    </h1>
    <p>
        📑 <a href="https://arxiv.org/abs/2601.00393">arXiv</a> &nbsp&nbsp | &nbsp&nbsp 🌐 <a href="https://neoverse-4d.github.io">Project</a> &nbsp&nbsp  | &nbsp&nbsp🖥️ <a href="https://github.com/IamCreateAI/NeoVerse">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Yuppie1204/NeoVerse">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/models/Yuppie1204/NeoVerse">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 🎞️ <a href="https://www.bilibili.com/video/BV1ezvYBBEMi">BiliBili</a> &nbsp&nbsp | &nbsp&nbsp 🎥 <a href="https://youtu.be/1k8Ikf8zbZw">YouTube</a> &nbsp&nbsp
    </p>
    </div>
    <div style="font-size: 16px; line-height: 1.5;">
        <p>NeoVerse is a versatile 4D world model that turns monocular videos into free-viewpoint video generation.
        Given a single video or a set of images, NeoVerse reconstructs the underlying 4D scene and lets you
        render novel-trajectory videos along any custom camera path.</p>
    <ol>
        <li><strong>Upload</strong> &mdash; In the left column, upload a video or multiple images and select the scene type (General / Static).</li>
        <li><strong>Reconstruct</strong> &mdash; Click "Reconstruct" to perform 4D reconstruction. The middle column visualises the scene as a Gaussian-Splatting-centred point cloud so you can inspect the spatial layout and camera scale.</li>
        <li><strong>Design Camera Trajectory</strong> &mdash; Two input modes are available under the <em>Camera Parameters</em> and <em>Trajectory File</em> tabs:
            <ul>
                <li><em>Camera Parameters</em>: select a camera motion type (pan, tilt, orbit, push, etc.) and adjust angle, distance, and orbit radius with the sliders. The coordinate convention is detailed in<a href="https://github.com/IamCreateAI/NeoVerse/blob/main/docs/coordinate_system.md">Coordinate System</a>.</li>
                <li><em>Trajectory File</em>: upload a trajectory JSON file for full control over keyframes. The format is described in<a href="https://github.com/IamCreateAI/NeoVerse/blob/main/docs/trajectory_format.md">Trajectory Format</a>.</li>
            </ul>
            Click "Render" to preview RGB and mask renderings of the planned path.
        </li>
        <li><strong>Generate</strong> &mdash; In the right column, enter your prompt and click "Generate". NeoVerse synthesises the final video conditioned on the designed trajectory.</li>
    </ol>
    <h3>Key Parameters:</h3>
    <ul>
        <li><strong>Scene Type</strong> &mdash; <em>General</em>: for videos with camera or object motion; frames are sampled across the full time range. <em>Static</em>: for a single image or a stationary scene; all frames share the same timestamp.</li>
        <li><strong>Mode</strong> &mdash; <em>Relative</em>: the designed trajectory is composed with the reconstructed input camera, so movements are relative to the original viewpoint. <em>Global</em>: the trajectory matrices are used directly in world space.</li>
        <li><strong>Alpha Threshold</strong> &mdash; Controls the binary mask derived from the rendered alpha channel. Default 1.0 keeps all regions re-painted.</li>
    </ul>
    <p><strong>Note:</strong> Selecting an example from the gallery will automatically trigger reconstruction. Please wait a few seconds for it to complete before clicking "Render" to preview the target trajectory and renderings.</p>
    </div>
    """)
    app_state = gr.State(value=None)
    selected_tab_state = gr.State(value=TAB_CAMERA_PARAMS)

    with gr.Row():
        # ---- Left column: Upload ----
        with gr.Column(scale=1):
            scene_type = gr.Radio(["General scene", "Static scene"],
                                  value="General scene", label="Scene Type")
            file_upload = gr.File(file_count="multiple", label="Upload Video or Images",
                                  interactive=True, file_types=["image", "video"])
            image_gallery = gr.Gallery(label="Preview", columns=4, height=200,
                                       object_fit="contain")
            reconstruct_btn = gr.Button("Reconstruct", variant="primary",
                                        interactive=False)

            gr.Markdown("### Examples")
            _examples = _get_example_videos()
            if _examples:
                _gallery_items = [(ex["file"], ex["name"]) for ex in _examples]
                example_gallery = gr.Gallery(
                    value=_gallery_items,
                    label="Click to load",
                    columns=2, height=300,
                    object_fit="contain",
                    show_label=False,
                    interactive=True, preview=False, allow_preview=False,
                )

        # ---- Middle column: Visualization + Trajectory ----
        with gr.Column(scale=3):
            with gr.Row():
                model3d = gr.Model3D(label="Point Clouds Reference", height=350,
                                    zoom_speed=0.5, pan_speed=0.5, scale=1.0)
                with gr.Column(scale=1):
                    preview_video = gr.Video(label="RGB Rendering", height=170)
                    mask_video = gr.Video(label="Mask Rendering", height=170)

            with gr.Accordion("Dynamic 4D Viewer", open=False):
                with gr.Row():
                    viewer_mode = gr.Radio(
                        ["Orbit Viewer", "Input Camera"],
                        value="Orbit Viewer",
                        label="Viewer Mode",
                    )
                    viewer_time = gr.Slider(
                        minimum=0,
                        maximum=80,
                        value=0,
                        step=1,
                        label="Frame / Time",
                    )
                    viewer_focal = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="Focal Scale",
                    )
                with gr.Row():
                    viewer_yaw = gr.Slider(minimum=-180, maximum=180, value=0, step=1, label="Yaw")
                    viewer_pitch = gr.Slider(minimum=-89, maximum=89, value=-10, step=1, label="Pitch")
                    viewer_roll = gr.Slider(minimum=-180, maximum=180, value=0, step=1, label="Roll")
                    viewer_radius = gr.Slider(minimum=0.2, maximum=5.0, value=1.5, step=0.05, label="Radius")
                with gr.Row():
                    viewer_center_x = gr.Slider(minimum=-2.0, maximum=2.0, value=0.0, step=0.05, label="Center X Offset")
                    viewer_center_y = gr.Slider(minimum=-2.0, maximum=2.0, value=0.0, step=0.05, label="Center Y Offset")
                    viewer_center_z = gr.Slider(minimum=-2.0, maximum=2.0, value=0.0, step=0.05, label="Center Z Offset")
                viewer_btn = gr.Button("Render Dynamic View", variant="secondary", interactive=False)
                viewer_status = gr.Textbox(label="Viewer Status", interactive=False)
                with gr.Row():
                    viewer_rgb = gr.Image(label="Viewer RGB", type="pil", interactive=False)
                    viewer_depth = gr.Image(label="Viewer Depth", type="pil", interactive=False)
                    viewer_alpha = gr.Image(label="Viewer Alpha", type="pil", interactive=False)

            with gr.Tabs() as traj_tabs:
                with gr.Tab("Camera Parameters", id=TAB_CAMERA_PARAMS) as tab_camera:
                    with gr.Row():
                        traj_type = gr.Dropdown(choices=VALID_TYPES, value="static",
                                                label="Camera Motion")
                        traj_angle = gr.Slider(minimum=0, maximum=60, value=0,
                                               step=1, label="Angle")
                        traj_distance = gr.Slider(minimum=0, maximum=1, value=0,
                                                  step=0.01, label="Distance")
                        traj_orbit = gr.Slider(minimum=0, maximum=2, value=0,
                                               step=0.1, label="Orbit Radius")
                with gr.Tab("Trajectory File", id=TAB_TRAJ_FILE) as tab_traj:
                    traj_upload = gr.File(
                        label="Upload Trajectory JSON",
                        file_types=[".json"],
                        file_count="single",
                        interactive=True,
                    )
            with gr.Row():
                traj_mode = gr.Radio(["relative", "global"], value="relative",
                                        label="Mode")
                zoom_ratio_input = gr.Slider(minimum=0.1, maximum=2, value=1.0,
                                                step=0.1, label="Zoom Ratio")
                alpha_threshold_input = gr.Slider(minimum=0, maximum=1, value=1.0,
                                                    step=0.01, label="Alpha Threshold")
                use_first_frame_input = gr.Checkbox(value=True,
                                                        label="Use First Frame")
            with gr.Accordion("Time Trajectory", open=True):
                with gr.Row():
                    time_preset_input = gr.Dropdown(
                        choices=TIME_PRESETS,
                        value=DEFAULT_TIME_PRESET,
                        label="Time Preset",
                        elem_id="time-preset-input",
                    )
                    time_apply_btn = gr.Button("Apply Preset", variant="secondary")
                time_curve_editor = gr.HTML(
                    value=TIME_CURVE_EDITOR_HTML,
                    js_on_load=TIME_CURVE_EDITOR_JS,
                    container=False,
                    elem_id="time-curve-editor",
                )
                time_keyframes_input = gr.Textbox(
                    value=DEFAULT_TIME_TEXT,
                    label="Time Keyframes JSON",
                    lines=8,
                    placeholder='[\n  {"0": 0.0},\n  {"40": 10.0},\n  {"80": 80.0}\n]',
                    elem_id="time-keyframes-input",
                )
                time_summary_output = gr.Textbox(
                    value=DEFAULT_TIME_SUMMARY,
                    label="Time Summary",
                    interactive=False,
                    elem_id="time-summary-output",
                )
            traj_download = gr.File(
                label="Download Trajectory JSON",
                interactive=False,
            )
            preview_btn = gr.Button("Render", variant="primary", interactive=False)

        # ---- Right column: Generation ----
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="A smooth video with complete scene content. "
                      "Inpaint any missing regions or margins naturally "
                      "to match the surrounding scene.",
            )
            neg_prompt = gr.Textbox(label="Negative Prompt", value="")
            seed = gr.Number(label="Seed", value=42, precision=0)
            output_video = gr.Video(label="Generated Video")
            compare_video = gr.Video(label="Render vs Generated", height=300)
            generate_btn = gr.Button("Generate", variant="primary", interactive=False)
            with gr.Accordion("Multi-round Workflow", open=False):
                workflow_plan_preset = gr.Dropdown(
                    label="Plan Preset",
                    choices=WORKFLOW_PLAN_PRESET_CHOICES,
                    value="World expansion sweep",
                )
                workflow_preset_info = gr.Textbox(
                    label="Plan Preset Summary",
                    value=workflow_preset_summary("World expansion sweep"),
                    interactive=False,
                    lines=2,
                )
                workflow_iterations = gr.Number(label="Iteration Count", value=6, precision=0)
                workflow_first_frame_only = gr.Checkbox(
                    value=False,
                    label="Use Input First Frame Only",
                )
                workflow_build_plan_btn = gr.Button("Build / Refresh Plan", variant="secondary")
                workflow_plan_text = gr.Textbox(
                    label="Iteration Plan JSON",
                    value=DEFAULT_WORKFLOW_PLAN_TEXT,
                    lines=22,
                )
                workflow_run_btn = gr.Button("Run Workflow", variant="primary")
                workflow_result_select = gr.Dropdown(
                    label="Workflow Results",
                    choices=[],
                    interactive=False,
                )
                workflow_status = gr.Textbox(label="Workflow Status", lines=10, interactive=False)
                workflow_manifest = gr.File(label="Workflow Manifest", interactive=False)

    # ================================================================
    # Wiring
    # ================================================================

    workflow_plan_preset.change(
        fn=workflow_preset_summary,
        inputs=[workflow_plan_preset],
        outputs=[workflow_preset_info],
    )

    # Sync default params when camera motion type changes
    _DEFAULT_PARAMS = {
        "pan_left": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "pan_right": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "tilt_up": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "tilt_down": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "move_left": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "move_right": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "push_in": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "pull_out": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "boom_up": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "boom_down": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "orbit_left": {"angle": 15, "distance": 0, "orbit_radius": 1.0},
        "orbit_right": {"angle": 15, "distance": 0, "orbit_radius": 1.0},
        "static": {"angle": 0, "distance": 0, "orbit_radius": 0},
    }

    def _sync_traj_params(ttype):
        p = _DEFAULT_PARAMS.get(ttype, {})
        return p.get("angle", 0), p.get("distance", 0), p.get("orbit_radius", 0)

    traj_type.input(fn=_sync_traj_params,
                     inputs=[traj_type],
                     outputs=[traj_angle, traj_distance, traj_orbit])

    # Track selected tab via state
    tab_camera.select(fn=lambda: TAB_CAMERA_PARAMS, inputs=[], outputs=[selected_tab_state])
    tab_traj.select(fn=lambda: TAB_TRAJ_FILE, inputs=[], outputs=[selected_tab_state])

    # Upload
    file_upload.upload(fn=handle_upload,
                       inputs=[file_upload, scene_type],
                       outputs=[app_state, image_gallery, reconstruct_btn])
    scene_type.input(fn=handle_upload,
                      inputs=[file_upload, scene_type],
                      outputs=[app_state, image_gallery, reconstruct_btn])

    # Example gallery
    if _examples:
        def _load_example(evt: gr.SelectData):
            """Load an example and apply its preset parameters."""
            ex = _examples[evt.index]
            sc_type = ex.get("scene_type", "General scene")
            state, pil_images, btn_update = handle_upload([ex["file"]], sc_type)
            traj_file = ex.get("traj_file", None)
            if traj_file and os.path.exists(traj_file):
                with open(traj_file, "r") as f:
                    traj_data = json.load(f)
                time_preset, time_text, time_summary = _build_time_controls_from_json(traj_data)
            else:
                time_preset, time_text, time_summary = _build_time_controls(
                    DEFAULT_TRAJECTORY_FRAMES,
                    preset="linear",
                )
            if traj_file:
                tab_sel = gr.Tabs(selected=TAB_TRAJ_FILE)
                tab_id = TAB_TRAJ_FILE
            else:
                tab_sel = gr.Tabs(selected=TAB_CAMERA_PARAMS)
                tab_id = TAB_CAMERA_PARAMS
            return (state, pil_images, btn_update,
                    sc_type,
                    ex.get("camera_motion", "static"),
                    ex.get("angle", 0),
                    ex.get("distance", 0),
                    ex.get("orbit_radius", 0),
                    ex.get("mode", "relative"),
                    ex.get("zoom_ratio", 1.0),
                    ex.get("alpha_threshold", 1.0),
                    ex.get("use_first_frame", True),
                    traj_file,
                    tab_sel,
                    tab_id,
                    time_preset,
                    time_text,
                    time_summary)

        example_gallery.select(
            fn=_load_example,
            inputs=[],
            outputs=[app_state, image_gallery, reconstruct_btn,
                     scene_type,
                     traj_type, traj_angle, traj_distance, traj_orbit,
                     traj_mode, zoom_ratio_input, alpha_threshold_input,
                     use_first_frame_input, traj_upload, traj_tabs,
                     selected_tab_state, time_preset_input, time_keyframes_input,
                     time_summary_output],
        ).then(
            fn=reconstruct,
            inputs=[app_state],
            outputs=[app_state, model3d, preview_btn, viewer_btn],
        )

    # Reconstruct
    reconstruct_btn.click(
        fn=reconstruct,
        inputs=[app_state],
        outputs=[app_state, model3d, preview_btn, viewer_btn],
    )
    # Preview (build trajectory + render + export JSON)
    # Active tab determines trajectory source
    preview_btn.click(
        fn=preview,
        inputs=[app_state, selected_tab_state, traj_upload, traj_type, traj_angle, traj_distance, traj_orbit,
                traj_mode, zoom_ratio_input, use_first_frame_input,
                alpha_threshold_input, time_preset_input, time_keyframes_input],
        outputs=[app_state, model3d,
                 preview_video, mask_video, generate_btn, traj_download, compare_video],
    )

    # Sync shared params from uploaded trajectory JSON
    traj_upload.change(
        fn=handle_traj_upload,
        inputs=[traj_upload],
        outputs=[
            traj_mode,
            zoom_ratio_input,
            use_first_frame_input,
            time_preset_input,
            time_keyframes_input,
            time_summary_output,
        ],
    )

    time_apply_btn.click(
        fn=_apply_time_preset,
        inputs=[app_state, time_preset_input, time_keyframes_input],
        outputs=[time_preset_input, time_keyframes_input, time_summary_output],
    )

    time_keyframes_input.change(
        fn=_refresh_time_controls,
        inputs=[app_state, time_keyframes_input],
        outputs=[time_preset_input, time_keyframes_input, time_summary_output],
    )

    # Generate
    generate_btn.click(
        fn=generate_final,
        inputs=[app_state, prompt, neg_prompt, seed],
        outputs=[output_video, compare_video],
    )

    workflow_build_plan_btn.click(
        fn=build_workflow_plan_text,
        inputs=[
            workflow_plan_preset,
            workflow_iterations,
            prompt,
            neg_prompt,
            seed,
            traj_type,
            traj_angle,
            traj_distance,
            traj_orbit,
            traj_mode,
            zoom_ratio_input,
            use_first_frame_input,
            alpha_threshold_input,
            time_preset_input,
            time_keyframes_input,
        ],
        outputs=[workflow_plan_text],
    )

    workflow_plan_preset.change(
        fn=build_workflow_plan_text,
        inputs=[
            workflow_plan_preset,
            workflow_iterations,
            prompt,
            neg_prompt,
            seed,
            traj_type,
            traj_angle,
            traj_distance,
            traj_orbit,
            traj_mode,
            zoom_ratio_input,
            use_first_frame_input,
            alpha_threshold_input,
            time_preset_input,
            time_keyframes_input,
        ],
        outputs=[workflow_plan_text],
    )

    workflow_run_btn.click(
        fn=run_workflow,
        inputs=[app_state, workflow_iterations, workflow_first_frame_only, workflow_plan_text],
        outputs=[
            app_state,
            model3d,
            output_video,
            compare_video,
            workflow_manifest,
            workflow_status,
            workflow_result_select,
            preview_btn,
            viewer_btn,
        ],
    )

    workflow_result_select.change(
        fn=select_workflow_result,
        inputs=[workflow_result_select],
        outputs=[
            app_state,
            model3d,
            output_video,
            compare_video,
            workflow_status,
            preview_btn,
            viewer_btn,
        ],
    )

    viewer_btn.click(
        fn=render_dynamic_view,
        inputs=[
            app_state,
            viewer_mode,
            viewer_time,
            viewer_yaw,
            viewer_pitch,
            viewer_roll,
            viewer_radius,
            viewer_center_x,
            viewer_center_y,
            viewer_center_z,
            viewer_focal,
        ],
        outputs=[model3d, viewer_rgb, viewer_depth, viewer_alpha, viewer_status],
    )


def _encode_image_bytes(image, modality):
    buffer = io.BytesIO()
    if modality == "rgb":
        image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), "image/jpeg"
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


VIEWER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NeoVerse Viewer</title>
  <style>
    :root {
      --paper: #f4efe6;
      --panel: rgba(255, 252, 246, 0.9);
      --ink: #171717;
      --muted: #6b685f;
      --signal: #d95f02;
      --signal-2: #1f6f78;
      --line: rgba(23, 23, 23, 0.12);
      --viewport: #111111;
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Helvetica Neue", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217, 95, 2, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(31, 111, 120, 0.1), transparent 25%),
        linear-gradient(180deg, #fbf7f1 0%, var(--paper) 100%);
    }
    .shell {
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 22px 36px;
    }
    .masthead {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 18px;
    }
    .title-wrap h1 {
      margin: 0;
      font-size: clamp(30px, 5vw, 54px);
      line-height: 0.94;
      letter-spacing: -0.04em;
      font-weight: 700;
    }
    .title-wrap h1 span {
      color: var(--signal);
    }
    .title-wrap p {
      margin: 10px 0 0;
      max-width: 760px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }
    .top-links {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .pill-link, .primary-btn, .secondary-btn {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 10px 16px;
      font-size: 13px;
      text-decoration: none;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.72);
      transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
    }
    .pill-link:hover, .primary-btn:hover, .secondary-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.08);
    }
    .primary-btn {
      border-color: rgba(217, 95, 2, 0.25);
      background: linear-gradient(135deg, rgba(217, 95, 2, 0.14), rgba(255, 255, 255, 0.94));
      cursor: pointer;
    }
    .secondary-btn {
      cursor: pointer;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .sidebar {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .sidebar section {
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }
    .sidebar section:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    .section-label {
      margin: 0 0 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 7px;
      margin-bottom: 10px;
    }
    .field:last-child {
      margin-bottom: 0;
    }
    .field label {
      font-size: 13px;
      font-weight: 600;
    }
    input[type="text"], select, input[type="number"] {
      width: 100%;
      border: 1px solid rgba(23, 23, 23, 0.14);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      background: rgba(255, 255, 255, 0.86);
      color: var(--ink);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--signal);
    }
    .button-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .button-row > * {
      flex: 1 1 0;
    }
    .viewer-card {
      padding: 18px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
      min-height: 760px;
    }
    .viewer-toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .viewer-toolbar .toolbar-group {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .status-chip {
      padding: 9px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid var(--line);
      font-size: 12px;
      letter-spacing: 0.02em;
      color: var(--muted);
    }
    .viewport-wrap {
      position: relative;
      overflow: hidden;
      border-radius: 24px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background:
        radial-gradient(circle at 20% 20%, rgba(217, 95, 2, 0.16), transparent 22%),
        radial-gradient(circle at 80% 24%, rgba(31, 111, 120, 0.16), transparent 18%),
        linear-gradient(180deg, #202020 0%, var(--viewport) 100%);
      min-height: 540px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    }
    #viewer-image {
      width: 100%;
      height: 100%;
      min-height: 540px;
      object-fit: contain;
      display: block;
      user-select: none;
      -webkit-user-drag: none;
    }
    .overlay {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      pointer-events: none;
    }
    .overlay-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      padding: 16px;
    }
    .overlay-bottom {
      padding: 16px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      flex-wrap: wrap;
    }
    .badge {
      background: rgba(17, 17, 17, 0.7);
      color: #f3efe8;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      backdrop-filter: blur(10px);
    }
    .badge strong {
      color: #fff6d5;
    }
    .hint-box {
      max-width: 520px;
      background: rgba(17, 17, 17, 0.62);
      color: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      padding: 14px 16px;
      font-size: 13px;
      line-height: 1.55;
      backdrop-filter: blur(10px);
    }
    .timeline {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
    }
    .timeline .time-label {
      min-width: 64px;
      font-size: 12px;
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .timeline button {
      min-width: 92px;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .mini-grid .field {
      margin: 0;
    }
    .scene-facts {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .scene-facts strong {
      display: block;
      color: var(--ink);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .loading-strip {
      height: 3px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(17, 17, 17, 0.06);
    }
    .loading-strip::before {
      content: "";
      display: block;
      width: 32%;
      height: 100%;
      background: linear-gradient(90deg, var(--signal), var(--signal-2));
      transform: translateX(-120%);
      animation: slide 1.2s infinite ease-in-out;
    }
    .hidden { display: none !important; }
    @keyframes slide {
      0% { transform: translateX(-120%); }
      100% { transform: translateX(420%); }
    }
    @media (max-width: 1080px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .viewer-card {
        min-height: 0;
      }
      .viewport-wrap, #viewer-image {
        min-height: 420px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="masthead">
      <div class="title-wrap">
        <h1><span>NeoVerse</span> Live 4D Viewer</h1>
        <p>Drag to orbit, shift-drag or right-drag to pan, scroll to dolly, scrub time continuously, and query the reconstructed 4D scene on demand. The legacy Gradio workflow is still available if you need preview rendering or full generation.</p>
      </div>
      <div class="top-links">
        <a class="pill-link" href="/gradio" target="_blank" rel="noreferrer">Open Legacy Gradio</a>
        <a class="pill-link" href="https://github.com/IamCreateAI/NeoVerse" target="_blank" rel="noreferrer">GitHub</a>
      </div>
    </div>

    <div class="grid">
      <aside class="panel sidebar">
        <section>
          <p class="section-label">Source</p>
          <div class="field">
            <label for="example-select">Example</label>
            <select id="example-select"></select>
          </div>
          <div class="button-row">
            <button class="secondary-btn" id="use-example-btn">Use Example Path</button>
          </div>
          <div class="field">
            <label for="path-input">Local Video Or Image Path</label>
            <input id="path-input" type="text" placeholder="/root/.../robot.mp4">
          </div>
          <div class="field">
            <label for="scene-type">Scene Type</label>
            <select id="scene-type">
              <option value="General scene">General scene</option>
              <option value="Static scene">Static scene</option>
            </select>
          </div>
          <div class="button-row">
            <button class="primary-btn" id="reconstruct-btn">Reconstruct Scene</button>
            <button class="secondary-btn" id="reset-view-btn">Reset View</button>
          </div>
        </section>

        <section>
          <p class="section-label">Render</p>
          <div class="mini-grid">
            <div class="field">
              <label for="viewer-mode">Camera Mode</label>
              <select id="viewer-mode">
                <option value="Orbit Viewer">Orbit Viewer</option>
                <option value="Input Camera">Input Camera</option>
              </select>
            </div>
            <div class="field">
              <label for="modality-select">Modality</label>
              <select id="modality-select">
                <option value="rgb">RGB</option>
                <option value="depth">Depth</option>
                <option value="alpha">Alpha</option>
                <option value="mask">Mask</option>
              </select>
            </div>
            <div class="field">
              <label for="radius-slider">Radius</label>
              <input id="radius-slider" type="range" min="0.2" max="5.0" step="0.01" value="1.5">
            </div>
            <div class="field">
              <label for="focal-slider">Focal Scale</label>
              <input id="focal-slider" type="range" min="0.5" max="2.0" step="0.01" value="1.0">
            </div>
          </div>
          <div class="field">
            <label for="mask-threshold">Mask Threshold</label>
            <input id="mask-threshold" type="range" min="0" max="1" step="0.01" value="0.95">
          </div>
          <div class="field">
            <label for="time-slider">Time</label>
            <div class="timeline">
              <button class="secondary-btn" id="play-btn">Play</button>
              <input id="time-slider" type="range" min="0" max="80" step="0.01" value="0">
              <div class="time-label" id="time-value">0.00</div>
            </div>
          </div>
        </section>

        <section>
          <p class="section-label">Scene</p>
          <div class="field">
            <label for="scene-select">Available Scene</label>
            <select id="scene-select"></select>
          </div>
          <div class="scene-facts" id="scene-facts">
            <div><strong>Source</strong><span id="fact-source">No scene</span></div>
            <div><strong>Frames</strong><span id="fact-frames">-</span></div>
            <div><strong>Resolution</strong><span id="fact-resolution">-</span></div>
            <div><strong>Scale</strong><span id="fact-scale">-</span></div>
          </div>
        </section>

        <section>
          <p class="section-label">Controls</p>
          <div style="font-size:13px; line-height:1.6; color:var(--muted);">
            <div><strong style="color:var(--ink);">Mouse</strong>: left drag orbit, shift-drag or right-drag pan, wheel dolly.</div>
            <div><strong style="color:var(--ink);">Keys</strong>: W/S forward-back, A/D strafe, Q/E vertical, space play-pause.</div>
            <div><strong style="color:var(--ink);">Behavior</strong>: while moving, the viewer renders a preview pass first and refines when interaction settles.</div>
          </div>
        </section>
      </aside>

      <main class="panel viewer-card">
        <div class="viewer-toolbar">
          <div class="toolbar-group">
            <div class="status-chip" id="status-line">Waiting for reconstruction.</div>
            <div class="status-chip" id="render-line">No render yet.</div>
          </div>
          <div class="toolbar-group">
            <div class="status-chip">Root: <strong>/</strong></div>
            <div class="status-chip">Fallback: <strong>/gradio</strong></div>
          </div>
        </div>

        <div class="viewport-wrap" id="viewport">
          <img id="viewer-image" alt="NeoVerse render viewport">
          <div class="overlay">
            <div class="overlay-top">
              <div class="badge" id="badge-left"><strong>Idle</strong> ready for scene reconstruction</div>
              <div class="badge" id="badge-right">mode orbit | modality rgb</div>
            </div>
            <div class="overlay-bottom">
              <div class="hint-box" id="hint-box">Paste a local path or choose an example, reconstruct the scene once, then drag anywhere in the viewport to request that camera pose. The time bar supports continuous timestamps instead of frame-only jumps.</div>
              <div style="min-width:240px;">
                <div class="loading-strip hidden" id="loading-strip"></div>
              </div>
            </div>
          </div>
        </div>

        <div class="viewer-toolbar">
          <div class="toolbar-group">
            <div class="status-chip" id="camera-line">camera unavailable</div>
          </div>
          <div class="toolbar-group">
            <div class="status-chip" id="timing-line">render latency unavailable</div>
          </div>
        </div>
      </main>
    </div>
  </div>

  <script>
    const state = {
      ready: false,
      playing: false,
      timeValue: 0,
      yaw: 0,
      pitch: -10,
      roll: 0,
      radius: 1.5,
      centerX: 0,
      centerY: 0,
      centerZ: 0,
      focalScale: 1.0,
      viewerMode: 'Orbit Viewer',
      modality: 'rgb',
      maskThreshold: 0.95,
      renderInFlight: false,
      pendingRender: null,
      interactiveRenderRaf: null,
      fullRenderTimer: null,
      playbackRaf: null,
      playbackLastTs: null,
      lastPlaybackEnqueueAt: 0,
      scene: null,
      sceneId: '',
      scenes: [],
      examples: [],
      dragging: false,
      dragMode: 'orbit',
      lastX: 0,
      lastY: 0,
      currentImageUrl: null,
    };

    const el = {
      exampleSelect: document.getElementById('example-select'),
      useExampleBtn: document.getElementById('use-example-btn'),
      pathInput: document.getElementById('path-input'),
      sceneType: document.getElementById('scene-type'),
      sceneSelect: document.getElementById('scene-select'),
      reconstructBtn: document.getElementById('reconstruct-btn'),
      resetViewBtn: document.getElementById('reset-view-btn'),
      viewerMode: document.getElementById('viewer-mode'),
      modalitySelect: document.getElementById('modality-select'),
      radiusSlider: document.getElementById('radius-slider'),
      focalSlider: document.getElementById('focal-slider'),
      maskThreshold: document.getElementById('mask-threshold'),
      timeSlider: document.getElementById('time-slider'),
      timeValue: document.getElementById('time-value'),
      playBtn: document.getElementById('play-btn'),
      viewerImage: document.getElementById('viewer-image'),
      viewport: document.getElementById('viewport'),
      statusLine: document.getElementById('status-line'),
      renderLine: document.getElementById('render-line'),
      badgeLeft: document.getElementById('badge-left'),
      badgeRight: document.getElementById('badge-right'),
      hintBox: document.getElementById('hint-box'),
      loadingStrip: document.getElementById('loading-strip'),
      cameraLine: document.getElementById('camera-line'),
      timingLine: document.getElementById('timing-line'),
      factSource: document.getElementById('fact-source'),
      factFrames: document.getElementById('fact-frames'),
      factResolution: document.getElementById('fact-resolution'),
      factScale: document.getElementById('fact-scale'),
    };

    function clamp(value, minValue, maxValue) {
      return Math.min(Math.max(value, minValue), maxValue);
    }

    function normalize(vec) {
      const norm = Math.hypot(vec[0], vec[1], vec[2]) || 1;
      return [vec[0] / norm, vec[1] / norm, vec[2] / norm];
    }

    function cross(a, b) {
      return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ];
    }

    function getCameraBasis() {
      const yaw = state.yaw * Math.PI / 180;
      const pitch = state.pitch * Math.PI / 180;
      const forward = normalize([
        -Math.sin(yaw) * Math.cos(pitch),
        -Math.sin(pitch),
        -Math.cos(yaw) * Math.cos(pitch),
      ]);
      const upWorld = Math.abs(forward[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
      const right = normalize(cross(upWorld, forward));
      const up = normalize(cross(forward, right));
      return { forward, right, up };
    }

    function movePivot(deltaRight, deltaUp, deltaForward) {
      const { forward, right, up } = getCameraBasis();
      const step = Math.max(state.radius * 0.025, 0.01);
      state.centerX += (right[0] * deltaRight + up[0] * deltaUp + forward[0] * deltaForward) * step;
      state.centerY += (right[1] * deltaRight + up[1] * deltaUp + forward[1] * deltaForward) * step;
      state.centerZ += (right[2] * deltaRight + up[2] * deltaUp + forward[2] * deltaForward) * step;
    }

    function setLoading(active, message) {
      el.loadingStrip.classList.toggle('hidden', !active);
      if (message) {
        el.badgeLeft.innerHTML = '<strong>' + message + '</strong>';
      }
    }

    function setStatus(message) {
      el.statusLine.textContent = message;
    }

    function setRenderInfo(message) {
      el.renderLine.textContent = message;
    }

    function snapshotRenderState() {
      return {
        sceneId: state.sceneId,
        viewerMode: state.viewerMode,
        modality: state.modality,
        timeValue: state.timeValue,
        yaw: state.yaw,
        pitch: state.pitch,
        roll: state.roll,
        radius: state.radius,
        centerX: state.centerX,
        centerY: state.centerY,
        centerZ: state.centerZ,
        focalScale: state.focalScale,
        maskThreshold: state.maskThreshold,
      };
    }

    function requestRender(resolutionScale) {
      if (!state.ready || !state.scene) {
        return;
      }
      state.pendingRender = {
        snapshot: snapshotRenderState(),
        resolutionScale,
      };
      if (!state.renderInFlight) {
        void flushRenderQueue();
      }
    }

    async function flushRenderQueue() {
      if (state.renderInFlight || !state.pendingRender || !state.ready || !state.scene) {
        return;
      }

      const job = state.pendingRender;
      state.pendingRender = null;
      state.renderInFlight = true;
      updateOverlayLabels();
      setLoading(true, job.resolutionScale < 1 ? 'preview' : 'rendering');

      const params = new URLSearchParams({
        scene_id: job.snapshot.sceneId || '',
        viewer_mode: job.snapshot.viewerMode,
        modality: job.snapshot.modality,
        time_value: String(job.snapshot.timeValue),
        yaw: String(job.snapshot.yaw),
        pitch: String(job.snapshot.pitch),
        roll: String(job.snapshot.roll),
        radius: String(job.snapshot.radius),
        center_x: String(job.snapshot.centerX),
        center_y: String(job.snapshot.centerY),
        center_z: String(job.snapshot.centerZ),
        focal_scale: String(job.snapshot.focalScale),
        mask_threshold: String(job.snapshot.maskThreshold),
        resolution_scale: String(job.resolutionScale),
      });

      const startedAt = performance.now();
      try {
        const response = await fetch('/api/render?' + params.toString(), {
          cache: 'no-store',
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({ detail: 'Render failed.' }));
          throw new Error(payload.detail || 'Render failed.');
        }

        const blob = await response.blob();
        if (state.currentImageUrl) {
          URL.revokeObjectURL(state.currentImageUrl);
        }
        state.currentImageUrl = URL.createObjectURL(blob);
        el.viewerImage.src = state.currentImageUrl;

        const renderMs = response.headers.get('x-render-ms') || (performance.now() - startedAt).toFixed(1);
        const cameraPos = response.headers.get('x-camera-pos') || 'n/a';
        const statusText = response.headers.get('x-status') || 'render complete';
        setRenderInfo(statusText);
        el.cameraLine.textContent = 'camera ' + cameraPos;
        el.timingLine.textContent = 'render ' + renderMs + ' ms';
        el.badgeLeft.innerHTML = '<strong>' + (job.resolutionScale < 1 ? 'Preview' : 'Stable') + '</strong> render complete';
      } catch (error) {
        setStatus(String(error.message || error));
      } finally {
        state.renderInFlight = false;
        if (state.pendingRender) {
          queueMicrotask(() => {
            void flushRenderQueue();
          });
        } else {
          setLoading(false, 'idle');
        }
      }
    }

    function resetViewFromScene() {
      if (!state.scene) {
        return;
      }
      state.yaw = state.scene.default_yaw;
      state.pitch = state.scene.default_pitch;
      state.roll = state.scene.default_roll;
      state.radius = state.scene.default_radius;
      state.centerX = 0;
      state.centerY = 0;
      state.centerZ = 0;
      state.focalScale = state.scene.default_focal_scale;
      state.timeValue = state.scene.time_min;
      el.radiusSlider.min = state.scene.min_radius;
      el.radiusSlider.max = state.scene.max_radius;
      el.radiusSlider.value = state.radius;
      el.focalSlider.value = state.focalScale;
      el.timeSlider.min = state.scene.time_min;
      el.timeSlider.max = state.scene.time_max;
      el.timeSlider.value = state.timeValue;
      el.timeValue.textContent = Number(state.timeValue).toFixed(2);
      updateOverlayLabels();
    }

    function applyScene(scene) {
      state.scene = scene;
      state.sceneId = scene.scene_id || state.sceneId || 'latest';
      state.ready = true;
      state.pendingRender = null;
      clearTimeout(state.fullRenderTimer);
      if (state.interactiveRenderRaf !== null) {
        cancelAnimationFrame(state.interactiveRenderRaf);
        state.interactiveRenderRaf = null;
      }
      resetViewFromScene();
      el.factSource.textContent = scene.source_label;
      el.factFrames.textContent = scene.static_scene ? 'static' : String(scene.num_frames);
      el.factResolution.textContent = scene.width + ' x ' + scene.height;
      el.factScale.textContent = scene.scene_scale.toFixed(2);
      if (el.sceneSelect.value !== state.sceneId) {
        el.sceneSelect.value = state.sceneId;
      }
      el.hintBox.textContent = 'Scene ready. Drag in the viewport to orbit. Shift-drag or right-drag pans the pivot. Mouse wheel changes radius. Space toggles playback.';
      setStatus('Scene reconstructed. Ready for live viewpoint queries.');
    }

    function updateOverlayLabels() {
      el.badgeRight.textContent = 'mode ' + (state.viewerMode === 'Input Camera' ? 'input' : 'orbit') + ' | modality ' + state.modality;
      el.timeValue.textContent = Number(state.timeValue).toFixed(2);
    }

    async function fetchExamples() {
      const response = await fetch('/api/examples');
      const payload = await response.json();
      state.examples = payload.examples || [];
      el.exampleSelect.innerHTML = '';
      state.examples.forEach((example, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = example.name + ' [' + example.scene_type + ']';
        el.exampleSelect.appendChild(option);
      });
      if (state.examples.length > 0) {
        const example = state.examples[0];
        el.pathInput.value = example.file;
        el.sceneType.value = example.scene_type;
      }
    }

    async function fetchScenes(preferredSceneId) {
      const response = await fetch('/api/scenes', { cache: 'no-store' });
      const payload = await response.json();
      state.scenes = payload.scenes || [];
      el.sceneSelect.innerHTML = '';
      if (state.scenes.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No scene available';
        el.sceneSelect.appendChild(option);
        state.sceneId = '';
        return '';
      }

      state.scenes.forEach((record) => {
        const option = document.createElement('option');
        option.value = record.scene_id;
        option.textContent = record.label;
        el.sceneSelect.appendChild(option);
      });

      const preferred = preferredSceneId || state.sceneId || payload.active_id || state.scenes[state.scenes.length - 1].scene_id;
      const selected = state.scenes.some((record) => record.scene_id === preferred)
        ? preferred
        : state.scenes[state.scenes.length - 1].scene_id;
      state.sceneId = selected;
      el.sceneSelect.value = selected;
      return selected;
    }

    async function fetchSceneIfAvailable(sceneId) {
      const selectedSceneId = sceneId || state.sceneId;
      const suffix = selectedSceneId ? '?scene_id=' + encodeURIComponent(selectedSceneId) : '';
      const response = await fetch('/api/scene' + suffix, { cache: 'no-store' });
      const payload = await response.json();
      if (payload.ready) {
        applyScene(payload.scene);
        queueFullRender(10);
      }
    }

    async function reconstructScene() {
      const sourcePath = el.pathInput.value.trim();
      if (!sourcePath) {
        setStatus('Please provide a local path or choose an example.');
        return;
      }
      stopPlayback(false);
      setLoading(true, 'reconstructing');
      setStatus('Reconstructing the 4D scene. This runs the feed-forward reconstructor once.');
      const startedAt = performance.now();
      try {
        const response = await fetch('/api/reconstruct', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_path: sourcePath,
            scene_type: el.sceneType.value,
          }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || 'Reconstruction failed.');
        }
        await fetchScenes(payload.scene.scene_id || 'latest');
        applyScene(payload.scene);
        setRenderInfo('reconstruction ' + payload.elapsed_s.toFixed(2) + ' s');
        setStatus('Reconstruction finished in ' + payload.elapsed_s.toFixed(2) + ' s. Rendering first view.');
        queueFullRender(20);
      } catch (error) {
        setStatus(String(error.message || error));
      } finally {
        const elapsed = ((performance.now() - startedAt) / 1000).toFixed(2);
        el.timingLine.textContent = 'last reconstruction ' + elapsed + ' s';
        setLoading(false, 'idle');
      }
    }

    function queueFullRender(delayMs) {
      clearTimeout(state.fullRenderTimer);
      state.fullRenderTimer = setTimeout(() => requestRender(1.0), delayMs);
    }

    function queueInteractiveRender(previewScale = 0.55) {
      if (state.interactiveRenderRaf === null) {
        state.interactiveRenderRaf = requestAnimationFrame(() => {
          state.interactiveRenderRaf = null;
          requestRender(previewScale);
        });
      }
      if (!state.playing) {
        queueFullRender(140);
      }
    }

    function stopPlayback(renderStable = true) {
      if (state.playbackRaf !== null) {
        cancelAnimationFrame(state.playbackRaf);
        state.playbackRaf = null;
      }
      state.playbackLastTs = null;
      state.lastPlaybackEnqueueAt = 0;
      state.playing = false;
      el.playBtn.textContent = 'Play';
      if (renderStable) {
        queueFullRender(40);
      }
    }

    function playbackStep(ts) {
      if (!state.playing || !state.scene) {
        return;
      }
      if (state.playbackLastTs === null) {
        state.playbackLastTs = ts;
      }

      const deltaSec = Math.min((ts - state.playbackLastTs) / 1000, 0.1);
      state.playbackLastTs = ts;

      const span = Math.max(state.scene.time_max - state.scene.time_min, 1e-6);
      const loopDurationSec = Math.max(span / 16, 4.5);
      state.timeValue += deltaSec * (span / loopDurationSec);
      if (state.timeValue > state.scene.time_max) {
        state.timeValue = state.scene.time_min + (state.timeValue - state.scene.time_max);
      }

      el.timeSlider.value = state.timeValue;
      updateOverlayLabels();

      if (ts - state.lastPlaybackEnqueueAt >= 45) {
        state.lastPlaybackEnqueueAt = ts;
        requestRender(0.62);
      }

      state.playbackRaf = requestAnimationFrame(playbackStep);
    }

    function togglePlayback() {
      if (!state.ready || !state.scene || state.scene.static_scene) {
        return;
      }
      if (state.playing) {
        stopPlayback(true);
        return;
      }
      clearTimeout(state.fullRenderTimer);
      state.playing = true;
      el.playBtn.textContent = 'Pause';
      state.playbackLastTs = null;
      state.lastPlaybackEnqueueAt = 0;
      state.playbackRaf = requestAnimationFrame(playbackStep);
    }

    el.useExampleBtn.addEventListener('click', () => {
      const index = Number(el.exampleSelect.value || 0);
      const example = state.examples[index];
      if (!example) {
        return;
      }
      el.pathInput.value = example.file;
      el.sceneType.value = example.scene_type;
      setStatus('Example selected. Reconstruct when ready.');
    });

    el.reconstructBtn.addEventListener('click', reconstructScene);
    el.sceneSelect.addEventListener('change', () => {
      const sceneId = el.sceneSelect.value;
      if (!sceneId) {
        return;
      }
      stopPlayback(false);
      state.sceneId = sceneId;
      void fetchSceneIfAvailable(sceneId);
    });
    el.resetViewBtn.addEventListener('click', () => {
      resetViewFromScene();
      queueFullRender(30);
    });
    el.playBtn.addEventListener('click', togglePlayback);

    el.viewerMode.addEventListener('change', () => {
      state.viewerMode = el.viewerMode.value;
      updateOverlayLabels();
      queueFullRender(20);
    });
    el.modalitySelect.addEventListener('change', () => {
      state.modality = el.modalitySelect.value;
      updateOverlayLabels();
      queueFullRender(10);
    });
    el.radiusSlider.addEventListener('input', () => {
      state.radius = Number(el.radiusSlider.value);
      queueInteractiveRender();
    });
    el.focalSlider.addEventListener('input', () => {
      state.focalScale = Number(el.focalSlider.value);
      queueInteractiveRender();
    });
    el.maskThreshold.addEventListener('input', () => {
      state.maskThreshold = Number(el.maskThreshold.value);
      if (state.modality === 'mask') {
        queueInteractiveRender();
      }
    });
    el.timeSlider.addEventListener('input', () => {
      state.timeValue = Number(el.timeSlider.value);
      updateOverlayLabels();
      queueInteractiveRender();
    });

    el.viewport.addEventListener('contextmenu', (event) => event.preventDefault());
    el.viewport.addEventListener('pointerdown', (event) => {
      if (!state.ready) {
        return;
      }
      state.dragging = true;
      state.dragMode = event.button === 2 || event.shiftKey ? 'pan' : 'orbit';
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      el.viewport.setPointerCapture(event.pointerId);
    });
    el.viewport.addEventListener('pointermove', (event) => {
      if (!state.dragging) {
        return;
      }
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      if (state.dragMode === 'orbit') {
        state.yaw += dx * 0.32;
        state.pitch = clamp(state.pitch - dy * 0.22, -89, 89);
      } else {
        movePivot(-dx * 0.8, dy * 0.8, 0);
      }
      queueInteractiveRender();
    });
    function endDrag(event) {
      if (!state.dragging) {
        return;
      }
      state.dragging = false;
      if (event && event.pointerId !== undefined) {
        try {
          el.viewport.releasePointerCapture(event.pointerId);
        } catch (error) {
        }
      }
      queueFullRender(20);
    }
    el.viewport.addEventListener('pointerup', endDrag);
    el.viewport.addEventListener('pointercancel', endDrag);
    el.viewport.addEventListener('wheel', (event) => {
      if (!state.ready || !state.scene) {
        return;
      }
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.06 : 0.94;
      state.radius = clamp(state.radius * factor, state.scene.min_radius, state.scene.max_radius);
      el.radiusSlider.value = state.radius;
      queueInteractiveRender();
    }, { passive: false });

    window.addEventListener('keydown', (event) => {
      if (event.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) {
        return;
      }
      if (event.code === 'Space') {
        event.preventDefault();
        togglePlayback();
        return;
      }
      if (!state.ready) {
        return;
      }
      if (event.key === 'r') {
        resetViewFromScene();
        queueFullRender(20);
        return;
      }
      let handled = true;
      if (event.key === 'w') movePivot(0, 0, 1);
      else if (event.key === 's') movePivot(0, 0, -1);
      else if (event.key === 'a') movePivot(-1, 0, 0);
      else if (event.key === 'd') movePivot(1, 0, 0);
      else if (event.key === 'q') movePivot(0, 1, 0);
      else if (event.key === 'e') movePivot(0, -1, 0);
      else handled = false;
      if (handled) {
        queueInteractiveRender();
      }
    });

    updateOverlayLabels();
    fetchExamples().then(() => fetchScenes()).then((sceneId) => fetchSceneIfAvailable(sceneId)).catch((error) => {
      setStatus(String(error.message || error));
    });
    window.setInterval(() => {
      fetchScenes(state.sceneId).catch(() => {});
    }, 4000);
  </script>
</body>
</html>
"""


server_app = FastAPI(title="NeoVerse Interactive Viewer")


@server_app.get("/", response_class=HTMLResponse)
async def viewer_home():
    return HTMLResponse(VIEWER_HTML)


@server_app.get("/healthz")
async def healthz():
    return {"ok": True, "device": device}


@server_app.get("/api/examples")
async def api_examples():
    return {
        "examples": [
            {
                "name": example["name"],
                "file": example["file"],
                "scene_type": example.get("scene_type", "General scene"),
            }
            for example in _examples
        ]
    }


@server_app.get("/api/scene")
async def api_scene(scene_id: str = ""):
    with VIEWER_LOCK:
        try:
            record = _get_scene_record_unlocked(scene_id or None)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Scene not found: {scene_id}") from exc
        ready = record is not None
        scene = _public_scene_record(record)["meta"] if record is not None else None
    return {"ready": ready, "scene": scene}


@server_app.get("/api/scenes")
async def api_scenes():
    with VIEWER_LOCK:
        records = [_public_scene_record(record) for record in _ordered_scene_records_unlocked()]
        active_id = SCENE_REGISTRY.get("active_id")
    return {"ready": bool(records), "active_id": active_id, "scenes": records}


@server_app.post("/api/reconstruct")
async def api_reconstruct(request: Request):
    payload = await request.json()
    source_path = str(payload.get("source_path", "")).strip()
    scene_type = payload.get("scene_type", "General scene")
    if not source_path:
        raise HTTPException(status_code=400, detail="source_path is required.")

    source_paths = [part.strip() for part in source_path.split(",") if part.strip()]
    started_at = time.perf_counter()
    try:
        state = _load_media_from_paths(source_paths, scene_type)
        state["source_label"] = os.path.basename(source_paths[0])
        with VIEWER_LOCK:
            _, _, meta = _run_reconstruction_core(state, source="api")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "scene": meta,
        "elapsed_s": float(time.perf_counter() - started_at),
    }


@server_app.get("/api/render")
async def api_render(
    scene_id: str = "",
    viewer_mode: str = "Orbit Viewer",
    modality: str = "rgb",
    time_value: float = 0.0,
    yaw: float = 0.0,
    pitch: float = -10.0,
    roll: float = 0.0,
    radius: float = 1.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    focal_scale: float = 1.0,
    mask_threshold: float = 0.95,
    resolution_scale: float = 1.0,
):
    with VIEWER_LOCK:
        try:
            record = _get_scene_record_unlocked(scene_id or None)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Scene not found: {scene_id}") from exc
        if record is None:
            raise HTTPException(status_code=400, detail="No reconstructed scene is available yet.")
        state = record["state"]

        started_at = time.perf_counter()
        try:
            image, status, meta, _ = _render_view_image(
                state,
                viewer_mode=viewer_mode,
                time_value=time_value,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                radius=radius,
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                focal_scale=focal_scale,
                modality=modality,
                mask_threshold=mask_threshold,
                resolution_scale=resolution_scale,
                include_scene_glb=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        render_ms = (time.perf_counter() - started_at) * 1000.0

    image_bytes, media_type = _encode_image_bytes(image, meta["modality"])
    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={
            "X-Render-Ms": f"{render_ms:.1f}",
            "X-Status": status,
            "X-Scene-Id": record["scene_id"],
            "X-Camera-Pos": ", ".join(f"{v:.3f}" for v in meta["camera_position"]),
            "X-Frame-Index": str(meta["frame_index"]),
            "X-Timestamp": f"{meta['timestamp']:.3f}",
            "X-Render-Size": f"{meta['render_width']}x{meta['render_height']}",
        },
    )


demo.queue(max_size=5)
server_app = gr.mount_gradio_app(
    server_app,
    demo,
    path="/gradio",
    server_name=args.server_name,
    server_port=args.server_port,
    show_error=True,
)


if __name__ == "__main__":
    if args.share:
        print("`--share` is not supported by the lightweight root viewer. Use the local URL or open `/gradio` manually.")
    uvicorn.run(server_app, host=args.server_name, port=args.server_port, log_level="info")
