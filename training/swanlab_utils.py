import inspect
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf


NULL_STRINGS = {"", "none", "None", "null", "Null"}


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in NULL_STRINGS)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _env_or_cfg(env_name: str, cfg: Any, key: str, default: Any = None) -> Any:
    env_value = os.environ.get(env_name)
    if not _is_null(env_value):
        return env_value
    value = _cfg_get(cfg, key, default)
    return default if _is_null(value) else value


def _as_bool(value: Any, default: bool = False) -> bool:
    if _is_null(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _as_tags(value: Any) -> list[str] | None:
    if _is_null(value):
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item] or None
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if not _is_null(item)] or None
    return [str(value)]


def _sanitize_config(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


class NullExperimentLogger:
    def log(self, metrics: Mapping[str, Any], step: int | None = None):
        return None

    def finish(self):
        return None


class SwanLabExperimentLogger:
    def __init__(self, swanlab_module, run):
        self._swanlab = swanlab_module
        self._run = run

    def log(self, metrics: Mapping[str, Any], step: int | None = None):
        clean = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().cpu().item()
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif not isinstance(value, (int, float, bool)):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
            clean[str(key)] = value
        if clean:
            try:
                self._swanlab.log(clean, step=step)
            except TypeError:
                self._swanlab.log(clean)

    def finish(self):
        if self._run is not None and hasattr(self._run, "finish"):
            self._run.finish()
        else:
            self._swanlab.finish()


def init_swanlab_logger(
    cfg: Any,
    *,
    default_project: str,
    default_experiment_name: str | None = None,
    output_path: str | None = None,
):
    enabled = _as_bool(_env_or_cfg("SWANLAB_ENABLED", cfg, "swanlab_enabled", True), True)
    if not enabled:
        return NullExperimentLogger()

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("swanlab is required for experiment logging. Install it with `pip install swanlab`.") from exc

    api_key = os.environ.get("SWANLAB_API_KEY")
    if not _is_null(api_key):
        login_kwargs = _supported_kwargs(
            swanlab.login,
            {
                "api_key": api_key,
                "host": None if _is_null(os.environ.get("SWANLAB_HOST")) else os.environ.get("SWANLAB_HOST"),
                "web_host": None if _is_null(os.environ.get("SWANLAB_WEB_HOST")) else os.environ.get("SWANLAB_WEB_HOST"),
                "save": False,
            },
        )
        swanlab.login(**login_kwargs)

    logdir = _env_or_cfg("SWANLAB_LOGDIR", cfg, "swanlab_logdir", None)
    if _is_null(logdir) and not _is_null(output_path):
        logdir = os.path.join(str(output_path), "swanlog")

    init_kwargs = {
        "project": _env_or_cfg("SWANLAB_PROJECT", cfg, "swanlab_project", default_project),
        "experiment_name": _env_or_cfg(
            "SWANLAB_EXPERIMENT_NAME",
            cfg,
            "swanlab_experiment_name",
            default_experiment_name,
        ),
        "workspace": _env_or_cfg("SWANLAB_WORKSPACE", cfg, "swanlab_workspace", None),
        "mode": _env_or_cfg("SWANLAB_MODE", cfg, "swanlab_mode", "cloud"),
        "logdir": logdir,
        "tags": _as_tags(_env_or_cfg("SWANLAB_TAGS", cfg, "swanlab_tags", None)),
        "config": _sanitize_config(cfg),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if not _is_null(value)}
    if not _is_null(os.environ.get("SWANLAB_MODE")):
        init_kwargs.pop("mode", None)
    run = swanlab.init(**_supported_kwargs(swanlab.init, init_kwargs))
    return SwanLabExperimentLogger(swanlab, run)


def _supported_kwargs(fn, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}
