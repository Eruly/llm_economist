"""Helpers for persisting and restoring simulation history metadata."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CONFIG_FILENAME = "simulation_config.json"


def _to_serialisable(payload: Any) -> Any:
    """Convert values that are not directly serialisable into JSON-friendly ones."""

    if isinstance(payload, Namespace):
        return {key: _to_serialisable(value) for key, value in vars(payload).items()}

    if isinstance(payload, (list, tuple)):
        return [_to_serialisable(item) for item in payload]

    if isinstance(payload, Path):
        return str(payload)

    try:  # Handle numpy scalar types without importing globally
        import numpy as np  # type: ignore

        if isinstance(payload, np.generic):
            return payload.item()
        if isinstance(payload, np.ndarray):
            return [_to_serialisable(item) for item in payload.tolist()]
    except Exception:  # pragma: no cover - numpy may be unavailable
        pass

    if hasattr(payload, "tolist") and not isinstance(payload, (str, bytes)):
        try:
            return payload.tolist()
        except TypeError:
            pass

    return payload


def determine_history_base_dir(path_str: Optional[str]) -> Optional[Path]:
    """Return the directory that should host shared history artefacts."""

    if not path_str:
        return None

    candidate = Path(path_str).expanduser()
    if candidate.is_dir():
        return candidate
    return candidate.parent if candidate.parent.exists() else candidate.parent


def metadata_path_for(base_dir: Path) -> Path:
    """Return the path where shared simulation metadata is stored."""

    return base_dir / CONFIG_FILENAME


def load_simulation_metadata(path_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load persisted simulation metadata if present."""

    base_dir = determine_history_base_dir(path_str)
    if base_dir is None:
        return None

    meta_path = metadata_path_for(base_dir)
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def save_simulation_metadata(path_str: Optional[str], payload: Dict[str, Any]) -> Optional[Path]:
    """Persist metadata to the directory associated with ``path_str``.

    Returns the resolved path when successful so callers can log it.
    """

    base_dir = determine_history_base_dir(path_str)
    if base_dir is None:
        return None

    base_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metadata_path_for(base_dir)

    serialisable_payload = {
        key: _to_serialisable(value)
        for key, value in payload.items()
    }

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(serialisable_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    return meta_path


def merge_args_from_metadata(args: Namespace, metadata: Dict[str, Any], *, override_keys: Optional[Iterable[str]] = None) -> None:
    """Update ``args`` in place using values stored in ``metadata``.

    If ``override_keys`` is provided, only those attributes are updated.
    """

    args_payload = metadata.get("args")
    if not isinstance(args_payload, dict):
        return

    if override_keys is None:
        override_keys = args_payload.keys()

    for key in override_keys:
        if key not in args_payload:
            continue
        setattr(args, key, args_payload[key])
