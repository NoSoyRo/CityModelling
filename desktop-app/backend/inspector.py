"""On-demand inspection of pipeline artifacts.

Given an absolute path, this module decides how to open the file and what
to surface to the UI:

- `.npy`        → shape, dtype, statistics, optional 2-bit preview.
- `.json`       → parsed object.
- `.pkl`        → recursive type tree; the WoE pickle gets a specialised
                  reader.
- `.png`        → embedded thumbnail.

Everything is bounded: previews are downsampled to keep payloads small.
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    ArtifactRef,
    InspectionResponse,
    JsonInspection,
    NumpyStats,
    PickleNode,
    WoeModelInspection,
    WoeVariableSummary,
)


# The WoE pickle stores objects from the `tesis_ac` package, so we must be
# able to import it before unpickling.

def ensure_tesis_ac_on_path(project_root: Path) -> None:
    src = project_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# ─── Numpy ────────────────────────────────────────────────────────────────

_PREVIEW_MAX_SIDE = 600


def _downsample_2d(arr: np.ndarray, max_side: int) -> np.ndarray:
    """Decimation downsampler (no interpolation, no scipy)."""
    h, w = arr.shape[:2]
    step_h = max(1, h // max_side)
    step_w = max(1, w // max_side)
    return arr[::step_h, ::step_w]


def _array_to_png_b64(arr: np.ndarray) -> str | None:
    """Render a 2-D numpy array as a PNG and return base64.

    Uses PIL which is already a dependency of the project.
    """
    try:
        from PIL import Image
    except Exception:
        return None

    work = arr
    if work.ndim == 3 and work.shape[2] not in (1, 3, 4):
        return None
    if work.ndim > 3:
        return None

    if work.ndim == 2:
        work = _downsample_2d(work, _PREVIEW_MAX_SIDE)
        unique = np.unique(work)
        if unique.size <= 2 and set(unique.tolist()).issubset({0, 1}):
            # Binary map: render as red urban / dark green non-urban.
            rgb = np.zeros((*work.shape, 3), dtype=np.uint8)
            rgb[work == 0] = (40, 80, 40)
            rgb[work == 1] = (220, 60, 60)
            img = Image.fromarray(rgb, mode="RGB")
        else:
            vmin, vmax = float(np.min(work)), float(np.max(work))
            if vmax - vmin < 1e-12:
                norm = np.zeros_like(work, dtype=np.uint8)
            else:
                norm = ((work - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            img = Image.fromarray(norm, mode="L")
    elif work.ndim == 3:
        work = _downsample_2d(work, _PREVIEW_MAX_SIDE)
        if work.dtype != np.uint8:
            work = work.astype(np.uint8)
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[work.shape[2]]
        if mode == "L":
            work = work[:, :, 0]
        img = Image.fromarray(work, mode=mode)
    else:
        return None

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def inspect_numpy(path: Path) -> tuple[NumpyStats, str | None]:
    arr = np.load(path, allow_pickle=False)
    stats = _array_stats(arr)
    preview = None
    if arr.ndim in (2, 3):
        preview = _array_to_png_b64(arr)
    return stats, preview


def _array_stats(arr: np.ndarray) -> NumpyStats:
    is_numeric = np.issubdtype(arr.dtype, np.number) or np.issubdtype(
        arr.dtype, np.bool_
    )
    distribution: dict[str, int] | None = None
    unique_count: int | None = None

    if is_numeric:
        flat = arr.ravel()
        amin = float(np.min(flat))
        amax = float(np.max(flat))
        amean = float(np.mean(flat))
    else:
        amin = amax = amean = None

    # For binary / categorical maps, expose the histogram.
    if is_numeric and arr.ndim <= 3 and arr.size <= 50_000_000:
        unique, counts = np.unique(arr, return_counts=True)
        if unique.size <= 16:
            distribution = {str(int(u) if u == int(u) else u): int(c)
                            for u, c in zip(unique.tolist(), counts.tolist())}
            unique_count = int(unique.size)

    return NumpyStats(
        shape=tuple(int(s) for s in arr.shape),
        dtype=str(arr.dtype),
        size=int(arr.size),
        nbytes=int(arr.nbytes),
        min=amin,
        max=amax,
        mean=amean,
        unique_count=unique_count,
        distribution=distribution,
    )


# ─── JSON ─────────────────────────────────────────────────────────────────


def inspect_json(path: Path) -> JsonInspection:
    with path.open("r", encoding="utf-8") as f:
        return JsonInspection(content=json.load(f))


# ─── Pickle (generic) ─────────────────────────────────────────────────────


def _summarise(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, (list, tuple)):
        return f"len={len(value)}"
    if isinstance(value, dict):
        return f"keys={len(value)}"
    if isinstance(value, (int, float, bool, str)):
        rendered = repr(value)
        return rendered if len(rendered) < 80 else rendered[:77] + "..."
    if value is None:
        return "None"
    return ""


def _build_tree(name: str, value: Any, depth: int = 0) -> PickleNode:
    type_name = type(value).__name__
    node = PickleNode(
        name=name,
        type=type_name,
        summary=_summarise(value),
        children=[],
    )
    if depth >= 4:
        return node

    if isinstance(value, dict):
        for k, v in list(value.items())[:64]:
            node.children.append(_build_tree(str(k), v, depth + 1))
    elif isinstance(value, (list, tuple)) and not isinstance(value, str):
        for i, v in enumerate(value[:32]):
            node.children.append(_build_tree(f"[{i}]", v, depth + 1))
    elif is_dataclass(value):
        for fname in value.__dataclass_fields__:
            try:
                node.children.append(
                    _build_tree(fname, getattr(value, fname), depth + 1)
                )
            except Exception:
                pass
    elif hasattr(value, "__dict__") and not isinstance(
        value, (int, float, str, bool)
    ):
        for k, v in list(vars(value).items())[:32]:
            if k.startswith("_"):
                continue
            node.children.append(_build_tree(k, v, depth + 1))
    return node


def inspect_pickle(path: Path) -> tuple[PickleNode, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    tree = _build_tree(path.stem, obj)
    return tree, obj


# ─── Pickle (specialised WoE reader) ──────────────────────────────────────


def _classify_iv(iv: float) -> str:
    if iv < 0.02:
        return "no_predictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "very_strong"


def inspect_woe(obj: Any) -> WoeModelInspection | None:
    """Recognise the structure produced by `train_woe_pooled.py`.

    The pickle stored by that script is::

        {
            "woe_calculator": WoECalculator,
            "metadata": {...},
        }

    where `WoECalculator.woe_results` is a `dict[str, WoEResult]`.
    """

    if not isinstance(obj, dict) or "woe_calculator" not in obj:
        return None

    calc = obj["woe_calculator"]
    metadata = obj.get("metadata", {}) or {}
    results = getattr(calc, "woe_results", None)
    if not isinstance(results, dict):
        return None

    variables: list[WoeVariableSummary] = []
    for name, result in results.items():
        iv = float(getattr(result, "iv_total", 0.0))
        bins = getattr(result, "bins", np.array([]))
        woe_values = getattr(result, "woe_values", np.array([]))
        variables.append(
            WoeVariableSummary(
                name=str(name),
                iv_total=iv,
                iv_strength=_classify_iv(iv),  # type: ignore[arg-type]
                n_bins=int(woe_values.shape[0]) if hasattr(woe_values, "shape") else 0,
                bin_edges=[float(x) for x in np.asarray(bins).tolist()],
                woe_values=[float(x) for x in np.asarray(woe_values).tolist()],
                positive_samples=int(getattr(result, "n_positive_samples", 0)),
                negative_samples=int(getattr(result, "n_negative_samples", 0)),
            )
        )

    # Sort by descending IV so the most informative variables come first.
    variables.sort(key=lambda v: v.iv_total, reverse=True)

    return WoeModelInspection(
        trained_years=metadata.get("trained_years"),
        n_periods=metadata.get("n_periods"),
        total_transitions=metadata.get("total_transitions"),
        variables=variables,
    )


# ─── PNG ──────────────────────────────────────────────────────────────────


def inspect_image(path: Path) -> str | None:
    try:
        from PIL import Image
    except Exception:
        return None
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─── Public façade ────────────────────────────────────────────────────────


def inspect_artifact(ref: ArtifactRef, project_root: Path) -> InspectionResponse:
    path = Path(ref.path)
    notes: list[str] = []
    response = InspectionResponse(artifact=ref, notes=notes)

    if not path.exists():
        notes.append(f"Path no encontrado: {path}")
        return response

    suffix = path.suffix.lower()

    try:
        if suffix == ".npy":
            stats, preview = inspect_numpy(path)
            response.numpy = stats
            response.image_preview_b64 = preview
        elif suffix == ".json":
            response.json_content = inspect_json(path)
        elif suffix == ".pkl":
            ensure_tesis_ac_on_path(project_root)
            tree, raw = inspect_pickle(path)
            response.pickle_tree = tree
            woe = inspect_woe(raw)
            if woe is not None:
                response.woe = woe
                notes.append("Pickle reconocido como modelo WoE entrenado.")
        elif suffix == ".png":
            response.image_preview_b64 = inspect_image(path)
        else:
            notes.append(f"Tipo de archivo no inspeccionable: {suffix}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Error inspeccionando: {exc}")

    return response
