"""Aggregation of `validation_results.json` into a dashboard payload.

The structure of `validation_results.json` (as produced by the quinquennal
validation scripts) is::

    metadata.validation_period
    confusion_matrix.{TP, TN, FP, FN}
    pixel_metrics.{accuracy, iou, precision, recall, f1_score, kappa}
    fom_metrics.{fom, B_hits, C_misses, A_false_alarms}
    growth_statistics.{urban_start, urban_end_observed,
                       urban_end_predicted, growth_observed,
                       growth_predicted}

AC parameters (threshold, neighbor_weight) are not stored per-run; they
come from `data/processed/quinquenal_best_config.json` which describes
the configuration used across the five windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from .models import MetricsDashboard, ValidationRunSummary


def _safe(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_global_ac_params(project_root: Path) -> dict[str, float]:
    cfg = project_root / "data" / "processed" / "quinquenal_best_config.json"
    if not cfg.is_file():
        return {}
    try:
        with cfg.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    params = data.get("ac_parameters") or {}
    return {
        "threshold": float(params.get("threshold", 0.0)),
        "neighbor_weight": float(params.get("neighbor_weight", 0.0)),
    }


def _summarise_run(path: Path, global_ac: dict[str, float]) -> ValidationRunSummary | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    window = (
        _safe(data, "metadata", "validation_period")
        or path.parent.name.replace("validation_quinquenal_", "").replace(
            "_v3_weighted", ""
        )
    )

    # Pontius-style decomposition derived from the confusion matrix when not
    # stored explicitly: quantity = |FP - FN|; allocation = 2 * min(FP, FN).
    fp = int(_safe(data, "confusion_matrix", "FP", default=0))
    fn = int(_safe(data, "confusion_matrix", "FN", default=0))
    quantity = abs(fp - fn)
    allocation = 2 * min(fp, fn)

    return ValidationRunSummary(
        window=window,
        fom=float(_safe(data, "fom_metrics", "fom", default=0.0)),
        kappa=float(_safe(data, "pixel_metrics", "kappa", default=0.0)),
        iou=float(_safe(data, "pixel_metrics", "iou", default=0.0)),
        accuracy=float(_safe(data, "pixel_metrics", "accuracy", default=0.0)),
        precision=float(_safe(data, "pixel_metrics", "precision", default=0.0)),
        recall=float(_safe(data, "pixel_metrics", "recall", default=0.0)),
        f1=float(_safe(data, "pixel_metrics", "f1_score", default=0.0)),
        quantity_disagreement=quantity,
        allocation_disagreement=allocation,
        urban_observed=int(
            _safe(data, "growth_statistics", "urban_end_observed", default=0)
        ),
        urban_predicted=int(
            _safe(data, "growth_statistics", "urban_end_predicted", default=0)
        ),
        growth_observed=int(
            _safe(data, "growth_statistics", "growth_observed", default=0)
        ),
        growth_predicted=int(
            _safe(data, "growth_statistics", "growth_predicted", default=0)
        ),
        threshold=float(global_ac.get("threshold", 0.0)),
        neighbor_weight=float(global_ac.get("neighbor_weight", 0.0)),
    )


def build_metrics_dashboard(project_root: Path) -> MetricsDashboard:
    base = project_root / "data" / "processed"
    global_ac = _load_global_ac_params(project_root)
    runs: list[ValidationRunSummary] = []
    if base.is_dir():
        for run_dir in sorted(base.glob("validation_quinquenal_*")):
            result = run_dir / "validation_results.json"
            if result.is_file():
                summary = _summarise_run(result, global_ac)
                if summary is not None:
                    runs.append(summary)

    aggregate: dict[str, float] = {}
    if runs:
        for field in ("fom", "kappa", "iou", "accuracy", "f1"):
            aggregate[field] = float(
                round(mean(getattr(r, field) for r in runs), 4)
            )
    return MetricsDashboard(runs=runs, aggregate=aggregate)
