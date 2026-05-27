"""Discovery of pipeline artifacts on disk.

The catalog scans a fixed set of directories under the project root and
classifies every file it finds into one of the recognised `ArtifactKind`s.
The result is a flat list of `ArtifactRef` that the UI can group and filter
client-side.

This module does not open any file: it only reads filesystem metadata.
Heavy inspection lives in `inspector.py`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .models import ArtifactKind, ArtifactRef, StageId


YEAR_REGEX = re.compile(r"(?P<year>\d{4})")


def _read_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read_mtime(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _extract_year(name: str) -> str | None:
    m = YEAR_REGEX.search(name)
    return m.group("year") if m else None


def _make_ref(
    root: Path,
    path: Path,
    kind: ArtifactKind,
    stage: StageId,
    label: str,
    **tags: str,
) -> ArtifactRef:
    rel = path.relative_to(root).as_posix()
    return ArtifactRef(
        id=rel,
        path=str(path),
        kind=kind,
        stage=stage,
        size_bytes=_read_size(path),
        modified=_read_mtime(path),
        label=label,
        tags={k: v for k, v in tags.items() if v is not None},
    )


def scan_raw(root: Path) -> list[ArtifactRef]:
    raw_dir = root / "data" / "raw"
    if not raw_dir.is_dir():
        return []
    refs: list[ArtifactRef] = []
    for p in sorted(raw_dir.glob("imagen_*.png")):
        year = _extract_year(p.name) or "?"
        refs.append(
            _make_ref(
                root,
                p,
                ArtifactKind.raw_image,
                StageId.E0,
                f"Imagen RGB {year}",
                year=year,
            )
        )
    return refs


def scan_standardized(root: Path) -> list[ArtifactRef]:
    base = root / "data" / "processed" / "standardized_maps"
    if not base.is_dir():
        return []
    refs: list[ArtifactRef] = []
    for p in sorted(base.glob("*.npy")):
        year = _extract_year(p.stem) or p.stem
        refs.append(
            _make_ref(
                root,
                p,
                ArtifactKind.standardized_map,
                StageId.E2,
                f"Mapa binario {year}",
                year=year,
            )
        )
    return refs


def scan_woe_models(root: Path) -> list[ArtifactRef]:
    base = root / "data" / "processed"
    if not base.is_dir():
        return []
    refs: list[ArtifactRef] = []
    for p in sorted(base.glob("woe_pooled_*.pkl")):
        refs.append(
            _make_ref(
                root,
                p,
                ArtifactKind.woe_model,
                StageId.E3,
                f"Modelo WoE — {p.stem.replace('woe_pooled_', '')}",
            )
        )
    return refs


def scan_validation_runs(root: Path) -> list[ArtifactRef]:
    base = root / "data" / "processed"
    if not base.is_dir():
        return []
    refs: list[ArtifactRef] = []
    for run_dir in sorted(base.glob("validation_quinquenal_*")):
        if not run_dir.is_dir():
            continue
        window = run_dir.name.replace("validation_quinquenal_", "").replace(
            "_v3_weighted", ""
        )
        # Directory itself
        refs.append(
            _make_ref(
                root,
                run_dir,
                ArtifactKind.validation_run,
                StageId.E4,
                f"Run quinquenal {window}",
                window=window,
            )
        )
        # JSON result
        result = run_dir / "validation_results.json"
        if result.is_file():
            refs.append(
                _make_ref(
                    root,
                    result,
                    ArtifactKind.validation_result,
                    StageId.E5,
                    f"Resultados {window}",
                    window=window,
                )
            )
        # Yearly predictions
        yearly_dir = run_dir / "yearly_predictions"
        if yearly_dir.is_dir():
            for p in sorted(yearly_dir.glob("*.npy")):
                year = _extract_year(p.stem) or p.stem
                refs.append(
                    _make_ref(
                        root,
                        p,
                        ArtifactKind.prediction_map,
                        StageId.E4,
                        f"Predicción {year} ({window})",
                        window=window,
                        year=year,
                    )
                )
        # Visualisations
        viz_dir = run_dir / "visualizations"
        if viz_dir.is_dir():
            for p in sorted(viz_dir.glob("*.png")):
                refs.append(
                    _make_ref(
                        root,
                        p,
                        ArtifactKind.figure,
                        StageId.E4,
                        f"Viz {p.stem} ({window})",
                        window=window,
                    )
                )
    return refs


def scan_configs(root: Path) -> list[ArtifactRef]:
    base = root / "data" / "processed"
    refs: list[ArtifactRef] = []
    for name in (
        "quinquenal_best_config.json",
        "ga_calibrated_params.json",
        "quinquenal_all_periods_summary.json",
    ):
        p = base / name
        if p.is_file():
            refs.append(
                _make_ref(
                    root, p, ArtifactKind.config, StageId.E4, p.stem
                )
            )
    return refs


def scan_all(root: Path) -> tuple[list[ArtifactRef], list[str]]:
    """Run every scanner and return the union plus the scanned paths."""

    scanned_paths = [
        "data/raw/",
        "data/processed/standardized_maps/",
        "data/processed/woe_pooled_*.pkl",
        "data/processed/validation_quinquenal_*/",
        "data/processed/*.json",
    ]
    refs = (
        scan_raw(root)
        + scan_standardized(root)
        + scan_woe_models(root)
        + scan_validation_runs(root)
        + scan_configs(root)
    )
    return refs, scanned_paths
