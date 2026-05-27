"""Job runners that wrap the `tesis_ac` pipeline with event streaming.

Each runner exposes one method that takes a job id, the parameters and the
event bus. It runs in a background thread (so it does not block the asyncio
event loop) and publishes `JobEvent`s as the pipeline progresses.

The implementation does NOT duplicate logic: it calls the same classes used
by `cli.py` and `validate_quinquenal_*.py`, so what runs from the UI is the
same code that produces the numbers reported in the thesis.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import JobEvent, JobEventLevel, JobKind
from .ws import EventBus


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ensure_tesis_ac_on_path(project_root: Path) -> None:
    src = project_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


class JobRunner:
    """Schedule jobs and publish their events on the shared bus."""

    def __init__(self, project_root: Path, bus: EventBus) -> None:
        self.project_root = project_root
        self.bus = bus

    # ─── Dispatch ────────────────────────────────────────────────────────

    async def submit(self, kind: JobKind, parameters: dict[str, Any]) -> str:
        job_id = new_job_id()
        loop = asyncio.get_running_loop()

        async def emit(level: JobEventLevel, message: str, *,
                       step: str | None = None,
                       payload: dict[str, Any] | None = None) -> None:
            event = JobEvent(
                job_id=job_id,
                ts=_now(),
                level=level,
                step=step,
                message=message,
                payload=payload,
            )
            await self.bus.publish(event)

        def emit_sync(level: JobEventLevel, message: str, *,
                      step: str | None = None,
                      payload: dict[str, Any] | None = None) -> None:
            future = asyncio.run_coroutine_threadsafe(
                emit(level, message, step=step, payload=payload),
                loop,
            )
            try:
                future.result(timeout=5.0)
            except Exception:
                pass

        await emit(
            JobEventLevel.info,
            f"Job {job_id} aceptado: {kind.value}",
            payload={"parameters": parameters},
        )

        async def runner() -> None:
            try:
                if kind == JobKind.classify_image:
                    await asyncio.to_thread(
                        self._classify_image, job_id, parameters, emit_sync
                    )
                else:
                    emit_sync(
                        JobEventLevel.error,
                        f"Job no soportado: {kind.value}",
                    )
            except Exception as exc:  # noqa: BLE001
                emit_sync(
                    JobEventLevel.error,
                    f"Excepción no controlada: {exc}",
                    payload={"traceback": traceback.format_exc()},
                )
            finally:
                emit_sync(JobEventLevel.done, "Job finalizado")

        asyncio.create_task(runner())
        return job_id

    # ─── Implementations ─────────────────────────────────────────────────

    def _classify_image(
        self,
        job_id: str,
        parameters: dict[str, Any],
        emit: "callable",
    ) -> None:
        """Run the full E1 + E2 pipeline on a single raw image.

        Parameters
        ----------
        parameters['image']: str
            Path (absolute or relative to project root) of the PNG to classify.
        parameters['sample_size']: int, default 5000
            SVM training subsample size.
        """

        # ── Step 0: resolve inputs and import the package
        image_param = parameters.get("image")
        if not image_param:
            emit(JobEventLevel.error, "Falta parámetro 'image'")
            return

        image_path = Path(image_param)
        if not image_path.is_absolute():
            image_path = self.project_root / image_path
        if not image_path.is_file():
            emit(JobEventLevel.error, f"Imagen no encontrada: {image_path}")
            return

        sample_size = int(parameters.get("sample_size", 5000))
        _ensure_tesis_ac_on_path(self.project_root)

        emit(
            JobEventLevel.info,
            f"Procesando {image_path.name}",
            step="setup",
            payload={"path": str(image_path), "sample_size": sample_size},
        )

        from tesis_ac.pipeline.feature_extraction import (
            FeatureExtractor,
            FeaturePreprocessor,
            load_image,
        )
        from tesis_ac.pipeline.clustering import SatelliteImageProcessor

        # ── Step 1: load image
        t0 = time.perf_counter()
        image = load_image(str(image_path))
        emit(
            JobEventLevel.metric,
            f"Imagen cargada en {(time.perf_counter() - t0) * 1000:.0f} ms",
            step="load_image",
            payload={
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            },
        )

        # ── Step 2: features
        t1 = time.perf_counter()
        extractor = FeatureExtractor(extract_full=True)
        X, feature_names, original_shape = extractor.extract_all_features(image)
        elapsed_features = (time.perf_counter() - t1) * 1000
        emit(
            JobEventLevel.metric,
            f"Extracción de {len(feature_names)} features en {elapsed_features:.0f} ms",
            step="features",
            payload={
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "X_shape": list(X.shape),
                "X_dtype": str(X.dtype),
                "elapsed_ms": int(elapsed_features),
            },
        )

        # ── Step 3: scaler + PCA
        t2 = time.perf_counter()
        preprocessor = FeaturePreprocessor(n_components=8)
        X_pca = preprocessor.fit_transform(X)
        variance = float(preprocessor.get_variance_explained())
        elapsed_pca = (time.perf_counter() - t2) * 1000
        emit(
            JobEventLevel.metric,
            f"PCA(8) explicó {variance * 100:.1f}% de varianza",
            step="pca",
            payload={
                "X_pca_shape": list(X_pca.shape),
                "explained_variance_ratio": variance,
                "elapsed_ms": int(elapsed_pca),
            },
        )

        # ── Step 4: K-Means + SVM
        t3 = time.perf_counter()
        processor = SatelliteImageProcessor(
            n_clusters_list=[2], svm_kernel="linear", random_state=42
        )
        processor.fit(X_pca, sample_size=sample_size)
        prediction_maps = processor.transform(X_pca, original_shape, method="all")
        elapsed_classify = (time.perf_counter() - t3) * 1000
        svm_map = prediction_maps.get("svm_2_classes")
        clusters = processor.clusterer.clustering_results[2]
        classifier_acc = processor.classifier.classification_results[2][
            "accuracy"
        ]
        emit(
            JobEventLevel.metric,
            f"K-Means + SVM en {elapsed_classify:.0f} ms",
            step="clustering",
            payload={
                "inertia": float(clusters["inertia"]),
                "distribution": clusters["distribution"].tolist(),
                "svm_test_accuracy": float(classifier_acc),
                "elapsed_ms": int(elapsed_classify),
            },
        )

        # ── Step 5: label standardisation (NDVI heuristic)
        t4 = time.perf_counter()
        normalised = _standardise_with_ndvi(svm_map, image)
        elapsed_std = (time.perf_counter() - t4) * 1000
        urban_pct = float(np.mean(normalised == 1) * 100)
        emit(
            JobEventLevel.metric,
            f"Standardización completada — urbano = {urban_pct:.2f}%",
            step="standardize",
            payload={
                "urban_percentage": urban_pct,
                "flipped": bool(normalised.flags.writeable
                                and not np.array_equal(svm_map, normalised)),
                "elapsed_ms": int(elapsed_std),
            },
        )

        # ── Step 6: persist
        out_dir = self.project_root / "desktop-app" / "runs" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "svm_2_classes.npy", svm_map)
        np.save(out_dir / "standardized.npy", normalised)

        manifest = {
            "job_id": job_id,
            "image": str(image_path),
            "image_name": image_path.name,
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "pca_variance_explained": variance,
            "svm_test_accuracy": float(classifier_acc),
            "kmeans_inertia": float(clusters["inertia"]),
            "urban_percentage": urban_pct,
            "timings_ms": {
                "load_image": int((time.perf_counter() - t0) * 1000),
                "features": int(elapsed_features),
                "pca": int(elapsed_pca),
                "clustering": int(elapsed_classify),
                "standardize": int(elapsed_std),
            },
            "outputs": {
                "svm_2_classes": str(out_dir / "svm_2_classes.npy"),
                "standardized": str(out_dir / "standardized.npy"),
            },
        }
        with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        emit(
            JobEventLevel.artifact,
            "Artefactos guardados",
            step="persist",
            payload={
                "manifest": str(out_dir / "manifest.json"),
                "out_dir": str(out_dir),
                "urban_percentage": urban_pct,
            },
        )


# ─── Helper: NDVI label standardisation ──────────────────────────────────


def _standardise_with_ndvi(binary: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """Return a copy of `binary` where 1 corresponds to urban.

    Uses the same NDVI heuristic implemented in `pipeline/cli.py`: the
    cluster with lower mean NDVI is considered urban (less vegetation).
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        return binary.copy()

    out = binary.copy()
    r = image_rgb[:, :, 0].astype(float)
    g = image_rgb[:, :, 1].astype(float)
    ndvi = (g - r) / (g + r + 1e-8)

    mask0 = out == 0
    mask1 = out == 1
    if mask0.sum() < 100 or mask1.sum() < 100:
        return out

    if float(np.mean(ndvi[mask0])) < float(np.mean(ndvi[mask1])):
        out[:] = 1 - out
    return out
