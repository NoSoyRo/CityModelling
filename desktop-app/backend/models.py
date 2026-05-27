"""Pydantic v2 schemas shared between backend endpoints and the frontend.

These models define the public contract of the application. Every payload
crossing the HTTP or WebSocket boundary is validated against one of them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── Catalog ───────────────────────────────────────────────────────────────

class ArtifactKind(str, Enum):
    """Recognised artifact kinds in the pipeline."""

    raw_image = "raw_image"
    standardized_map = "standardized_map"
    woe_model = "woe_model"
    validation_run = "validation_run"
    validation_result = "validation_result"
    prediction_map = "prediction_map"
    config = "config"
    figure = "figure"
    other = "other"


class StageId(str, Enum):
    """Pipeline stages as documented in `docs/pipeline.md`."""

    E0 = "E0"  # raw input
    E1 = "E1"  # feature extraction + clustering
    E2 = "E2"  # label standardization
    E3 = "E3"  # WoE training
    E4 = "E4"  # CA simulation
    E5 = "E5"  # validation metrics


class ArtifactRef(BaseModel):
    """Lightweight reference to a file living somewhere under `data/`.

    The reference holds enough metadata for the UI to render a row in the
    catalog without opening the file. Heavy inspection (statistics, previews)
    is requested on demand from `/api/inspect`.
    """

    id: str = Field(..., description="Stable identifier (relative path).")
    path: str = Field(..., description="Absolute path on disk.")
    kind: ArtifactKind
    stage: StageId
    size_bytes: int
    modified: datetime
    label: str
    tags: dict[str, str] = Field(default_factory=dict)


class CatalogResponse(BaseModel):
    artifacts: list[ArtifactRef]
    total: int
    scanned_paths: list[str]


# ─── Stage descriptors ─────────────────────────────────────────────────────

class StageDescriptor(BaseModel):
    """Static metadata about a pipeline stage shown in the sidebar."""

    id: StageId
    title: str
    one_liner: str
    inputs: list[str]
    outputs: list[str]
    code_module: str
    parameters: list[str] = Field(default_factory=list)


# ─── Inspection ────────────────────────────────────────────────────────────

class NumpyStats(BaseModel):
    shape: tuple[int, ...]
    dtype: str
    size: int
    nbytes: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    unique_count: int | None = None
    distribution: dict[str, int] | None = None


class JsonInspection(BaseModel):
    content: Any


class PickleNode(BaseModel):
    """One node in the recursive tree of a pickle inspection."""

    name: str
    type: str
    summary: str
    children: list["PickleNode"] = Field(default_factory=list)


PickleNode.model_rebuild()


class WoeVariableSummary(BaseModel):
    """Specialised summary for a `WoEResult` instance.

    The WoE pickle is the central model artifact. This summary exposes the
    information value (IV), the bin edges and the per-bin weights so the UI
    can plot them directly without loading the whole object.
    """

    name: str
    iv_total: float
    iv_strength: Literal[
        "no_predictive", "weak", "medium", "strong", "very_strong"
    ]
    n_bins: int
    bin_edges: list[float]
    woe_values: list[float]
    positive_samples: int
    negative_samples: int


class WoeModelInspection(BaseModel):
    trained_years: str | None
    n_periods: int | None
    total_transitions: int | None
    variables: list[WoeVariableSummary]


class InspectionResponse(BaseModel):
    artifact: ArtifactRef
    numpy: NumpyStats | None = None
    json_content: JsonInspection | None = None
    pickle_tree: PickleNode | None = None
    woe: WoeModelInspection | None = None
    image_preview_b64: str | None = None
    notes: list[str] = Field(default_factory=list)


# ─── Runs and metrics ──────────────────────────────────────────────────────

class ValidationRunSummary(BaseModel):
    """Compact summary of one `validation_quinquenal_*` directory."""

    window: str  # e.g. "2011-2016"
    fom: float
    kappa: float
    iou: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    quantity_disagreement: int
    allocation_disagreement: int
    urban_observed: int
    urban_predicted: int
    growth_observed: int
    growth_predicted: int
    threshold: float
    neighbor_weight: float


class MetricsDashboard(BaseModel):
    runs: list[ValidationRunSummary]
    aggregate: dict[str, float]  # mean FoM, Kappa, etc.


# ─── Jobs (live execution) ─────────────────────────────────────────────────

class JobKind(str, Enum):
    classify_image = "classify_image"


class JobRequest(BaseModel):
    kind: JobKind
    parameters: dict[str, Any] = Field(default_factory=dict)


class JobAck(BaseModel):
    job_id: str
    started_at: datetime
    kind: JobKind


class JobEventLevel(str, Enum):
    info = "info"
    debug = "debug"
    warn = "warn"
    error = "error"
    metric = "metric"
    artifact = "artifact"
    progress = "progress"
    done = "done"


class JobEvent(BaseModel):
    job_id: str
    ts: datetime
    level: JobEventLevel
    step: str | None = None
    message: str
    payload: dict[str, Any] | None = None
