"""FastAPI application — `Querétaro Urban Lab` backend.

The HTTP surface is intentionally small. Every endpoint maps to a concrete
user action visible in the UI:

| Method  | Path                          | Action                              |
|---------|-------------------------------|-------------------------------------|
| GET     | /api/health                   | Health check                        |
| GET     | /api/stages                   | Static descriptor of the 6 stages   |
| GET     | /api/catalog                  | Discover artifacts under data/      |
| POST    | /api/inspect                  | Inspect one artifact in depth       |
| GET     | /api/metrics                  | Aggregated validation runs          |
| POST    | /api/jobs                     | Submit a new job (e.g. classify)    |
| GET     | /api/jobs/{job_id}/history    | Replay buffered events of a job     |
| WS      | /ws                           | Live event stream of all jobs       |

The frontend bundle, when present, is served from `/`.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .catalog import scan_all
from .inspector import inspect_artifact
from .metrics import build_metrics_dashboard
from .models import (
    ArtifactRef,
    CatalogResponse,
    InspectionResponse,
    JobAck,
    JobEvent,
    JobRequest,
    MetricsDashboard,
    StageDescriptor,
)
from .runner import JobRunner, _now
from .stages import STAGES
from .ws import get_bus


PROJECT_ROOT = Path(
    os.environ.get("TESIS_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()


app = FastAPI(
    title="Querétaro Urban Lab",
    version="0.1.0",
    description="UI for the WoE+AC urban growth model of the ZMQ.",
)

# Permissive CORS only for the dev frontend on 5173.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


_runner = JobRunner(PROJECT_ROOT, get_bus())


# ─── REST ─────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "time": _now().isoformat(),
    }


@app.get("/api/stages", response_model=list[StageDescriptor])
def stages() -> list[StageDescriptor]:
    return STAGES


@app.get("/api/catalog", response_model=CatalogResponse)
def catalog() -> CatalogResponse:
    refs, scanned = scan_all(PROJECT_ROOT)
    return CatalogResponse(
        artifacts=refs, total=len(refs), scanned_paths=scanned
    )


@app.post("/api/inspect", response_model=InspectionResponse)
def inspect(ref: ArtifactRef) -> InspectionResponse:
    return inspect_artifact(ref, PROJECT_ROOT)


@app.get("/api/metrics", response_model=MetricsDashboard)
def metrics() -> MetricsDashboard:
    return build_metrics_dashboard(PROJECT_ROOT)


@app.post("/api/jobs", response_model=JobAck)
async def submit_job(req: JobRequest) -> JobAck:
    job_id = await _runner.submit(req.kind, req.parameters)
    return JobAck(job_id=job_id, started_at=_now(), kind=req.kind)


@app.get("/api/jobs/{job_id}/history", response_model=list[JobEvent])
def job_history(job_id: str) -> list[JobEvent]:
    bus = get_bus()
    history = bus.history(job_id)
    if not history:
        # Not necessarily an error: the job may have just been created.
        return []
    return history


# ─── WebSocket ────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = get_bus()
    _, iterator = await bus.subscribe()
    try:
        async for event in iterator:
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
    except Exception:
        # Close cleanly on backend error.
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Static frontend (production) ─────────────────────────────────────────


_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/")
    def spa_index() -> FileResponse:
        return FileResponse(_FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        # API routes are matched first by FastAPI; everything else falls
        # back to the SPA shell so the React router can take over.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")

else:

    @app.get("/")
    def no_frontend() -> JSONResponse:
        return JSONResponse(
            {
                "status": "backend_only",
                "message": (
                    "El bundle del frontend no existe. Ejecuta "
                    "`npm run build` dentro de desktop-app/frontend, o "
                    "arranca el modo desarrollo en localhost:5173."
                ),
            }
        )
