#!/usr/bin/env python3
"""Single-command launcher for `Querétaro Urban Lab`.

What it does, in order:

1. Locates the desktop-app directory next to this script.
2. Ensures the frontend bundle is present, building it with `npm` if not.
3. Starts `uvicorn backend.main:app` bound to localhost:8765.
4. Opens the default browser at the app URL.
5. Forwards SIGINT cleanly to shut everything down.

The script is intentionally dependency-free (only standard library).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
PROJECT_ROOT = APP_DIR.parent
FRONTEND_DIR = APP_DIR / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"


def _info(msg: str) -> None:
    print(f"\033[2m[launch]\033[0m {msg}", flush=True)


def _err(msg: str) -> None:
    print(f"\033[31m[launch] {msg}\033[0m", flush=True, file=sys.stderr)


def ensure_frontend_built(skip: bool = False) -> bool:
    """Build the frontend if `dist/index.html` is missing.

    Returns whether the static bundle is available.
    """
    if skip:
        return DIST_DIR.is_dir() and (DIST_DIR / "index.html").is_file()

    if (DIST_DIR / "index.html").is_file():
        return True

    npm = shutil.which("npm")
    if npm is None:
        _err(
            "npm no está disponible en el PATH. "
            "Instala Node 18+ o corre el frontend en modo desarrollo."
        )
        return False

    if not (FRONTEND_DIR / "node_modules").is_dir():
        _info("Instalando dependencias del frontend (npm install)…")
        rc = subprocess.run([npm, "install"], cwd=FRONTEND_DIR).returncode
        if rc != 0:
            _err("npm install falló.")
            return False

    _info("Construyendo bundle del frontend (npm run build)…")
    rc = subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR).returncode
    if rc != 0:
        _err("npm run build falló.")
        return False

    return (DIST_DIR / "index.html").is_file()


def wait_until_up(url: str, timeout: float = 20.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with urlopen(url, timeout=0.5) as r:
                if r.status == 200:
                    return True
        except (URLError, OSError):
            time.sleep(0.2)
    return False


def run(port: int, no_browser: bool, no_build: bool) -> int:
    has_bundle = ensure_frontend_built(skip=no_build)
    if not has_bundle:
        _err(
            "El bundle del frontend no está disponible. La app se servirá "
            "como API; abre Vite en modo dev para usar la UI."
        )

    env = os.environ.copy()
    env["TESIS_PROJECT_ROOT"] = str(PROJECT_ROOT)

    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--app-dir",
        str(APP_DIR),
        "--log-level",
        "info",
    ]

    _info(f"Iniciando backend → http://localhost:{port}")
    proc = subprocess.Popen(uvicorn_cmd, env=env)

    def _shutdown(*_: object) -> None:
        _info("Cerrando backend…")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def _open_browser() -> None:
        url = f"http://localhost:{port}"
        if wait_until_up(url + "/api/health"):
            if not no_browser:
                _info(f"Abriendo navegador en {url}")
                webbrowser.open(url, new=2)
        else:
            _err("El backend no respondió a /api/health en 20 s.")

    threading.Thread(target=_open_browser, daemon=True).start()

    return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lanzador de la aplicación de escritorio."
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="No intentar construir el frontend; usar el bundle existente.",
    )
    args = parser.parse_args()
    return run(port=args.port, no_browser=args.no_browser, no_build=args.no_build)


if __name__ == "__main__":
    raise SystemExit(main())
