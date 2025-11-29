"""Entrypoint script to launch the FastAPI server with Uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from service.config import ServiceSettings
from service.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nanoGPT inference API server.")
    parser.add_argument("--host", type=str, default=None, help="Host to bind (default from settings).")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default from settings).")
    parser.add_argument(
        "--reload", action="store_true", help="Enable autoreload (development only, requires watchfiles)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ServiceSettings()
    app = create_app(settings)

    host = args.host or settings.host
    port = args.port or settings.port

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
