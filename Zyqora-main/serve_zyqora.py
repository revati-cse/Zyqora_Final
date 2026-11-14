#!/usr/bin/env python3
"""Start the Zyqora AI API + dashboard (single process, model loaded once).

    python serve_zyqora.py                                  # http://127.0.0.1:8000 (this machine only)
    python serve_zyqora.py --host 0.0.0.0 --port 8000       # reachable from laptops on the local network
    python serve_zyqora.py --profile waste_mobilenetv2      # force the trained 2-class model
    python serve_zyqora.py --no-detector                    # single-item classification only (no YOLO)

The dashboard is served from frontend/dist (build it with `npm run build` in frontend/).
There is no authentication: only expose it on a trusted local network.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from zyqora.logs import setup_logging  # noqa: E402
from zyqora.paths import OUTPUTS_DIR  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 = all interfaces on the LAN)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--profile", default=None, help="model profile (default: config/models.yaml active)")
    ap.add_argument("--allow-fallback", action="store_true", help="use the fallback profile if this one fails (reported)")
    ap.add_argument("--threads", type=int, default=None, help="ExecuTorch thread count (default: config)")
    ap.add_argument("--camera", default=None, help="auto | picamera2 | rpicam | folder:<dir> (default: hardware.yaml)")
    ap.add_argument("--enable-actuator", action="store_true", help="allow the configured GPIO actuator to move")
    ap.add_argument("--cors-origin", action="append", default=None,
                    help="extra allowed browser origin, e.g. http://localhost:5173 (dev only; repeatable)")
    ap.add_argument("--static-dir", type=Path, default=ROOT / "frontend" / "dist")
    ap.add_argument("--models-config", type=Path, default=ROOT / "config" / "models.yaml",
                    help="model profiles file (advanced: e.g. a dev copy pointing at another .pte)")
    ap.add_argument("--no-thumbnails", action="store_true", help="do not store image thumbnails in history")
    ap.add_argument("--max-upload-mb", type=int, default=10)
    ap.add_argument("--no-detector", action="store_true", help="do not load the YOLO detector (mixed-waste mode off)")
    ap.add_argument("--detector-config", type=Path, default=ROOT / "config" / "detector.yaml")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    import uvicorn

    from zyqora.api.app import create_app
    from zyqora.api.service import ServiceSettings, ZyqoraService

    setup_logging(OUTPUTS_DIR / "web" / "server.log")
    settings = ServiceSettings(profile=args.profile, models_config=args.models_config, allow_fallback=args.allow_fallback,
                               num_threads=args.threads,
                               camera=args.camera, enable_actuator=args.enable_actuator,
                               store_thumbnails=not args.no_thumbnails, max_upload_bytes=args.max_upload_mb * 1048576,
                               enable_detector=not args.no_detector, detector_config=args.detector_config)
    service = ZyqoraService(settings)
    origins = args.cors_origin or [o.strip() for o in os.environ.get("ZYQORA_CORS_ORIGINS", "").split(",") if o.strip()]
    app = create_app(service, static_dir=args.static_dir, cors_origins=origins)
    if not (args.static_dir / "index.html").is_file():
        print(f"NOTE: no dashboard build at {args.static_dir}; API only. Build it with: cd frontend && npm run build")
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: listening on {args.host}: anyone on this network can use the dashboard (no authentication).")
    print(f"Zyqora AI: http://{'<this-device-ip>' if args.host == '0.0.0.0' else args.host}:{args.port}  (API docs at /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
