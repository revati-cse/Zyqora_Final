#!/usr/bin/env python3
"""Zyqora station: capture -> classify -> explain -> speak -> sort (simulated by default) -> GreenScore.

One capture per key press (no continuous video inference). Fully offline.

    python zyqora_station.py --camera folder:data/demo --voice ta --user DEMO01        # simulation, no camera
    python zyqora_station.py --camera auto --voice en --user DEMO01                     # Pi camera
    python zyqora_station.py --camera folder:data/demo --auto 20 --voice off            # unattended latency run

Keys: Enter = capture, q = quit.
The model is loaded once. If the active profile cannot be loaded, the configured fallback profile
is used and announced (disable with --no-fallback). REVIEW items are never sorted automatically.
The GPIO actuator only runs with `actuator.backend: gpio` in config/hardware.yaml AND --enable-actuator.

Writes outputs/station/<run>/events.jsonl, station.log, summary.json (and captures with --save-captures).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from zyqora.benchmark import platform_facts, read_proc_status  # noqa: E402
from zyqora.camera import CameraError, open_camera  # noqa: E402
from zyqora.config import ConfigError, load_app_config, load_yaml  # noqa: E402
from zyqora.evaluator import latency_stats  # noqa: E402
from zyqora.explain import explanation_card  # noqa: E402
from zyqora.logs import setup_logging  # noqa: E402
from zyqora.model_runtime import ModelLoadError, RuntimeUnavailableError  # noqa: E402
from zyqora.outputs import write_json  # noqa: E402
from zyqora.pipeline import ZyqoraPipeline  # noqa: E402
from zyqora.profiles import load_models_config, load_with_fallback  # noqa: E402
from zyqora.sorter import ActuatorError, decide_sort, make_sorter  # noqa: E402

MAX_CONSECUTIVE_CAMERA_FAILURES = 3


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None, help="model profile (default: config/models.yaml active)")
    ap.add_argument("--no-fallback", action="store_true", help="exit instead of using the fallback model")
    ap.add_argument("--camera", default=None, help="auto | picamera2 | rpicam | folder:<dir> (default: hardware.yaml)")
    ap.add_argument("--voice", choices=["off", "en", "ta"], default="off")
    ap.add_argument("--user", default=None, help="GreenScore user id: records a PENDING event per sorted item")
    ap.add_argument("--verify-now", action="store_true", help="ask for supervisor verification after each item")
    ap.add_argument("--enable-actuator", action="store_true", help="allow the configured GPIO actuator to move")
    ap.add_argument("--auto", type=int, default=None, metavar="N", help="capture N items without key presses")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--save-captures", action="store_true", help="keep captured frames (JPEG) in the run folder")
    ap.add_argument("--hardware", type=Path, default=ROOT / "config" / "hardware.yaml")
    ap.add_argument("--models-config", type=Path, default=ROOT / "config" / "models.yaml")
    ap.add_argument("--greenscore-db", type=Path, default=None, help="override GreenScore database path")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)
    run_dir = args.out or ROOT / "outputs" / "station" / started.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "station.log")
    import logging

    log = logging.getLogger("zyqora.station")

    try:
        app = load_app_config()
        models = load_models_config(app, args.models_config)
        hw = load_yaml(args.hardware) if args.hardware.is_file() else {}
        loaded = load_with_fallback(models, args.profile, allow_fallback=not args.no_fallback,
                                    num_threads=args.threads or app.num_threads, require_backend=app.require_backend)
    except (ModelLoadError, RuntimeUnavailableError) as exc:
        print(f"MODEL ERROR: {exc}", file=sys.stderr)
        log.error("model load failed: %s", exc)
        return 3
    except (ConfigError, ValueError, OSError) as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    pipe = ZyqoraPipeline(loaded.classifier, loaded.decoder, loaded.guard, loaded.spec)
    import numpy as np

    loaded.classifier.infer(np.zeros(loaded.spec.input_shape, dtype=np.float32))  # warm-up

    cam_cfg = hw.get("camera", {}) or {}
    cam_spec = args.camera or cam_cfg.get("backend", "auto")
    try:
        camera = open_camera(cam_spec, int(cam_cfg.get("width", 640)), int(cam_cfg.get("height", 480)))
    except CameraError as exc:
        print(f"CAMERA ERROR: {exc}\nTip: run without a camera using --camera folder:<image folder>.", file=sys.stderr)
        log.error("camera open failed: %s", exc)
        return 5
    try:
        sorter = make_sorter(hw.get("actuator", {}), args.enable_actuator)
    except ActuatorError as exc:
        print(f"ACTUATOR ERROR: {exc}", file=sys.stderr)
        camera.close()
        return 6

    voice = None
    if args.voice != "off":
        from zyqora.voicesort import VoiceSort

        voice = VoiceSort(args.voice)
    gs = None
    if args.user:
        from zyqora.greenscore import GreenScore

        db = args.greenscore_db or app.greenscore_db
        db.parent.mkdir(parents=True, exist_ok=True)
        gs = GreenScore(db, app.points_correct, app.points_incorrect, loaded.category_set.actions)
        gs.ensure_user(args.user)

    p = loaded.profile
    print(f"Zyqora station | model {p.name} ({p.pte.name}){'  ** FALLBACK **' if loaded.fallback_used else ''} | "
          f"camera {camera.name}{' (SIMULATION)' if camera.simulated else ''} | sorter {sorter.name}"
          f"{'' if sorter.hardware else ' (SIMULATION)'} | voice {args.voice} | EcoGuard {loaded.guard.threshold}")
    if loaded.fallback_used:
        print(f"WARNING: {loaded.failures[0]}")
        if voice:
            voice.announce("MANUAL_REVIEW")
    print("Enter = capture, q = quit" if args.auto is None else f"Auto mode: {args.auto} captures")

    events, cam_fail, n = [], 0, 0
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as ev_file:
        while True:
            if args.auto is not None:
                if n >= args.auto:
                    break
            else:
                try:
                    if input("\n> ").strip().lower() == "q":
                        break
                except EOFError:
                    break
            n += 1
            t0 = time.perf_counter_ns()
            try:
                frame = camera.capture()
                cam_fail = 0
            except CameraError as exc:
                cam_fail += 1
                print(f"  CAMERA ERROR: {exc} (nothing classified)")
                log.error("capture failed (%d in a row): %s", cam_fail, exc)
                if cam_fail >= MAX_CONSECUTIVE_CAMERA_FAILURES:
                    print("  Too many camera failures; stopping.")
                    break
                continue
            result = pipe.process_image(frame.image, frame.name)
            camera_to_result_ms = (time.perf_counter_ns() - t0) / 1e6
            print(f"[{n}] {frame.name}  (capture {frame.capture_ms:.0f} ms, camera-to-result {camera_to_result_ms:.0f} ms)")
            print(explanation_card(result, loaded.guard.threshold, p.name, p.decoder))
            decision = decide_sort(result.ecoguard_status, result.disposal_action)
            try:
                sort_msg = sorter.execute(decision)
            except ActuatorError as exc:
                sort_msg = f"ACTUATOR ERROR: {exc} (item held for supervisor)"
                log.error("actuator failed: %s", exc)
            print(f"  Sorter:   {sort_msg}")
            if voice:
                voice.announce(result.disposal_action)
            event_id = None
            if gs and not result.error:
                event_id = gs.record_event(args.user, frame.name, result.predicted_category, result.disposal_action,
                                           result.ecoguard_status, run_id)
                print(f"  GreenScore: PENDING event {event_id} (points only after supervisor verification)")
                if args.verify_now and args.auto is None:
                    from zyqora.greenscore import _interactive_verify

                    observed, correct = _interactive_verify(gs, event_id)
                    res = gs.verify(event_id, observed, correct)
                    print(f"  GreenScore: {res.status} +{res.awarded_points} (total {res.total_points})")
                    if voice:
                        voice.announce(res.status)
            if args.save_captures:
                frame.image.save(run_dir / f"{Path(frame.name).stem}_{n:04d}.jpg", quality=90)
            rec = {"n": n, "frame": frame.name, "camera": frame.source, "simulated_camera": frame.simulated,
                   "capture_ms": frame.capture_ms, "preprocess_ms": result.preprocess_time_ms,
                   "inference_ms": result.inference_time_ms, "camera_to_result_ms": camera_to_result_ms,
                   "total_ms_incl_sort_voice": (time.perf_counter_ns() - t0) / 1e6,
                   "predicted_category": result.predicted_category, "score": result.predicted_category_score,
                   "ecoguard": result.ecoguard_status, "reason": result.review_reason, "action": result.disposal_action,
                   "sorter": sort_msg, "greenscore_event": event_id, "error": result.error}
            ev_file.write(json.dumps(rec) + "\n")
            ev_file.flush()
            events.append(rec)
            log.info("item %d %s -> %s %s %s", n, frame.name, result.predicted_category, result.ecoguard_status,
                     result.disposal_action)
    camera.close()
    sorter.close()

    ok = [e for e in events if not e["error"]]
    summary = {
        "items": len(events), "classified": len(ok), "camera": camera.name, "camera_simulated": camera.simulated,
        "sorter": sorter.name, "sorter_hardware": sorter.hardware, "model_profile": p.name,
        "fallback_used": loaded.fallback_used, "accepted": sum(e["ecoguard"] == "ACCEPT" for e in ok),
        "review": sum(e["ecoguard"] != "ACCEPT" for e in events),
        "camera_to_result_ms": latency_stats([e["camera_to_result_ms"] for e in ok]),
        "capture_ms": latency_stats([e["capture_ms"] for e in ok]),
        "inference_ms": latency_stats([e["inference_ms"] for e in ok]),
        "memory": read_proc_status(), "platform": platform_facts(), "started_utc": started.isoformat(timespec="seconds"),
    }
    write_json(summary, run_dir / "summary.json")
    c2r = summary["camera_to_result_ms"]
    print(f"\n{len(events)} item(s): ACCEPT {summary['accepted']}, REVIEW {summary['review']}"
          + (f" | camera-to-result median {c2r['median_ms']:.0f} ms, p95 {c2r['p95_ms']:.0f} ms" if ok else ""))
    print(f"Log and events: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
