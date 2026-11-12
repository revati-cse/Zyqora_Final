#!/usr/bin/env python3
"""Offline readiness check for the demo (run on the Pi with Wi-Fi/Ethernet disabled).

Verifies that every runtime asset is local and reports whether the network is
reachable. It does NOT disable networking itself; see README_PI.md section 14.

    python scripts/check_offline.py
"""

from __future__ import annotations

import argparse
import socket
import sys

import _bootstrap  # noqa: F401

from zyqora.config import load_app_config
from zyqora.imagenet_labels import load_labels
from zyqora.materialwise import load_mapping
from zyqora.paths import AUDIO_DIR
from zyqora.voicesort import LANGUAGES, REQUIRED_SYSTEM_KEYS, clip_path


def network_reachable(timeout: float = 2.0) -> bool:
    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ok = True
    app = load_app_config()
    labels = load_labels(app.labels)
    mapping = load_mapping(app.mapping, labels)
    checks = [
        ("model artifact", app.pte.is_file(), str(app.pte)),
        ("model contract", app.contract.is_file(), str(app.contract)),
        ("ImageNet labels (local)", len(labels) == 1000, "bundled"),
        ("mapping", True, f"{app.mapping} (status {mapping.status})"),
    ]
    for lang in LANGUAGES:
        keys = [*mapping.actions, *REQUIRED_SYSTEM_KEYS]
        missing = [k for k in keys if not clip_path(AUDIO_DIR, lang, k).is_file()]
        checks.append((f"audio clips [{lang}]", not missing, "all present" if not missing else f"missing {missing}"))
    try:
        from executorch.extension.pybindings import portable_lib

        backends = list(portable_lib._get_registered_backend_names())
        checks.append(("ExecuTorch runtime + XNNPACK", "XnnpackBackend" in backends, str(backends)))
    except ImportError as exc:
        checks.append(("ExecuTorch runtime + XNNPACK", False, str(exc)))
    for name, passed, detail in checks:
        ok &= passed
        print(f"{'OK  ' if passed else 'FAIL'} {name}: {detail}")
    online = network_reachable()
    print(f"{'WARN' if online else 'OK  '} network: {'REACHABLE - disable Wi-Fi/Ethernet for the offline demo' if online else 'not reachable (offline)'}")
    print("RESULT:", "READY" if ok else "NOT READY", "| OFFLINE" if not online else "| ONLINE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
