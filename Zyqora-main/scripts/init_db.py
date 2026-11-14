#!/usr/bin/env python3
"""Create or migrate the dashboard database (the server also does this on start-up).

    python scripts/init_db.py                       # outputs/web/zyqora_web.sqlite3
    python scripts/init_db.py --db path/to/file.sqlite3
"""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import _bootstrap  # noqa: F401

from zyqora.api.detection_store import DetectionStore  # noqa: E402
from zyqora.api.store import HistoryStore  # noqa: E402
from zyqora.paths import OUTPUTS_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=OUTPUTS_DIR / "web" / "zyqora_web.sqlite3")
    args = ap.parse_args()
    HistoryStore(args.db)
    ver = DetectionStore(args.db, args.db.parent / "detections").migrate()
    with closing(sqlite3.connect(args.db)) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
    print(f"database: {args.db}\nschema version: {ver}")
    for t, n in counts.items():
        print(f"  {t:18s} {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
