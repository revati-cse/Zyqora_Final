import argparse, json
from pathlib import Path
from .core.engine import get_engine
from .core.pipeline import classify_image
from .core.wastepulse import summarize

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    parser = argparse.ArgumentParser(description="ZYQORA AI single-image classification")
    parser.add_argument("--images", required=True)
    args = parser.parse_args()
    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit("No supported images found")
    engine, rows = get_engine(), []
    for path in paths:
        row = classify_image(str(path), engine).model_dump()
        rows.append(row)
        print(json.dumps(row, indent=2))
    print("\nWastePulse:")
    print(json.dumps(summarize(rows), indent=2))

if __name__ == "__main__":
    main()
