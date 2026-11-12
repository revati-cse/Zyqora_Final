from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from .core.engine import get_engine
from .core.pipeline import classify_image
from .db import save_item
from .schemas import ModeEnum

app = FastAPI(title="ZYQORA AI API", version="0.1.0")

@app.get("/api/health")
def health():
    return {"status": "ok", "project": "ZYQORA AI"}

@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")
    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(content)
            temp_path = temp.name
        result = classify_image(temp_path, get_engine(), ModeEnum.LIVE)
        data = result.model_dump()
        data["filename"] = file.filename or "uploaded-image"
        save_item(data)
        return data
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
