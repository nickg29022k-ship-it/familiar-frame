from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import get_settings
from app.embedding import FaceEmbedder
from app.roboflow_detector import build_face_detector
from app.store import ProfileStore

app = FastAPI(title="Familiar Frame", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@lru_cache
def get_store() -> ProfileStore:
    return ProfileStore(get_settings().database_path)


@lru_cache
def get_detector():
    settings = get_settings()
    return build_face_detector(settings.roboflow_api_key, settings.roboflow_model_id)


@lru_cache
def get_embedder() -> FaceEmbedder:
    return FaceEmbedder()


def read_image(payload: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # Pillow raises several possible parsing errors.
        raise HTTPException(status_code=400, detail="Invalid image") from exc


def crop_with_margin(image: Image.Image, box, margin: float = 0.12) -> Image.Image:
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    mx = round(width * margin)
    my = round(height * margin)
    x1 = max(0, box.x1 - mx)
    y1 = max(0, box.y1 - my)
    x2 = min(image.width, box.x2 + mx)
    y2 = min(image.height, box.y2 + my)
    return image.crop((x1, y1, x2, y2))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path("static/index.html"))


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    detector_name = (
        "roboflow"
        if settings.roboflow_api_key and settings.roboflow_model_id
        else "mtcnn_local"
    )
    return {"status": "ok", "detector": detector_name}


@app.get("/api/profiles")
def list_profiles() -> list[dict]:
    return get_store().list_profiles()


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: int) -> dict[str, bool]:
    return {"deleted": get_store().delete_profile(profile_id)}


@app.post("/api/enroll")
async def enroll(
    name: str = Form(...),
    notes: str = Form("{}"),
    image: UploadFile = File(...),
) -> dict:
    try:
        note_data = json.loads(notes)
        if not isinstance(note_data, dict):
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="notes must be a JSON object") from exc

    picture = read_image(await image.read())
    boxes = get_detector().detect(picture)
    if not boxes:
        raise HTTPException(status_code=422, detail="No face detected")

    face = crop_with_margin(picture, boxes[0])
    embedding = get_embedder().embed(face)
    profile_id = get_store().add_profile(name=name, notes=note_data, embedding=embedding)
    return {"profile_id": profile_id, "name": name, "detector_confidence": boxes[0].confidence}


@app.post("/api/identify")
async def identify(image: UploadFile = File(...)) -> dict:
    picture = read_image(await image.read())
    boxes = get_detector().detect(picture)
    settings = get_settings()
    results = []

    for box in boxes:
        face = crop_with_margin(picture, box)
        embedding = get_embedder().embed(face)
        match = get_store().match(embedding, threshold=settings.match_threshold)
        results.append(
            {
                "box": box.__dict__,
                "identity": None
                if match is None
                else {
                    "profile_id": match.profile_id,
                    "name": match.name,
                    "notes": match.notes,
                    "similarity": round(match.similarity, 4),
                },
            }
        )

    return {"faces": results, "match_threshold": settings.match_threshold}
