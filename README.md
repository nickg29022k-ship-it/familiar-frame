# Familiar Frame

A consent-based mobile face memory prototype for associating an enrolled person's face with user-provided notes.

## What it does

1. Captures a photo from a browser webcam or file upload.
2. Detects faces with **Roboflow Inference** (optional) or a local **MTCNN** fallback.
3. Generates a FaceNet embedding for each detected face.
4. Matches embeddings against an enrolled SQLite profile store using cosine similarity.
5. Returns the person's name and notes only when the score clears an explicit threshold; otherwise returns `unknown`.

This repository is a prototype, not a production biometric system. Use only with informed consent. Do not enroll or identify people without permission. Do not use it for surveillance, employment, housing, credit, healthcare, policing, or other high-impact decisions.

## Architecture

```text
Browser camera / upload
        |
        v
FastAPI service
        |
        +--> Roboflow face detector (optional) or local MTCNN --> bounding boxes
        |
        +--> FaceNet embedding model --> normalized vectors
        |
        +--> cosine similarity + unknown threshold
        |
        +--> SQLite profile metadata + embeddings
```

Identity matching is always local (FaceNet + SQLite). Roboflow is used only for face detection when credentials are configured.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Optional: set ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID for cloud detection.
# Leave them empty to use local MTCNN.
export TORCH_HOME="$PWD/.torch_cache"   # optional; caches FaceNet weights
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` in a desktop browser (webcam) or on a phone on the same network.

Never commit `.env` — it is gitignored.

## Environment variables

- `ROBOFLOW_API_KEY`: Roboflow API key (optional; empty → local MTCNN).
- `ROBOFLOW_MODEL_ID`: face detector model/version, such as `face-detection-bj0ij/1`.
- `MATCH_THRESHOLD`: cosine similarity cutoff; default `0.72`.
- `DATABASE_PATH`: SQLite path; default `familiar-frame.db`.

## Endpoints

- `POST /api/enroll`: multipart fields `name`, `notes`, and `image`.
- `POST /api/identify`: multipart field `image`.
- `GET /api/profiles`: list enrolled profile metadata without embeddings.
- `DELETE /api/profiles/{profile_id}`: delete a profile.

## Reproducibility and evaluation plan

Before describing this as a completed research project, collect a consented evaluation set and report:

- Identification accuracy at multiple thresholds.
- False-accept rate and false-reject rate.
- Unknown-person rejection accuracy.
- Latency on phone-to-service requests.
- Robustness across lighting, pose, distance, and occlusion.
- Ablation: Roboflow crop vs. uncropped image embedding.

## Tests

```bash
pytest -q
```

The included tests validate profile enrollment, cosine matching, and unknown-person rejection without requiring cloud credentials.
