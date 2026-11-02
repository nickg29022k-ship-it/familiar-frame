from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class RoboflowFaceDetector:
    """Thin adapter around Roboflow Inference.

    The model ID is intentionally configurable so the prototype can use a private,
    consented face-detection dataset or an appropriate public face detector.
    """

    def __init__(self, api_key: str, model_id: str) -> None:
        if not api_key or not model_id:
            raise RuntimeError(
                "ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID must be configured"
            )
        from inference_sdk import InferenceHTTPClient

        self.model_id = model_id
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )

    def detect(self, image: Image.Image) -> list[BoundingBox]:
        result: dict[str, Any] = self.client.infer(image, model_id=self.model_id)
        width, height = image.size
        boxes: list[BoundingBox] = []

        for prediction in result.get("predictions", []):
            center_x = float(prediction["x"])
            center_y = float(prediction["y"])
            box_width = float(prediction["width"])
            box_height = float(prediction["height"])
            x1 = max(0, round(center_x - box_width / 2))
            y1 = max(0, round(center_y - box_height / 2))
            x2 = min(width, round(center_x + box_width / 2))
            y2 = min(height, round(center_y + box_height / 2))
            boxes.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(prediction.get("confidence", 0.0)),
                )
            )

        return sorted(boxes, key=lambda box: box.confidence, reverse=True)


class MTCNNFaceDetector:
    """Local face detector for webcam demos when Roboflow is not configured."""

    def __init__(self) -> None:
        from facenet_pytorch import MTCNN

        self.mtcnn = MTCNN(keep_all=True, post_process=False)

    def detect(self, image: Image.Image) -> list[BoundingBox]:
        boxes_arr, probs = self.mtcnn.detect(image.convert("RGB"))
        if boxes_arr is None:
            return []

        boxes: list[BoundingBox] = []
        for box, prob in zip(boxes_arr, probs):
            if prob is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            boxes.append(
                BoundingBox(
                    x1=max(0, x1),
                    y1=max(0, y1),
                    x2=x2,
                    y2=y2,
                    confidence=float(prob),
                )
            )
        return sorted(boxes, key=lambda box: box.confidence, reverse=True)


def build_face_detector(api_key: str, model_id: str):
    if api_key and model_id:
        return RoboflowFaceDetector(api_key, model_id)
    return MTCNNFaceDetector()
