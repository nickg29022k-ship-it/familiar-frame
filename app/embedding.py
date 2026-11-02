from __future__ import annotations

import numpy as np
from PIL import Image


class FaceEmbedder:
    def __init__(self) -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1
        from torchvision import transforms

        self.torch = torch
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

    def embed(self, face: Image.Image) -> np.ndarray:
        tensor = self.transform(face.convert("RGB")).unsqueeze(0)
        with self.torch.no_grad():
            embedding = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Model produced an invalid zero embedding")
        return embedding / norm
