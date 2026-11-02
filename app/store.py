from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Match:
    profile_id: int
    name: str
    notes: dict[str, Any]
    similarity: float


class ProfileStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    notes_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        vector = np.asarray(embedding, dtype=np.float32)
        buffer = io.BytesIO()
        np.save(buffer, vector, allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def _deserialize_embedding(payload: bytes) -> np.ndarray:
        return np.load(io.BytesIO(payload), allow_pickle=False).astype(np.float32)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Embedding cannot be the zero vector")
        return vector / norm

    def add_profile(self, name: str, notes: dict[str, Any], embedding: np.ndarray) -> int:
        normalized = self._normalize(embedding)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO profiles (name, notes_json, embedding) VALUES (?, ?, ?)",
                (name.strip(), json.dumps(notes, sort_keys=True), self._serialize_embedding(normalized)),
            )
            return int(cursor.lastrowid)

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, notes_json, created_at FROM profiles ORDER BY id"
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "notes": json.loads(row["notes_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete_profile(self, profile_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            return cursor.rowcount > 0

    def match(self, embedding: np.ndarray, threshold: float) -> Match | None:
        query = self._normalize(embedding)
        best: Match | None = None

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, notes_json, embedding FROM profiles"
            ).fetchall()

        for row in rows:
            candidate = self._normalize(self._deserialize_embedding(row["embedding"]))
            similarity = float(np.dot(query, candidate))
            if best is None or similarity > best.similarity:
                best = Match(
                    profile_id=int(row["id"]),
                    name=row["name"],
                    notes=json.loads(row["notes_json"]),
                    similarity=similarity,
                )

        if best is None or best.similarity < threshold:
            return None
        return best
