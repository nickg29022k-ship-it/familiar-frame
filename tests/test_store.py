from pathlib import Path

import numpy as np

from app.store import ProfileStore


def test_add_and_match_profile(tmp_path: Path) -> None:
    store = ProfileStore(str(tmp_path / "profiles.db"))
    alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    profile_id = store.add_profile("Alice", {"context": "Roboflow meetup"}, alice)

    match = store.match(np.array([0.99, 0.02, 0.0], dtype=np.float32), threshold=0.90)

    assert profile_id == 1
    assert match is not None
    assert match.name == "Alice"
    assert match.notes["context"] == "Roboflow meetup"
    assert match.similarity > 0.99


def test_unknown_person_rejected(tmp_path: Path) -> None:
    store = ProfileStore(str(tmp_path / "profiles.db"))
    store.add_profile("Alice", {}, np.array([1.0, 0.0], dtype=np.float32))

    match = store.match(np.array([0.0, 1.0], dtype=np.float32), threshold=0.72)

    assert match is None


def test_profile_listing_excludes_embeddings(tmp_path: Path) -> None:
    store = ProfileStore(str(tmp_path / "profiles.db"))
    store.add_profile("Bob", {"company": "Example"}, np.array([0.5, 0.5], dtype=np.float32))

    profiles = store.list_profiles()

    assert profiles[0]["name"] == "Bob"
    assert "embedding" not in profiles[0]
