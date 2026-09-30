"""테스트는 합성 이미지로 돈다. 네트워크와 큰 모델 파일 없이 실행되도록 CLIP은 기본적으로 끈다."""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from photoreel import semantic, store

    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(semantic, "get_clip", lambda: None)
    store.Project._locks.clear()
    yield


def make_jpeg(seed: int, size=(640, 480), taken_at: datetime | None = None) -> bytes:
    """사진마다 색과 무늬가 다른 합성 이미지."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = rng.integers(40, 215, 3)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = base[c] + 40 * np.sin(xx / (20 + 9 * seed % 50) + c) + 30 * np.cos(yy / (15 + 7 * seed % 40))
    cx, cy, r = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4), min(w, h) // 6
    img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.integers(0, 255, 3)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    if taken_at is not None:
        exif = Image.Exif()
        exif[306] = taken_at.strftime("%Y:%m:%d %H:%M:%S")
        ifd = exif.get_ifd(0x8769)
        ifd[36867] = taken_at.strftime("%Y:%m:%d %H:%M:%S")
        pil.save(buf, "JPEG", quality=90, exif=exif)
    else:
        pil.save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def jpeg():
    return make_jpeg


@pytest.fixture
def project_with_photos():
    """사진 5장(가로·세로 섞임, 촬영 시각 있음)이 들어 있는 프로젝트. 올린 순서는 시간 역순."""
    from photoreel.store import Project

    p = Project.create()
    t0 = datetime(2026, 5, 1, 9, 0, 0)
    for i in range(5):
        size = (640, 480) if i % 2 == 0 else (480, 640)
        p.add_photo(f"p{i}.jpg", make_jpeg(i + 1, size, t0 + timedelta(minutes=30 * (4 - i))))
    return p
