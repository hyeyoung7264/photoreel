import io
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from photoreel.store import Project


def test_exif_rotation_is_applied_and_time_is_read(jpeg):
    # 640x480으로 저장됐지만 '시계 방향 90도 회전' 표시가 있는 사진 (세로로 찍은 폰 사진)
    img = Image.open(io.BytesIO(jpeg(3, (640, 480))))
    exif = Image.Exif()
    exif[274] = 6
    exif.get_ifd(0x8769)[36867] = "2026:04:05 06:07:08"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    p = Project.create()
    a = p.add_photo("rotated.jpg", buf.getvalue())
    assert (a["width"], a["height"]) == (480, 640)
    assert a["taken_at"] == datetime(2026, 4, 5, 6, 7, 8).isoformat()
    assert Image.open(p.proxy_path(a["id"])).size == (480, 640)
    assert abs(sum(map(sum, a["saliency"])) - 1) < 1e-3 and len(a["saliency"]) == 24


def test_missing_exif_time_is_none_not_file_time(jpeg):
    a = Project.create().add_photo("plain.jpg", jpeg(4))
    assert a["taken_at"] is None


def test_heic_and_png_are_accepted(jpeg):
    img = Image.open(io.BytesIO(jpeg(5, (400, 300))))
    p = Project.create()
    png = io.BytesIO()
    img.convert("RGBA").save(png, "PNG")
    assert p.add_photo("shot.png", png.getvalue())["width"] == 400
    heic = io.BytesIO()
    try:
        img.save(heic, "HEIF")
    except Exception:  # noqa: BLE001
        pytest.skip("이 환경의 pillow-heif에 인코더가 없음")
    assert p.add_photo("phone.HEIC", heic.getvalue())["height"] == 300
    assert len(p.photos()) == 2


def test_fallback_embedding_without_clip_is_normalized(jpeg):
    p = Project.create()
    a = p.add_photo("a.jpg", jpeg(6))
    emb = p.embeddings()[a["id"]]
    assert a["methods"]["semantic"] == "color-hist" and abs(np.linalg.norm(emb) - 1) < 1e-4
