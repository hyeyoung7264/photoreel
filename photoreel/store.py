"""프로젝트 파일 저장소. 한 사람이 자기 컴퓨터에서 쓰는 것을 전제로 한 단순한 폴더 구조다.

data/projects/<id>/
  orig/    올린 원본            proxy/  방향을 바로잡은 렌더링용 사본
  thumb/   화면용 작은 그림     emb/    사진 임베딩
  photos.json  분석 결과 (올린 순서)   plan.json  현재 구성안
  renders/<n>/ video.mp4, storyboard.json, stats.json
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import analyze
from .config import DATA_DIR
from .models import Storyboard

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
_ID = re.compile(r"^[0-9a-f]{6,32}$")


class Project:
    _locks: dict[str, threading.Lock] = {}

    def __init__(self, pid: str) -> None:
        if not _ID.match(pid):
            raise ValueError("잘못된 프로젝트 id")
        self.id = pid
        self.dir = DATA_DIR / "projects" / pid
        self.lock = Project._locks.setdefault(pid, threading.Lock())

    @classmethod
    def create(cls) -> "Project":
        p = cls(uuid.uuid4().hex[:12])
        for sub in ("orig", "proxy", "thumb", "emb", "renders"):
            (p.dir / sub).mkdir(parents=True, exist_ok=True)
        p._write("photos.json", [])
        return p

    def exists(self) -> bool:
        return (self.dir / "photos.json").exists()

    def _read(self, name: str, default=None):
        path = self.dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def _write(self, name: str, data) -> None:
        path = self.dir / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    # ----- 사진 -----
    def photos(self) -> list[dict]:
        return self._read("photos.json", [])

    def add_photo(self, filename: str, data: bytes) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in IMAGE_EXT:
            raise ValueError(f"지원하지 않는 파일 형식: {filename}")
        photo_id = uuid.uuid4().hex[:10]
        original = self.dir / "orig" / f"{photo_id}{ext}"
        original.write_bytes(data)
        proxy = self.proxy_path(photo_id)
        try:
            result, emb = analyze.analyze(photo_id, original, proxy, Path(filename).name)
            analyze.make_thumb(proxy, self.thumb_path(photo_id))
        except Exception as e:
            original.unlink(missing_ok=True)
            proxy.unlink(missing_ok=True)
            raise ValueError(f"사진을 읽지 못함: {filename} ({e})") from e
        np.save(self.dir / "emb" / f"{photo_id}.npy", emb)
        with self.lock:
            photos = self.photos()
            photos.append(result)
            self._write("photos.json", photos)
        return result

    def remove_photo(self, photo_id: str) -> None:
        with self.lock:
            photos = [p for p in self.photos() if p["id"] != photo_id]
            self._write("photos.json", photos)
        for path in self.dir.glob(f"*/{photo_id}.*"):
            path.unlink(missing_ok=True)

    def proxy_path(self, photo_id: str) -> Path:
        return self.dir / "proxy" / f"{photo_id}.jpg"

    def thumb_path(self, photo_id: str) -> Path:
        return self.dir / "thumb" / f"{photo_id}.jpg"

    def embeddings(self) -> dict[str, np.ndarray]:
        return {p["id"]: np.load(self.dir / "emb" / f"{p['id']}.npy") for p in self.photos()}

    # ----- 구성안 -----
    def plan(self) -> Storyboard | None:
        data = self._read("plan.json")
        return Storyboard.model_validate(data) if data else None

    def save_plan(self, board: Storyboard) -> None:
        self._write("plan.json", board.model_dump())

    # ----- 렌더 결과 -----
    def new_render(self) -> tuple[int, Path]:
        with self.lock:
            existing = [int(p.name) for p in (self.dir / "renders").iterdir() if p.name.isdigit()]
            n = max(existing, default=0) + 1
            path = self.dir / "renders" / str(n)
            path.mkdir(parents=True)
        return n, path

    def render_dir(self, n: int) -> Path:
        return self.dir / "renders" / str(int(n))

    def renders(self) -> list[dict]:
        out = []
        root = self.dir / "renders"
        for d in sorted((p for p in root.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
            stats = d / "stats.json"
            if stats.exists():
                out.append(json.loads(stats.read_text(encoding="utf-8")))
        return out

    def save_render(self, n: int, board: Storyboard, stats: dict) -> dict:
        d = self.render_dir(n)
        (d / "storyboard.json").write_text(json.dumps(board.model_dump(), ensure_ascii=False, indent=1), encoding="utf-8")
        stats = {"render": n, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **stats}
        (d / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
        return stats
