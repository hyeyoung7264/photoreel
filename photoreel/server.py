"""로컬 웹 서버. 한 사람이 자기 컴퓨터에서 쓰는 검증용이다 (로그인·다중 사용자 없음)."""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import assets, planner, render, semantic
from .config import WEB_DIR
from .engines import ENGINES
from .models import ASPECTS, AUTO_DURATION, DEFAULT_ASPECT, Storyboard
from .store import Project
from .styles import FRAMINGS, GRADES, MOTIONS, MUSIC_MOODS, STYLES, TRANSITIONS

log = logging.getLogger("photoreel")
app = FastAPI(title="photoreel", docs_url=None, redoc_url=None)

MAX_PHOTOS = 80
MAX_BYTES = 60 * 1024 * 1024
JOBS: dict[str, dict] = {}
RENDER_LOCK = threading.Lock()  # CPU를 다 쓰는 작업이라 한 번에 하나만

# 입력이 막막한 사람을 위한 예시. 용도를 제한하지 않는다 (자유 문장이 기본).
CONCEPT_EXAMPLES = [
    {"label": "여행", "text": "잔잔하고 따뜻한 분위기의 여행 기록"},
    {"label": "여행 하이라이트", "text": "신나고 경쾌한 여행 하이라이트, 빠르게"},
    {"label": "일상", "text": "담백하게 정리한 이번 주 일상"},
    {"label": "가족", "text": "필름 사진 같은 가족의 추억, 인물 위주로"},
    {"label": "반려동물", "text": "우리 강아지의 발랄한 하루"},
    {"label": "행사", "text": "영화처럼 묵직한 분위기의 행사 스케치, 시간순으로"},
]


def _project(pid: str) -> Project:
    try:
        p = Project(pid)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    if not p.exists():
        raise HTTPException(404, "프로젝트를 찾을 수 없습니다.")
    return p


def _public_photo(a: dict) -> dict:
    tags = a.get("tags") or {}
    return {
        "id": a["id"],
        "filename": a["filename"],
        "width": a["width"],
        "height": a["height"],
        "taken_at": a.get("taken_at"),
        "faces": len(a.get("faces", [])),
        "scene": semantic.SCENE_NAMES.get(tags.get("scene")),
        "scene_conf": tags.get("scene_conf"),
        "light": semantic.LIGHT_NAMES.get(tags.get("light")),
        "methods": a.get("methods"),
    }


def _job(pid: str) -> dict:
    return JOBS.get(pid, {"state": "idle"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/meta")
def meta() -> dict:
    return {
        "styles": [s.public() for s in STYLES.values()],
        "motions": MOTIONS,
        "transitions": TRANSITIONS,
        "framings": FRAMINGS,
        "grades": GRADES,
        "music_moods": MUSIC_MOODS,
        "aspects": {k: list(v) for k, v in ASPECTS.items()},
        "default_aspect": DEFAULT_ASPECT,
        "auto_duration": list(AUTO_DURATION),
        "engines": [e.public() for e in ENGINES.values()],
        "concept_examples": CONCEPT_EXAMPLES,
        "understanding": {
            "clip": semantic.get_clip() is not None,
            "note": "사진 분석은 전부 이 컴퓨터에서 실행되며 사진을 외부로 보내지 않습니다.",
        },
    }


@app.post("/api/projects")
def create_project() -> dict:
    return {"id": Project.create().id}


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict:
    p = _project(pid)
    plan = p.plan()
    return {
        "id": p.id,
        "photos": [_public_photo(a) for a in p.photos()],
        "plan": plan.model_dump() if plan else None,
        "renders": p.renders(),
        "job": _job(pid),
    }


@app.post("/api/projects/{pid}/photos")
def upload_photos(pid: str, files: list[UploadFile] = File(...)) -> dict:
    p = _project(pid)
    added, errors = [], []
    for f in files:
        name = f.filename or "photo"
        if len(p.photos()) >= MAX_PHOTOS:
            errors.append({"filename": name, "error": f"사진은 최대 {MAX_PHOTOS}장까지입니다."})
            continue
        data = f.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            errors.append({"filename": name, "error": "파일이 너무 큽니다 (60MB 초과)."})
            continue
        try:
            added.append(_public_photo(p.add_photo(name, data)))
        except ValueError as e:
            errors.append({"filename": name, "error": str(e)})
    return {"added": added, "errors": errors}


@app.delete("/api/projects/{pid}/photos/{photo_id}")
def delete_photo(pid: str, photo_id: str) -> dict:
    p = _project(pid)
    if not any(a["id"] == photo_id for a in p.photos()):
        raise HTTPException(404, "사진을 찾을 수 없습니다.")
    p.remove_photo(photo_id)
    plan = p.plan()
    if plan:
        plan.scenes = [s for s in plan.scenes if s.photo_id != photo_id]
        p.save_plan(plan.finalize())
    return {"ok": True}


@app.get("/api/projects/{pid}/photos/{photo_id}/thumb")
def thumb(pid: str, photo_id: str) -> FileResponse:
    p = _project(pid)
    if not any(a["id"] == photo_id for a in p.photos()):
        raise HTTPException(404, "사진을 찾을 수 없습니다.")
    return FileResponse(p.thumb_path(photo_id), media_type="image/jpeg")


class PlanRequest(BaseModel):
    concept: str = ""
    title: str = ""
    aspect: str = DEFAULT_ASPECT
    target_duration: float | None = None
    music: bool = True
    style: str | None = None
    keep_current_order: bool = False


@app.post("/api/projects/{pid}/plan")
def make_plan(pid: str, req: PlanRequest) -> dict:
    p = _project(pid)
    photos = p.photos()
    if not photos:
        raise HTTPException(400, "사진을 먼저 올려 주세요.")
    if req.aspect not in ASPECTS:
        raise HTTPException(400, "지원하지 않는 화면 비율입니다.")
    if req.style and req.style not in STYLES:
        raise HTTPException(400, "알 수 없는 스타일입니다.")
    if req.target_duration is not None and not (5 <= req.target_duration <= 180):
        raise HTTPException(400, "길이는 5~180초 사이로 정해 주세요.")
    keep = None
    current = p.plan()
    if req.keep_current_order and current:
        keep = [(s.photo_id, s.included, s.exclude_reason) for s in current.scenes]
    board = planner.build(
        photos, p.embeddings(), req.concept[:500], title=req.title[:60], aspect=req.aspect,
        target_duration=req.target_duration, music=req.music, style_id=req.style, keep=keep,
    )
    if keep is not None and ENGINES.get(current.engine) and ENGINES[current.engine].available:
        board.engine = current.engine  # 스타일만 바꿀 때는 고른 생성 방식을 유지
    p.save_plan(board)
    return board.model_dump()


@app.put("/api/projects/{pid}/plan")
def update_plan(pid: str, board: Storyboard) -> dict:
    p = _project(pid)
    ids = {a["id"] for a in p.photos()}
    seen: set[str] = set()
    for s in board.scenes:
        if s.photo_id not in ids or s.photo_id in seen:
            raise HTTPException(400, "구성안에 없는 사진이거나 같은 사진이 두 번 들어 있습니다.")
        seen.add(s.photo_id)
    if board.engine not in ENGINES:
        raise HTTPException(400, "알 수 없는 생성 방식입니다.")
    try:
        board.finalize()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    p.save_plan(board)
    return board.model_dump()


def _run_render(p: Project, board: Storyboard, n: int, job: dict) -> None:
    rdir = p.render_dir(n)
    stage = {"name": "대기"}

    def progress(frac: float, name: str) -> None:
        stage["name"] = name
        job.update(progress=round(frac, 3), stage=name)

    try:
        job.update(stage="다른 작업이 끝나길 기다리는 중")
        with RENDER_LOCK:
            photos = {a["id"]: a for a in p.photos()}
            proxies = {pid: p.proxy_path(pid) for pid in photos}
            stats = render.render(board, photos, proxies, rdir / "video.mp4", rdir, progress)
        stats["status"] = "ok"
        stats = p.save_render(n, board, stats)
        job.update(state="done", progress=1.0, stage="완료", stats=stats)
    except Exception as e:  # noqa: BLE001
        log.error("렌더 실패: %s\n%s", e, traceback.format_exc())
        failed = {"status": "failed", "failed_stage": stage["name"], "error": str(e)[:600],
                  "elapsed_sec": round(time.time() - job["started_at"], 2)}
        p.save_render(n, board, failed)
        job.update(state="error", stage=stage["name"], error=str(e)[:600])


@app.post("/api/projects/{pid}/render")
def start_render(pid: str) -> dict:
    p = _project(pid)
    board = p.plan()
    if not board or not board.included():
        raise HTTPException(400, "구성안에 장면이 없습니다.")
    engine = ENGINES.get(board.engine)
    if not engine or not engine.available:
        raise HTTPException(400, f"이 생성 방식은 아직 쓸 수 없습니다. {engine.requirement if engine else ''}")
    if _job(pid).get("state") == "running":
        raise HTTPException(409, "이미 만드는 중입니다.")
    n, _ = p.new_render()
    job = {"state": "running", "progress": 0.0, "stage": "준비", "render": n, "started_at": time.time()}
    JOBS[pid] = job
    threading.Thread(target=_run_render, args=(p, board, n, job), daemon=True).start()
    return job


@app.get("/api/projects/{pid}/render")
def render_status(pid: str) -> dict:
    _project(pid)
    job = dict(_job(pid))
    if job.get("state") == "running":
        job["elapsed_sec"] = round(time.time() - job["started_at"], 1)
    return job


@app.get("/api/projects/{pid}/renders/{n}/video")
def video(pid: str, n: int, download: int = 0) -> FileResponse:
    p = _project(pid)
    path = p.render_dir(n) / "video.mp4"
    if not path.exists():
        raise HTTPException(404, "영상을 찾을 수 없습니다.")
    if download:
        return FileResponse(path, media_type="video/mp4", filename=f"photoreel-{pid[:6]}-{n}.mp4")
    return FileResponse(path, media_type="video/mp4")


def _warm_up() -> None:
    assets.ensure_all()
    semantic.get_clip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import uvicorn

    host = os.environ.get("PHOTOREEL_HOST", "127.0.0.1")
    port = int(os.environ.get("PHOTOREEL_PORT", "8765"))
    threading.Thread(target=_warm_up, daemon=True).start()
    print(f"\n  photoreel 실행 중 → http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
