"""화면 없이 폴더의 사진으로 영상을 만든다. 웹 화면과 같은 코드를 쓴다.

    uv run python -m photoreel.cli samples/jeju --concept "잔잔하고 따뜻한 여행 기록" --out out.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import planner, render
from .models import ASPECTS, DEFAULT_ASPECT
from .store import IMAGE_EXT, Project


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="사진 폴더 → 영상")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--concept", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--aspect", default=DEFAULT_ASPECT, choices=list(ASPECTS))
    ap.add_argument("--seconds", type=float, default=None, help="목표 길이(초). 생략하면 15~30초 범위에서 정함")
    ap.add_argument("--style", default=None)
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out.mp4"))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files:
        print(f"사진이 없습니다: {args.folder}", file=sys.stderr)
        return 1
    project = Project.create()
    t0 = time.time()
    for f in files:
        a = project.add_photo(f.name, f.read_bytes())
        tags = a["tags"]
        print(f"분석 {f.name}: {tags.get('scene')} / {tags.get('light')} / 얼굴 {len(a['faces'])}")
    analyze_sec = time.time() - t0

    board = planner.build(
        project.photos(), project.embeddings(), args.concept, title=args.title, aspect=args.aspect,
        target_duration=args.seconds, music=not args.no_music, style_id=args.style,
    )
    project.save_plan(board)
    names = {p["id"]: p["filename"] for p in project.photos()}
    print(f"\n스타일: {board.style} ({board.style_reason}) · 색감 {board.grade} · {board.total_duration:.1f}초")
    print(f"순서 기준: {board.order_basis}")
    for s in board.scenes:
        if s.included:
            print(f"  {s.start:5.1f}s  {names[s.photo_id]:<14} {s.duration:4.1f}s  {s.framing:<8} {s.motion:<9} "
                  f"{s.transition:<10} {s.reason}")
        else:
            print(f"  제외    {names[s.photo_id]:<14} {s.exclude_reason}")
    for note in board.notes + board.unsupported:
        print(f"  * {note}")
    if args.plan_only:
        return 0

    n, rdir = project.new_render()
    last = [0.0]

    def progress(frac: float, stage: str) -> None:
        if frac - last[0] >= 0.1 or frac >= 1.0:
            last[0] = frac
            print(f"  {frac * 100:3.0f}%  {stage}")

    proxies = {p["id"]: project.proxy_path(p["id"]) for p in project.photos()}
    stats = render.render(board, {p["id"]: p for p in project.photos()}, proxies, rdir / "video.mp4", rdir, progress)
    stats["timings"]["analyze_sec"] = round(analyze_sec, 2)
    project.save_render(n, board, stats)
    shutil.copyfile(rdir / "video.mp4", args.out)
    print(f"\n완료: {args.out}  ({stats['duration']}초, {stats['bytes'] / 1e6:.1f}MB)")
    print(json.dumps(stats["timings"], ensure_ascii=False))
    print(f"프로젝트 폴더: {project.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
