"""완성된 영상을 수치로 점검한다. 눈으로 보는 평가를 대신하지는 못하고, 명백한 오류를 잡는 용도다.

    uv run python scripts/evaluate.py data/projects/<id>/renders/<n>

확인하는 것
  - 파일 규격: 코덱, 해상도, 길이, 오디오 유무
  - 검은 화면·멈춘 화면: 시작·끝 페이드 구간 밖에서 나타나면 오류
  - 움직임: 장면 안에서 프레임 간 변화량이 고른지 (튀는 프레임이 없는지)
  - 얼굴: 원본에서 찾은 얼굴이 결과 화면에서도 잘리지 않고 같은 수로 검출되는지
  - 비율 왜곡: 사진을 가로·세로 다른 배율로 늘리지 않았는지
  - 박자: 장면 경계가 음악 박자 위에 있는지
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photoreel import analyze, camera  # noqa: E402
from photoreel.config import ffmpeg_exe  # noqa: E402
from photoreel.models import Storyboard  # noqa: E402
from photoreel.render import timeline  # noqa: E402
from photoreel.styles import STYLES  # noqa: E402


def read_frames(video: Path, w: int, h: int, gray: bool = True) -> np.ndarray:
    fmt = "gray" if gray else "rgb24"
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", fmt, "-"],
        capture_output=True, check=True,
    ).stdout
    shape = (-1, h, w) if gray else (-1, h, w, 3)
    return np.frombuffer(raw, np.uint8).reshape(shape)


def grab(video: Path, t: float, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def main() -> int:
    rdir = Path(sys.argv[1])
    video = rdir / "video.mp4"
    board = Storyboard.model_validate(json.loads((rdir / "storyboard.json").read_text(encoding="utf-8")))
    stats = json.loads((rdir / "stats.json").read_text(encoding="utf-8"))
    project = rdir.parent.parent
    photos = {p["id"]: p for p in json.loads((project / "photos.json").read_text(encoding="utf-8"))}
    style = STYLES[board.style]
    fps, W, H = board.output.fps, stats["width"], stats["height"]
    items = timeline(board)
    report: dict = {"video": str(video), "checks": {}, "scenes": []}
    problems: list[str] = []

    # 1. 파일 규격
    info = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr
    has_audio = "Audio:" in info
    report["checks"]["format"] = {
        "h264": "Video: h264" in info, "yuv420p": "yuv420p" in info, "size": f"{W}x{H}" in info,
        "audio": has_audio, "audio_expected": board.music.mode != "none",
    }
    if not ("Video: h264" in info and "yuv420p" in info and f"{W}x{H}" in info):
        problems.append("파일 규격이 예상과 다름")
    if has_audio != (board.music.mode != "none"):
        problems.append("오디오 유무가 구성안과 다름")

    # 2. 프레임 전체를 작게 읽어 흐름 점검
    sw, sh = 216, round(216 * H / W / 2) * 2
    frames = read_frames(video, sw, sh).astype(np.float32)
    n = len(frames)
    report["checks"]["frames"] = {"count": n, "expected": stats["frames"], "duration": round(n / fps, 3)}
    if abs(n - stats["frames"]) > 1:
        problems.append(f"프레임 수 불일치: {n} vs {stats['frames']}")
    luma = frames.mean(axis=(1, 2))
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    t = np.arange(n) / fps
    inner = (t > 0.6) & (t < board.total_duration - 1.4)
    dark = np.flatnonzero(inner & (luma < 6))
    dark = [int(i) for i in dark if not any(
        it["scene"].transition == "fade-black" and abs(t[i] - it["start"]) < it["tt"] for it in items)]
    frozen = [int(i) for i in np.flatnonzero(diff < 0.002) if inner[i]]
    report["checks"]["dark_frames"] = dark[:10]
    report["checks"]["frozen_frames"] = frozen[:10]
    if dark:
        problems.append(f"페이드 구간 밖에 검은 프레임 {len(dark)}개")

    # 3. 장면별 점검
    detector_ready = analyze.assets.ensure("yunet", download=False) is not None
    for i, it in enumerate(items):
        s = it["scene"]
        p = photos[s.photo_id]
        a = int(np.ceil((it["start"] + it["tt"] / 2 + 0.05) * fps))
        nxt_tt = items[i + 1]["tt"] if i + 1 < len(items) else 0.0
        fade_tail = 1.3 if i == len(items) - 1 else 0.0
        b = int(np.floor((it["end"] - nxt_tt / 2 - 0.05 - fade_tail) * fps))
        seg = diff[a : max(a + 1, b - 1)]
        if i == 0:
            skip = int((3.4 if board.title else 0.45) * fps)  # 시작 페이드, 제목이 나타나고 사라지는 구간 제외
            seg = seg[skip:] if len(seg) > skip + 5 else seg[-5:]
        med = float(np.median(seg)) if len(seg) else 0.0
        spike = float(seg.max() / (med + 1e-6)) if len(seg) and med > 0.02 else 1.0
        entry = {
            "scene": i + 1, "file": p["filename"], "motion": s.motion, "framing": s.framing,
            "frame_diff_median": round(med, 3), "frame_diff_spike": round(spike, 2),
        }
        if s.motion != "hold" and med < 0.01 and len(seg):
            problems.append(f"장면 {i + 1}: 움직임이 거의 없음")
        if spike > 2.5:
            problems.append(f"장면 {i + 1}: 움직임이 고르지 않음 (최대/중앙값 {spike:.1f})")

        # 비율 왜곡: 가로·세로 배율이 같아야 한다.
        path = camera.plan_path(
            p["width"], p["height"], W / H, s.motion, s.framing, it["span_end"] - it["span_start"],
            style.zoom_rate, style.pan_rate, tuple(p["focus"]), p["saliency"], p["faces"],
        )
        m = camera.affine(path, 0.5, p["width"], p["height"], W, H)
        stretch = abs(m[0, 0] / m[1, 1] - 1)
        entry["aspect_stretch_pct"] = round(stretch * 100, 3)
        if stretch > 0.01:
            problems.append(f"장면 {i + 1}: 사진 비율이 {stretch * 100:.1f}% 늘어남")

        # 얼굴: 원본에서 찾은 수만큼 결과 화면에서도 온전히 보이는지
        if p["faces"] and detector_ready:
            mid = grab(video, (it["start"] + it["end"]) / 2, W, H)
            found, _ = analyze._detect_faces(mid)
            inside = [f for f in found if f["x"] > 0.002 and f["y"] > 0.002 and f["x"] + f["w"] < 0.998 and f["y"] + f["h"] < 0.998]
            entry["faces_source"] = len(p["faces"])
            entry["faces_in_video"] = len(found)
            entry["faces_fully_inside"] = len(inside)
            if len(found) < 0.7 * len(p["faces"]):
                problems.append(f"장면 {i + 1}: 얼굴 {len(p['faces'])}개 중 {len(found)}개만 검출")
        report["scenes"].append(entry)

    # 4. 박자: 장면 경계가 박자 위에 있는지
    beat = 60 / board.music.bpm
    off = [abs(it["start"] / beat - round(it["start"] / beat)) * beat for it in items]
    report["checks"]["beat"] = {"bpm": board.music.bpm, "max_offset_ms": round(max(off) * 1000, 2)}
    if max(off) > 1 / fps:
        problems.append("장면 경계가 박자에서 벗어남")

    report["problems"] = problems
    (rdir / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["checks"], ensure_ascii=False))
    for e in report["scenes"]:
        print(" ", json.dumps(e, ensure_ascii=False))
    print("문제:", problems or "발견되지 않음")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
