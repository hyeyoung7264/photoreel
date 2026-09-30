"""구성안을 MP4로 만든다.

프레임을 직접 그려서(서브픽셀 단위 이동) ffmpeg에 넘겨 H.264로 인코딩한다.
전환·색보정·제목·페이드는 모두 일반 영상 편집 처리다.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import assets, music as music_mod
from .config import ffmpeg_exe
from .engines import ENGINES, ClipContext
from .models import Storyboard
from .styles import STYLES

Progress = Callable[[float, str], None]


# ---------- 색보정 (사진당 한 번) ----------

def _curve(x: np.ndarray, contrast: float) -> np.ndarray:
    return np.clip((x - 0.5) * contrast + 0.5, 0, 1)


def _saturate(img: np.ndarray, amount: float) -> np.ndarray:
    luma = (img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
    return np.clip(luma + (img - luma) * amount, 0, 1)


def apply_grade(rgb: np.ndarray, grade: str) -> np.ndarray:
    if grade == "neutral":
        return rgb
    img = rgb.astype(np.float32) / 255.0
    if grade == "warm":
        img = _saturate(img * np.array([1.035, 1.0, 0.95], dtype=np.float32), 1.05)
    elif grade == "vivid":
        img = _saturate(_curve(img, 1.07), 1.16)
    elif grade == "cine":
        luma = (img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
        tint = np.array([-0.018, 0.004, 0.026], dtype=np.float32) * (1 - luma) + np.array(
            [0.022, 0.004, -0.02], dtype=np.float32
        ) * luma
        img = _saturate(_curve(img + tint, 1.06), 0.92)
    elif grade == "film":
        img = _saturate(img, 0.84) * np.array([1.02, 1.0, 0.95], dtype=np.float32)
        img = 0.06 + np.clip(img, 0, 1) * 0.89
    elif grade == "bw":
        luma = (img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
        img = np.repeat(_curve(luma, 1.1), 3, axis=2)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------- 전환 ----------

def _ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def blend(a: np.ndarray, b: np.ndarray, kind: str, u: float) -> np.ndarray:
    """장면 a에서 b로 넘어가는 중간 프레임. u는 0(전부 a)~1(전부 b)."""
    if kind == "crossfade":
        e = _ease(u)
        return cv2.addWeighted(a, 1 - e, b, e, 0)
    if kind == "fade-black":
        if u < 0.5:
            return cv2.convertScaleAbs(a, alpha=1 - _ease(u * 2))
        return cv2.convertScaleAbs(b, alpha=_ease(u * 2 - 1))
    if kind in ("push-left", "push-up"):
        e = _ease(u)
        h, w = a.shape[:2]
        out = np.empty_like(a)
        if kind == "push-left":
            off = int(round(e * w))
            out[:, : w - off] = a[:, off:]
            out[:, w - off :] = b[:, :off]
        else:
            off = int(round(e * h))
            out[: h - off] = a[off:]
            out[h - off :] = b[:off]
        return out
    return b if u >= 0.5 else a


# ---------- 제목 ----------

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = assets.ensure("font")
    if path is not None:
        try:
            font = ImageFont.truetype(str(path), size)
            try:
                font.set_variation_by_axes([700])
            except Exception:  # noqa: BLE001
                pass
            return font
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.splitlines() or [text]:
        cur = ""
        for word in para.split(" "):
            trial = (cur + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines[:3]


def make_title(text: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    """제목 레이어. (rgb*alpha, alpha, 좌상단 위치)"""
    text = text.strip()
    if not text:
        return None
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    size = max(10, int(w * 0.088))
    smallest = max(10, int(w * 0.04))
    while True:
        font = _font(size)
        lines = _wrap(draw, text, font, int(w * 0.84))
        fits = all(draw.textlength(line, font=font) <= w * 0.86 for line in lines) and len(lines) <= 2
        if fits or size <= smallest:
            break
        size = max(smallest, int(size * 0.9))
    line_h = int(size * 1.32)
    y = int(h * 0.5 - line_h * len(lines) / 2)
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    for i, line in enumerate(lines):
        x = (w - draw.textlength(line, font=font)) / 2
        sdraw.text((x, y + i * line_h), line, font=font, fill=(0, 0, 0, 170))
        draw.text((x, y + i * line_h), line, font=font, fill=(255, 255, 255, 255))
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.16))
    layer = Image.alpha_composite(shadow, layer)
    arr = np.asarray(layer).astype(np.float32) / 255.0
    ys, xs = np.nonzero(arr[..., 3] > 0.003)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = arr[y0:y1, x0:x1, 3:4]
    return arr[y0:y1, x0:x1, :3] * alpha * 255.0, alpha, (int(x0), int(y0))


def _vignette_mask(w: int, h: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    mask = 1 - strength * np.clip((d - 0.35) / 0.65, 0, 1) ** 1.6
    return np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)


def _grain_frames(w: int, h: int, count: int = 6) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    out = []
    for _ in range(count):
        g = rng.normal(127.5, 42, (h // 2, w // 2)).clip(0, 255).astype(np.uint8)
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(np.repeat(g[..., None], 3, axis=2))
    return out


# ---------- 타임라인 ----------

def timeline(board: Storyboard) -> list[dict]:
    """장면별 시작·끝과 전환 구간. 전환은 장면 경계(박자)를 가운데에 두고 겹친다."""
    style = STYLES[board.style]
    scenes = board.included()
    items = []
    for i, s in enumerate(scenes):
        tt = 0.0
        if i > 0 and s.transition != "cut":
            base = style.transition_sec if style.transition_sec > 0 else 0.35
            if s.transition == "fade-black":
                base = max(base, 0.7)
            tt = min(base, 0.6 * s.duration, 0.6 * scenes[i - 1].duration)
        items.append({"scene": s, "start": s.start, "end": s.start + s.duration, "tt": tt})
    for i, it in enumerate(items):
        nxt = items[i + 1]["tt"] if i + 1 < len(items) else 0.0
        it["span_start"] = it["start"] - it["tt"] / 2
        it["span_end"] = it["end"] + nxt / 2
    return items


def render(
    board: Storyboard,
    photos: dict[str, dict],
    proxies: dict[str, Path],
    out_path: Path,
    work_dir: Path,
    progress: Progress | None = None,
) -> dict:
    """구성안대로 영상을 만든다. 단계별 소요 시간과 결과 정보를 돌려준다."""
    progress = progress or (lambda f, s: None)
    t_start = time.time()
    engine = ENGINES[board.engine]
    style = STYLES[board.style]
    w, h, fps = board.output.width // 2 * 2, board.output.height // 2 * 2, board.output.fps
    items = timeline(board)
    if not items:
        raise ValueError("영상에 넣을 장면이 없습니다.")
    total = board.total_duration
    n_frames = max(1, int(round(total * fps)))
    work_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    audio_path = None
    audio_info = None
    if board.music.mode == "synth":
        progress(0.01, "음악 합성")
        t0 = time.time()
        audio_path = work_dir / "music.wav"
        audio_info = music_mod.synthesize(audio_path, n_frames / fps, board.music.mood, board.music.bpm)
        timings["music_sec"] = round(time.time() - t0, 2)

    title = make_title(board.title, w, h)
    title_end = min(items[0]["end"] - 0.15, 3.2)
    vignette = _vignette_mask(w, h, style.vignette) if style.vignette > 0 else None
    grain = _grain_frames(w, h) if style.grain > 0 else None
    fade_in, fade_out = 0.35, min(1.2, items[-1]["scene"].duration * 0.5)

    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "pipe:0"]
    if audio_path:
        cmd += ["-i", str(audio_path)]
    cmd += ["-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart"]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [str(out_path)]
    log_path = work_dir / "ffmpeg.log"

    clips: dict[int, object] = {}

    def clip(i: int):
        if i not in clips:
            it = items[i]
            s = it["scene"]
            img = np.asarray(Image.open(proxies[s.photo_id]).convert("RGB"))
            ctx = ClipContext(
                image=apply_grade(img, board.grade), analysis=photos[s.photo_id], motion=s.motion,
                framing=s.framing, seconds=it["span_end"] - it["span_start"], style=style,
                out_w=w, out_h=h, fps=fps,
            )
            clips[i] = engine.make_clip(ctx)
        for old in [k for k in clips if k < i - 1]:
            del clips[old]
        return clips[i]

    def scene_frame(i: int, t: float) -> np.ndarray:
        it = items[i]
        p = (t - it["span_start"]) / max(1e-6, it["span_end"] - it["span_start"])
        return clip(i).frame(p)

    t0 = time.time()
    cur = 0
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            for f in range(n_frames):
                t = f / fps
                while cur + 1 < len(items) and t >= items[cur]["end"]:
                    cur += 1
                it = items[cur]
                nxt = items[cur + 1] if cur + 1 < len(items) else None
                if it["tt"] > 0 and t < it["start"] + it["tt"] / 2:
                    u = (t - it["span_start"]) / it["tt"]
                    frame = blend(scene_frame(cur - 1, t), scene_frame(cur, t), it["scene"].transition, u)
                elif nxt and nxt["tt"] > 0 and t >= nxt["span_start"]:
                    u = (t - nxt["span_start"]) / nxt["tt"]
                    frame = blend(scene_frame(cur, t), scene_frame(cur + 1, t), nxt["scene"].transition, u)
                else:
                    frame = scene_frame(cur, t)
                if not frame.flags.writeable or not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame).copy()

                if vignette is not None:
                    frame = cv2.multiply(frame, vignette, scale=1 / 255.0)
                if grain is not None:
                    g = style.grain * 2.2
                    frame = cv2.addWeighted(frame, 1.0, grain[f % len(grain)], g, -127.5 * g)
                if title is not None and t < title_end:
                    a = min(1.0, max(0.0, (t - 0.25) / 0.4), max(0.0, (title_end - t) / 0.5))
                    if a > 0:
                        rgb, alpha, (x0, y0) = title
                        roi = frame[y0 : y0 + alpha.shape[0], x0 : x0 + alpha.shape[1]].astype(np.float32)
                        roi = roi * (1 - alpha * a) + rgb * a
                        frame[y0 : y0 + alpha.shape[0], x0 : x0 + alpha.shape[1]] = np.clip(roi, 0, 255).astype(np.uint8)
                k = 1.0
                if t < fade_in:
                    k = _ease(t / fade_in)
                elif t > total - fade_out:
                    k = _ease(max(0.0, (total - t) / fade_out))
                if k < 0.999:
                    frame = cv2.convertScaleAbs(frame, alpha=k)
                proc.stdin.write(frame.tobytes())
                if f % 15 == 0:
                    progress(0.03 + 0.92 * f / n_frames, f"장면 {cur + 1}/{len(items)} 그리는 중")
            proc.stdin.close()
            progress(0.96, "인코딩 마무리")
            code = proc.wait(timeout=600)
        except BrokenPipeError:
            code = proc.wait(timeout=30)
        except BaseException:
            proc.kill()
            raise
    if code != 0 or not out_path.exists():
        detail = log_path.read_text(errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg 인코딩 실패 (코드 {code}): {detail}")
    timings["frames_sec"] = round(time.time() - t0, 2)
    timings["total_sec"] = round(time.time() - t_start, 2)
    progress(1.0, "완료")
    return {
        "engine": engine.public(),
        "width": w,
        "height": h,
        "fps": fps,
        "frames": n_frames,
        "duration": round(n_frames / fps, 3),
        "scenes": len(items),
        "bytes": out_path.stat().st_size,
        "timings": timings,
        "audio": audio_info,
        "cost_usd": 0.0,
        "cost_note": "외부 유료 서비스를 호출하지 않았습니다 (전부 이 컴퓨터에서 처리).",
    }
