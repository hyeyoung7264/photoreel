"""장면 하나를 '움직이는 화면'으로 만드는 방식(엔진)들.

세 갈래를 구분한다.
  편집        edit      원본 사진을 그대로 쓰는 편집 (자르기·확대·이동·전환·색보정)      ← 동작
              parallax  편집 + 깊이 추정 모델로 가까운 것과 먼 것을 다른 속도로 이동     ← 동작 (실험)
  움직임 생성 generate  사진 속 장면에 움직임을 만들어 넣는 생성 (image-to-video 모델)   ← 자리만 있음
  스타일 변환 stylize   그림체를 바꾸는 변환 (image-to-image 모델)                        ← 자리만 있음

parallax도 픽셀을 새로 만들지는 않는다. 이미 있는 픽셀을 깊이에 따라 조금씩 밀 뿐이다.

렌더러는 장면마다 `frame(p)` 를 가진 클립만 받으면 되므로, 새 엔진은 클립을 돌려주는
`make_clip` 하나만 구현하면 끼워 넣을 수 있다. 생성 서비스가 mp4를 돌려준다면 VideoFileClip으로 감싸면 된다.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import camera, depth as depth_mod
from .config import ffmpeg_exe
from .styles import Style


class EngineUnavailable(RuntimeError):
    pass


@dataclass
class ClipContext:
    image: np.ndarray  # 색보정이 끝난 RGB 사진
    analysis: dict
    motion: str
    framing: str
    seconds: float  # 이 장면이 화면에 보이는 전체 시간 (전환 겹침 포함)
    style: Style
    out_w: int
    out_h: int
    fps: int
    depth_cache: Path | None = None  # 깊이 지도를 저장해 둘 파일 (다시 만들 때 재사용)


class PhotoClip:
    """사진 위에서 창을 움직여 프레임을 만든다. 픽셀을 새로 만들지 않는다."""

    def __init__(self, ctx: ClipContext) -> None:
        a = ctx.analysis
        h, w = ctx.image.shape[:2]
        self.out_w, self.out_h = ctx.out_w, ctx.out_h
        self.path = camera.plan_path(
            w, h, ctx.out_w / ctx.out_h, ctx.motion, ctx.framing, ctx.seconds,
            ctx.style.zoom_rate, ctx.style.pan_rate, tuple(a["focus"]), a["saliency"], a["faces"],
        )
        # 원본이 필요 이상으로 크면 미리 줄여 둔다 (축소 시 생기는 자글거림 방지, 속도).
        scales = [camera.affine(self.path, p, w, h, ctx.out_w, ctx.out_h)[0, 0] for p in (0.0, 1.0)]
        f = float(np.sqrt(min(scales) * max(scales)))
        src = ctx.image
        if f < 0.98:
            src = cv2.resize(src, (max(2, round(w * f)), max(2, round(h * f))), interpolation=cv2.INTER_AREA)
        self.src = np.ascontiguousarray(src)
        self.background = None
        if self.path["framing"] == "fit-blur":
            self.background = self._blurred_background(ctx.image)

    def _blurred_background(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ww, wh = camera.cover_window(w, h, self.out_w / self.out_h)
        cw, ch = int(ww * w), int(wh * h)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        small = cv2.resize(image[y0 : y0 + ch, x0 : x0 + cw], (max(2, self.out_w // 8), max(2, self.out_h // 8)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(2.0, self.out_w / 8 / 18))
        bg = cv2.resize(small, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC)
        return cv2.convertScaleAbs(bg, alpha=0.62)

    def frame(self, p: float) -> np.ndarray:
        sh, sw = self.src.shape[:2]
        m = camera.affine(self.path, p, sw, sh, self.out_w, self.out_h)
        if self.background is None:
            return cv2.warpAffine(self.src, m, (self.out_w, self.out_h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        dst = self.background.copy()
        cv2.warpAffine(self.src, m, (self.out_w, self.out_h), dst=dst, flags=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_TRANSPARENT)
        return dst


class ParallaxClip(PhotoClip):
    """PhotoClip에 깊이에 따른 시차를 더한다. 가까운 것은 많이, 먼 것은 적게 움직인다.

    깊이 경계에서는 배경이 늘어나 보일 수 있다 (가려져 있던 부분을 새로 그려 넣지 않기 때문).
    그래서 이동량을 작게 잡고, 얼굴 영역은 깊이를 평평하게 만들어 얼굴 모양이 변하지 않게 한다.
    """

    SHIFT = 0.013  # 팬: 가장 가까운 것과 기준면의 이동 차이 (화면 폭 대비, 한쪽 끝에서)
    SCALE = 0.035  # 줌: 가장 가까운 것이 기준면보다 더 커지는 비율

    def __init__(self, ctx: ClipContext) -> None:
        super().__init__(ctx)
        d = None
        if ctx.depth_cache is not None and ctx.depth_cache.exists():
            d = np.load(ctx.depth_cache)
        if d is None:
            d = depth_mod.estimate(ctx.image)
            if ctx.depth_cache is not None:
                ctx.depth_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(ctx.depth_cache, d.astype(np.float16))
        sh, sw = self.src.shape[:2]
        d = cv2.resize(d.astype(np.float32), (sw, sh), interpolation=cv2.INTER_CUBIC)
        k = max(3, int(sw * 0.004) | 1)
        d = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))  # 앞쪽 물체 가장자리를 조금 넓힌다
        d = cv2.GaussianBlur(d, (0, 0), max(1.0, sw * 0.004))
        for f in ctx.analysis.get("faces", []):
            x0, x1 = int(max(0, (f["x"] - 0.3 * f["w"]) * sw)), int(min(sw, (f["x"] + 1.3 * f["w"]) * sw))
            y0, y1 = int(max(0, (f["y"] - 0.4 * f["h"]) * sh)), int(min(sh, (f["y"] + 1.4 * f["h"]) * sh))
            if x1 - x0 > 2 and y1 - y0 > 2:
                d[y0:y1, x0:x1] = float(np.median(d[y0:y1, x0:x1]))
        if ctx.analysis.get("faces"):
            d = cv2.GaussianBlur(d, (0, 0), max(1.0, sw * 0.006))
        self.depth = np.clip(d, 0, 1)
        self.plane = float(np.median(self.depth))  # 이 깊이에 있는 것은 시차 없이 움직인다
        self.u = np.arange(self.out_w, dtype=np.float32)[None, :]
        self.v = np.arange(self.out_h, dtype=np.float32)[:, None]

    def frame(self, p: float) -> np.ndarray:
        if self.background is not None:
            return super().frame(p)
        sh, sw = self.src.shape[:2]
        w, h = self.out_w, self.out_h
        m = camera.affine(self.path, p, sw, sh, w, h)
        sx, sy, tx, ty = m[0, 0], m[1, 1], m[0, 2], m[1, 2]
        rel = cv2.warpAffine(self.depth, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rel -= self.plane
        k = 2 * float(np.clip(p, 0, 1)) - 1
        base_x, base_y = (self.u - tx) / sx, (self.v - ty) / sy
        motion = self.path["motion"]
        if motion in ("pan-up", "pan-down"):
            sign = 1.0 if motion == "pan-down" else -1.0
            map_x = np.broadcast_to(base_x, (h, w)).astype(np.float32)
            map_y = (base_y + rel * (sign * k * self.SHIFT * w / sy)).astype(np.float32)
        elif motion in ("zoom-in", "zoom-out"):
            sign = 1.0 if motion == "zoom-in" else -1.0
            g = rel * (sign * k * self.SCALE)
            map_x = (base_x - g * ((self.u - w / 2) / sx)).astype(np.float32)
            map_y = (base_y - g * ((self.v - h / 2) / sy)).astype(np.float32)
        else:  # 옆으로 훑기, 고정(살짝 옆으로 흔들림)
            sign = -1.0 if motion == "pan-left" else 1.0
            map_x = (base_x + rel * (sign * k * self.SHIFT * w / sx)).astype(np.float32)
            map_y = np.broadcast_to(base_y, (h, w)).astype(np.float32)
        return cv2.remap(self.src, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


class VideoFileClip:
    """이미 만들어진 영상 파일을 장면으로 쓴다. 생성 엔진이 돌려준 mp4를 끼워 넣는 자리."""

    def __init__(self, path: Path, out_w: int, out_h: int, fps: int, seconds: float) -> None:
        self.out_w, self.out_h = out_w, out_h
        count = max(1, round(seconds * fps))
        vf = (
            f"fps={fps},scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},format=rgb24"
        )
        raw = subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vf", vf, "-frames:v", str(count), "-f", "rawvideo", "-"],
            capture_output=True, check=True,
        ).stdout
        size = out_w * out_h * 3
        n = len(raw) // size
        if n == 0:
            raise EngineUnavailable(f"영상에서 프레임을 읽지 못함: {path}")
        self.frames = np.frombuffer(raw[: n * size], dtype=np.uint8).reshape(n, out_h, out_w, 3)

    def frame(self, p: float) -> np.ndarray:
        i = int(np.clip(round(p * (len(self.frames) - 1)), 0, len(self.frames) - 1))
        return self.frames[i]


@dataclass(frozen=True)
class Engine:
    id: str
    kind: str  # edit | generate | stylize
    name: str
    uses_generative_model: bool
    disclosure: str
    ai_models: str = ""  # 생성은 아니지만 쓰인 AI 모델 (있는 경우)
    requirement: str = ""
    check: Callable[[], bool] | None = None  # None이면 아직 구현되지 않은 자리

    @property
    def available(self) -> bool:
        return bool(self.check and self.check())

    def make_clip(self, ctx: ClipContext):
        if not self.available:
            raise EngineUnavailable(f"‘{self.name}’ 엔진은 지금 쓸 수 없습니다. {self.requirement}")
        if self.id == "edit":
            return PhotoClip(ctx)
        if self.id == "parallax":
            return ParallaxClip(ctx)
        raise EngineUnavailable(f"‘{self.name}’ 엔진은 아직 연결되지 않았습니다. {self.requirement}")

    def public(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "name": self.name, "available": self.available,
            "uses_generative_model": self.uses_generative_model, "ai_models": self.ai_models,
            "disclosure": self.disclosure, "requirement": self.requirement,
        }


ENGINES = {
    e.id: e
    for e in [
        Engine(
            id="edit",
            kind="edit",
            name="원본 사진 편집",
            uses_generative_model=False,
            disclosure=(
                "원본 사진을 자르고 천천히 확대·이동시키고, 장면 사이를 전환 효과로 잇고, 색감을 보정한 영상입니다. "
                "AI가 장면을 새로 그리거나 사진 속 대상을 움직이게 한 부분은 없습니다."
            ),
            check=lambda: True,
        ),
        Engine(
            id="parallax",
            kind="edit",
            name="입체감 있는 움직임 (실험)",
            uses_generative_model=False,
            ai_models="깊이 추정 모델 Depth Anything V2 Small (이 컴퓨터에서 실행)",
            disclosure=(
                "원본 사진 편집에 더해, 깊이 추정 AI 모델로 가까운 것과 먼 것을 구분해 서로 다른 속도로 움직였습니다. "
                "새 장면이나 대상을 만들어 내지는 않았고 사진 속 대상이 스스로 움직이지도 않습니다. "
                "앞뒤 물체의 경계가 늘어나 보이는 왜곡이 생길 수 있습니다."
            ),
            requirement="깊이 추정 모델 파일(약 100MB)을 내려받아야 합니다. 서버를 처음 켤 때 자동으로 받습니다.",
            check=lambda: depth_mod.available(),
        ),
        Engine(
            id="generate",
            kind="generate",
            name="사진에 움직임 생성",
            uses_generative_model=True,
            disclosure="사진을 첫 장면으로 삼아 image-to-video 생성 모델이 몇 초짜리 움직이는 영상을 만듭니다.",
            requirement="외부 생성 서비스의 계정·API 키와 유료 호출 승인, 사진 외부 전송 동의가 필요합니다.",
        ),
        Engine(
            id="stylize",
            kind="stylize",
            name="그림체 변환",
            uses_generative_model=True,
            disclosure="사진을 애니메이션 등 다른 그림체로 다시 그린 뒤 영상으로 만듭니다. 인물·장소가 달라 보일 수 있습니다.",
            requirement="외부 이미지 생성 서비스의 계정·API 키와 유료 호출 승인, 사진 외부 전송 동의가 필요합니다.",
        ),
    ]
}
