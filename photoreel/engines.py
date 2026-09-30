"""장면 하나를 '움직이는 화면'으로 만드는 방식(엔진)들.

세 가지를 구분한다.
  edit      원본 사진을 그대로 쓰는 편집 (자르기·확대·이동·전환·색보정)   ← 지금 실제로 동작
  generate  사진 속 장면에 움직임을 만들어 넣는 생성 (image-to-video 모델)  ← 자리만 있음
  stylize   그림체를 바꾸는 스타일 변환 (image-to-image 모델)               ← 자리만 있음

렌더러는 장면마다 `frame(p)` 를 가진 클립만 받으면 되므로, 새 엔진은 클립을 돌려주는
`make_clip` 하나만 구현하면 끼워 넣을 수 있다. 생성 서비스가 mp4를 돌려준다면 VideoFileClip으로 감싸면 된다.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import camera
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
    kind: str
    name: str
    available: bool
    uses_generative_model: bool
    disclosure: str
    requirement: str = ""

    def make_clip(self, ctx: ClipContext):
        if self.id == "edit":
            return PhotoClip(ctx)
        raise EngineUnavailable(f"‘{self.name}’ 엔진은 아직 연결되지 않았습니다. {self.requirement}")

    def public(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "name": self.name, "available": self.available,
            "uses_generative_model": self.uses_generative_model, "disclosure": self.disclosure,
            "requirement": self.requirement,
        }


ENGINES = {
    e.id: e
    for e in [
        Engine(
            id="edit",
            kind="edit",
            name="원본 사진 편집",
            available=True,
            uses_generative_model=False,
            disclosure=(
                "원본 사진을 자르고 천천히 확대·이동시키고, 장면 사이를 전환 효과로 잇고, 색감을 보정한 영상입니다. "
                "AI가 장면을 새로 그리거나 사진 속 대상을 움직이게 한 부분은 없습니다."
            ),
        ),
        Engine(
            id="generate",
            kind="generate",
            name="사진에 움직임 생성",
            available=False,
            uses_generative_model=True,
            disclosure="사진을 첫 장면으로 삼아 image-to-video 생성 모델이 몇 초짜리 움직이는 영상을 만듭니다.",
            requirement="외부 생성 서비스의 계정·API 키와 유료 호출 승인, 사진 외부 전송 동의가 필요합니다.",
        ),
        Engine(
            id="stylize",
            kind="stylize",
            name="그림체 변환",
            available=False,
            uses_generative_model=True,
            disclosure="사진을 애니메이션 등 다른 그림체로 다시 그린 뒤 영상으로 만듭니다. 인물·장소가 달라 보일 수 있습니다.",
            requirement="외부 이미지 생성 서비스의 계정·API 키와 유료 호출 승인, 사진 외부 전송 동의가 필요합니다.",
        ),
    ]
}
