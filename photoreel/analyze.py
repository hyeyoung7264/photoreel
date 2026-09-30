"""사진 한 장을 분석한다. 전부 이 컴퓨터 안에서 실행된다.

측정하는 것: 촬영 시각(EXIF), 선명도·밝기·색감, 얼굴 위치, 시선이 갈 만한 영역,
내용 분류와 임베딩(CLIP 또는 대체 특징). 구성안과 렌더러가 이 결과를 쓴다.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import assets, semantic

log = logging.getLogger(__name__)
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

try:  # 아이폰 HEIC
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001
    pass

PROXY_MAX_SIDE = 3200
SALIENCY_GRID = 24
_face_lock = threading.Lock()


def load_image(path: Path) -> Image.Image:
    """EXIF 회전을 적용한 RGB 이미지."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def read_taken_at(path: Path) -> str | None:
    """EXIF 촬영 시각 (ISO 형식). 없으면 None. 파일 수정 시각은 믿지 않는다."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(0x8769)
            raw = sub.get(36867) or sub.get(36868) or exif.get(306)
        if not raw:
            return None
        return datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except Exception:  # noqa: BLE001
        return None


def make_proxy(img: Image.Image, dest: Path, max_side: int = PROXY_MAX_SIDE) -> tuple[int, int]:
    """렌더링용 사본. 방향이 바로잡혀 있고 너무 크지 않다."""
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(dest, "JPEG", quality=95, subsampling=0)
    return img.size


def _keep_face(f: dict, sw: int, sh: int) -> bool:
    """아주 작은 검출은 오검출이 많고 구도에도 영향이 없어 버린다."""
    return f["h"] * sh >= 0.035 * min(sw, sh)


def _detect_faces(rgb: np.ndarray) -> tuple[list[dict], str]:
    faces, method = _detect_faces_raw(rgb)
    h, w = rgb.shape[:2]
    return [f for f in faces if _keep_face(f, w, h)], method


def _detect_faces_raw(rgb: np.ndarray) -> tuple[list[dict], str]:
    h, w = rgb.shape[:2]
    scale = min(1.0, 1280 / max(h, w))
    small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    sh, sw = small.shape[:2]
    model = assets.ensure("yunet")
    if model is not None:
        try:
            with _face_lock:
                det = cv2.FaceDetectorYN.create(str(model), "", (sw, sh), 0.75, 0.3, 200)
                _, found = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            faces = []
            for f in found if found is not None else []:
                x, y, fw, fh = (float(v) for v in f[:4])
                faces.append(
                    {
                        "x": max(0.0, x / sw),
                        "y": max(0.0, y / sh),
                        "w": min(1.0, fw / sw),
                        "h": min(1.0, fh / sh),
                        "score": round(float(f[-1]), 3),
                    }
                )
            return faces, "yunet"
        except Exception as e:  # noqa: BLE001
            log.warning("YuNet 얼굴 검출 실패, Haar로 대체: %s", e)
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        found = cascade.detectMultiScale(gray, 1.1, 6, minSize=(max(24, sw // 30), max(24, sw // 30)))
        faces = [
            {"x": x / sw, "y": y / sh, "w": fw / sw, "h": fh / sh, "score": 0.5} for x, y, fw, fh in found
        ]
        return faces, "haar"
    except Exception:  # noqa: BLE001
        return [], "none"


def _saliency(rgb: np.ndarray, faces: list[dict]) -> np.ndarray:
    """시선이 갈 만한 영역의 거친 지도 (합이 1). Spectral residual + 얼굴 가중."""
    n = 64
    gray = cv2.cvtColor(cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    spec = np.fft.fft2(gray)
    log_amp = np.log(np.abs(spec) + 1e-8)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)
    sal /= sal.max() + 1e-8
    # 가장자리보다 가운데를 조금 더 믿는다.
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / (n - 1)
    sal *= 0.6 + 0.4 * np.exp(-(((xx - 0.5) ** 2) + ((yy - 0.5) ** 2)) / 0.18)
    sal = cv2.resize(sal, (SALIENCY_GRID, SALIENCY_GRID), interpolation=cv2.INTER_AREA)
    sal = sal / (sal.sum() + 1e-8)
    if faces:
        g = SALIENCY_GRID
        face_map = np.zeros((g, g), dtype=np.float32)
        for f in faces:
            # 얼굴 아래 몸통까지 조금 포함한다.
            x0, x1 = f["x"] - 0.25 * f["w"], f["x"] + 1.25 * f["w"]
            y0, y1 = f["y"] - 0.35 * f["h"], f["y"] + 2.2 * f["h"]
            c0, c1 = int(np.clip(x0 * g, 0, g - 1)), int(np.clip(np.ceil(x1 * g), 1, g))
            r0, r1 = int(np.clip(y0 * g, 0, g - 1)), int(np.clip(np.ceil(y1 * g), 1, g))
            face_map[r0:r1, c0:c1] += f["w"] * f["h"]
        if face_map.sum() > 0:
            face_map /= face_map.sum()
            sal = 0.3 * sal + 0.7 * face_map
    return sal / (sal.sum() + 1e-8)


def _colorfulness(rgb: np.ndarray) -> float:
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 255.0


def analyze(photo_id: str, original: Path, proxy: Path, filename: str) -> tuple[dict, np.ndarray]:
    """분석 결과(dict)와 임베딩을 돌려준다. proxy 파일도 여기서 만든다."""
    img = load_image(original)
    width, height = make_proxy(img, proxy)
    if img.size != (width, height):
        img = load_image(proxy)
    rgb = np.asarray(img)

    small = cv2.resize(rgb, (512, max(1, round(512 * height / width))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean() / 255.0)
    mean_rgb = small.reshape(-1, 3).mean(axis=0) / 255.0
    warmth = float(mean_rgb[0] - mean_rgb[2])

    faces, face_method = _detect_faces(rgb)
    sal = _saliency(rgb, faces)
    g = SALIENCY_GRID
    ys, xs = (np.arange(g) + 0.5) / g, (np.arange(g) + 0.5) / g
    focus = [float((sal.sum(axis=0) * xs).sum()), float((sal.sum(axis=1) * ys).sum())]
    if faces:
        area = np.array([f["w"] * f["h"] for f in faces])
        cx = np.array([f["x"] + f["w"] / 2 for f in faces])
        cy = np.array([f["y"] + f["h"] / 2 for f in faces])
        focus = [float((cx * area).sum() / area.sum()), float((cy * area).sum() / area.sum())]

    clip = semantic.get_clip()
    tags: dict = {}
    if clip is not None:
        emb = clip.embed_image(img)
        scene = clip.classify(emb, semantic.SCENE_LABELS)
        light = clip.classify(emb, semantic.LIGHT_LABELS)
        scale = clip.classify(emb, semantic.SCALE_LABELS)
        tags = {
            "scene": scene[0][0],
            "scene_conf": round(scene[0][1], 3),
            "scene_top": [[k, round(p, 3)] for k, p in scene[:3]],
            "light": light[0][0],
            "light_conf": round(light[0][1], 3),
            "scale": scale[0][0],
            "scale_conf": round(scale[0][1], 3),
        }
        semantic_method = "clip"
    else:
        emb = semantic.color_embedding(rgb)
        tags = {
            "scene": None,
            "light": "night" if brightness < 0.22 else ("dusk" if brightness < 0.4 and warmth > 0.08 else "day"),
            "scale": None,
        }
        semantic_method = "color-hist"

    result = {
        "id": photo_id,
        "filename": filename,
        "width": width,
        "height": height,
        "taken_at": read_taken_at(original),
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 3),
        "warmth": round(warmth, 3),
        "colorfulness": round(_colorfulness(small), 3),
        "faces": faces,
        "face_area": round(float(sum(f["w"] * f["h"] for f in faces)), 4),
        "focus": [round(focus[0], 3), round(focus[1], 3)],
        "saliency": [[round(float(v), 5) for v in row] for row in sal],
        "tags": tags,
        "methods": {"faces": face_method, "semantic": semantic_method},
    }
    return result, emb.astype(np.float32)


def make_thumb(proxy: Path, dest: Path, size: int = 480) -> None:
    img = Image.open(proxy)
    img.thumbnail((size, size), Image.LANCZOS)
    img.save(dest, "JPEG", quality=85)
