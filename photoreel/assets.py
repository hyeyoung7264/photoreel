"""실행에 필요한 외부 파일(모델·폰트)을 내려받아 캐시한다.

모두 선택 사항이다. 내려받지 못하면 해당 기능만 단순한 방식으로 대체되고,
어떤 방식이 쓰였는지는 분석 결과에 기록된다.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import ASSET_DIR

log = logging.getLogger(__name__)
_lock = threading.Lock()


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    min_bytes: int
    note: str


ASSETS = {
    "yunet": Asset(
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        100_000,
        "얼굴 검출 (OpenCV Zoo YuNet, MIT)",
    ),
    "font": Asset(
        "NotoSansKR-wght.ttf",
        "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf",
        1_000_000,
        "제목 글꼴 (Noto Sans KR, OFL)",
    ),
    "clip_vision": Asset(
        "clip-vit-base-patch32-vision.onnx",
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
        100_000_000,
        "사진 내용 분류·유사도 (OpenAI CLIP ViT-B/32, MIT)",
    ),
    "clip_text": Asset(
        "clip-vit-base-patch32-text.onnx",
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
        100_000_000,
        "CLIP 텍스트 인코더",
    ),
    "clip_tokenizer": Asset(
        "clip-vit-base-patch32-tokenizer.json",
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
        100_000,
        "CLIP 토크나이저",
    ),
}


def path_of(key: str) -> Path:
    return ASSET_DIR / ASSETS[key].name


def ensure(key: str, download: bool = True) -> Path | None:
    """파일 경로를 돌려준다. 없고 내려받지도 못하면 None."""
    asset = ASSETS[key]
    dest = path_of(key)
    if dest.exists() and dest.stat().st_size >= asset.min_bytes:
        return dest
    if not download:
        return None
    with _lock:
        if dest.exists() and dest.stat().st_size >= asset.min_bytes:
            return dest
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            log.info("내려받는 중: %s (%s)", asset.name, asset.note)
            req = urllib.request.Request(asset.url, headers={"User-Agent": "photoreel/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size < asset.min_bytes:
                raise OSError(f"파일이 너무 작음: {tmp.stat().st_size} bytes")
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            log.warning("내려받기 실패 (%s): %s", asset.name, e)
            tmp.unlink(missing_ok=True)
            return None


def ensure_all() -> dict[str, bool]:
    return {key: ensure(key) is not None for key in ASSETS}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for k, ok in ensure_all().items():
        print(f"{'OK  ' if ok else '실패'} {k}: {ASSETS[k].note}")
