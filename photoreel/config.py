"""경로와 실행 환경 설정."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PHOTOREEL_DATA", ROOT / "data"))
ASSET_DIR = Path(os.environ.get("PHOTOREEL_ASSETS", ROOT / "assets_cache"))
WEB_DIR = Path(__file__).resolve().parent / "web"


def ffmpeg_exe() -> str:
    """ffmpeg 실행 파일. 환경변수 > PATH > imageio-ffmpeg 번들 순으로 찾는다."""
    env = os.environ.get("PHOTOREEL_FFMPEG")
    if env:
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
