"""사진 한 장의 깊이(가까움·멂)를 추정한다. '입체감 있는 움직임' 엔진에서만 쓴다.

Depth Anything V2 Small 모델을 이 컴퓨터에서 실행한다. 사진을 외부로 보내지 않는다.
이 모델은 이미 있는 픽셀이 카메라에서 얼마나 먼지를 추정할 뿐, 새 이미지를 만들지 않는다.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from . import assets

_session = None
_lock = threading.Lock()
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LONG_SIDE = 518


def available(download: bool = False) -> bool:
    return assets.ensure("depth", download=download) is not None


def _get_session():
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                import onnxruntime as ort

                path = assets.ensure("depth")
                if path is None:
                    raise RuntimeError("깊이 추정 모델을 준비하지 못함")
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                _session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return _session


def estimate(rgb: np.ndarray) -> np.ndarray:
    """가까울수록 1, 멀수록 0인 깊이 지도 (입력보다 작은 해상도, float32)."""
    h, w = rgb.shape[:2]
    scale = LONG_SIDE / max(h, w)
    tw, th = max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14)
    small = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((small - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
    session = _get_session()
    with _lock:
        out = session.run(None, {session.get_inputs()[0].name: x})[0]
    d = np.squeeze(out).astype(np.float32)
    lo, hi = np.percentile(d, [2, 98])
    return np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
