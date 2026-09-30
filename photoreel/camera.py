"""사진 위에서 화면(카메라 창)이 어떻게 움직일지 계산한다.

전부 기하 계산이다. 사진의 픽셀을 자르고 확대·이동할 뿐 새로 만들어 내지 않는다.
좌표는 사진 크기에 대한 비율(0~1)로 다룬다.
"""

from __future__ import annotations

import numpy as np

MAX_ZOOM = 1.35
FACE_PAD = (0.45, 0.7, 0.45, 1.1)  # 얼굴 상자 여유: 좌, 위, 우, 아래 (얼굴 크기 대비)


def cover_window(w: int, h: int, aspect: float) -> tuple[float, float]:
    """확대 없이 화면을 가득 채울 때 보이는 창의 크기 (사진 대비 비율)."""
    if w / h > aspect:
        return (h * aspect) / w, 1.0
    return 1.0, (w / aspect) / h


def face_box(faces: list[dict]) -> tuple[float, float, float, float] | None:
    """얼굴 전체를 감싸는 상자 (여유 포함)."""
    if not faces:
        return None
    l, t, r, b = FACE_PAD
    x0 = min(f["x"] - l * f["w"] for f in faces)
    y0 = min(f["y"] - t * f["h"] for f in faces)
    x1 = max(f["x"] + (1 + r) * f["w"] for f in faces)
    y1 = max(f["y"] + (1 + b) * f["h"] for f in faces)
    return max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)


def _integral(sal: np.ndarray, n: int = 96) -> np.ndarray:
    import cv2

    up = cv2.resize(np.asarray(sal, dtype=np.float32), (n, n), interpolation=cv2.INTER_LINEAR)
    up = up / (up.sum() + 1e-8)
    return np.pad(up.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def best_center(sal, win_w: float, win_h: float) -> tuple[float, float, float]:
    """창 안에 시선 영역이 가장 많이 들어오는 중심. (cx, cy, 담긴 비율)"""
    n = 96
    ii = _integral(sal, n)
    kw, kh = max(1, min(n, round(win_w * n))), max(1, min(n, round(win_h * n)))
    mass = ii[kh:, kw:] - ii[:-kh, kw:] - ii[kh:, :-kw] + ii[:-kh, :-kw]
    rows, cols = mass.shape
    # 같은 값이면 가운데에 가까운 쪽을 고른다.
    yy, xx = np.mgrid[0:rows, 0:cols]
    center_pull = -1e-4 * (((xx - (cols - 1) / 2) / n) ** 2 + ((yy - (rows - 1) / 2) / n) ** 2)
    r, c = np.unravel_index(np.argmax(mass + center_pull), mass.shape)
    return (c + kw / 2) / n, (r + kh / 2) / n, float(mass[r, c])


def _clamp_center(cx: float, cy: float, ww: float, wh: float) -> tuple[float, float]:
    return float(np.clip(cx, ww / 2, 1 - ww / 2)), float(np.clip(cy, wh / 2, 1 - wh / 2))


def _keep_box(cx: float, cy: float, ww: float, wh: float, box) -> tuple[float, float]:
    """가능하면 상자(얼굴)가 창 밖으로 나가지 않게 중심을 옮긴다."""
    if box is None:
        return cx, cy
    x0, y0, x1, y1 = box
    if x1 - x0 <= ww:
        cx = float(np.clip(cx, x1 - ww / 2, x0 + ww / 2))
    else:
        cx = (x0 + x1) / 2
    if y1 - y0 <= wh:
        cy = float(np.clip(cy, y1 - wh / 2, y0 + wh / 2))
    else:
        cy = (y0 + y1) / 2
    return cx, cy


def choose_framing(w: int, h: int, aspect: float, sal, faces: list[dict]) -> tuple[str, str]:
    """화면을 채울지(cover) 사진 전체를 보일지(fit-blur) 고르고 이유를 돌려준다."""
    ww, wh = cover_window(w, h, aspect)
    visible = ww * wh
    box = face_box(faces)
    if box is not None:
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw <= ww * 0.96 and bh <= wh * 0.96:
            return "cover", "얼굴이 모두 화면 안에 들어와 가득 채움"
        return "fit-blur", "화면을 채우면 얼굴이 잘려 넓게 보임"
    if visible >= 0.5:
        return "cover", "잘리는 부분이 적어 가득 채움"
    if visible < 0.3:
        return "fit-blur", "가로로 매우 긴 사진이라 넓게 보임"
    return "cover", "화면을 가득 채우고 잘리는 방향으로 훑음"


def plan_path(
    w: int,
    h: int,
    aspect: float,
    motion: str,
    framing: str,
    duration: float,
    zoom_rate: float,
    pan_rate: float,
    focus: tuple[float, float],
    sal,
    faces: list[dict],
) -> dict:
    """장면 하나의 카메라 경로. start/end 는 (cx, cy, zoom)."""
    zoom_total = float(min(MAX_ZOOM, (1 + zoom_rate) ** duration))

    if framing == "fit-blur":
        # 사진 전체가 보이는 크기를 1로 두고, 얼굴이 잘리지 않는 한도 안에서 조금 키워 빈 곳을 줄인다.
        ratio = w / h
        fit_w, fit_h = (1.0, ratio / aspect) if ratio > aspect else (aspect / ratio, 1.0)
        step = float(min(1.12, (1 + zoom_rate * 0.6) ** duration))
        box = face_box(faces)
        if box is not None:
            limit = min(0.96 * fit_w / max(1e-6, box[2] - box[0]), 0.96 * fit_h / max(1e-6, box[3] - box[1]))
            hi = float(np.clip(limit, 1.0, 1.3))
            lo = max(1.0, hi / step)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        else:
            lo, hi, cx, cy = 1.0, step, 0.5, 0.5
        a, b = (round(cx, 5), round(cy, 5), round(lo, 5)), (round(cx, 5), round(cy, 5), round(hi, 5))
        if motion == "zoom-out":
            start, end = b, a
        elif motion == "hold":
            start = end = b if box is not None else a
        else:
            start, end = a, b
        return {"framing": "fit-blur", "motion": motion if motion in ("zoom-out", "hold") else "zoom-in",
                "start": start, "end": end}

    ww, wh = cover_window(w, h, aspect)
    box = face_box(faces)
    if box is not None:
        # 얼굴이 잘릴 만큼 확대하지 않는다.
        bw, bh = max(1e-6, box[2] - box[0]), max(1e-6, box[3] - box[1])
        zoom_total = float(max(1.0, min(zoom_total, ww / bw, wh / bh)))

    def at(cx: float, cy: float, z: float) -> tuple[float, float, float]:
        cx, cy = _keep_box(cx, cy, ww / z, wh / z, box)
        cx, cy = _clamp_center(cx, cy, ww / z, wh / z)
        return round(cx, 5), round(cy, 5), round(z, 5)

    wide_x, wide_y, _ = best_center(sal, ww, wh)

    if motion in ("pan-left", "pan-right", "pan-up", "pan-down"):
        horizontal = motion in ("pan-left", "pan-right")
        z = 1.0
        slack = (1 - ww) if horizontal else (1 - wh)
        if slack < 0.04:
            z = 1.12
            slack = (1 - ww / z) if horizontal else (1 - wh / z)
        size = (ww if horizontal else wh) / z
        dist = min(slack, pan_rate * duration * size)
        if box is not None:
            free = size - ((box[2] - box[0]) if horizontal else (box[3] - box[1]))
            dist = min(dist, max(0.0, free))
        if dist >= 0.25 * pan_rate * duration * size:
            cx, cy, _ = best_center(sal, ww / z, wh / z)
            mid = cx if horizontal else cy
            lo = float(np.clip(mid - dist / 2, size / 2, 1 - size / 2 - dist))
            a, b = lo, lo + dist
            if motion in ("pan-left", "pan-up"):
                a, b = b, a
            start = at(a, cy, z) if horizontal else at(cx, a, z)
            end = at(b, cy, z) if horizontal else at(cx, b, z)
            return {"framing": "cover", "motion": motion, "start": start, "end": end}
        motion = "zoom-in"  # 움직일 여유가 없으면 확대로 바꾼다.

    if motion == "hold" or zoom_total <= 1.004:
        p = at(wide_x, wide_y, 1.0)
        return {"framing": "cover", "motion": "hold", "start": p, "end": p}

    fx, fy = focus
    tight_x = wide_x + 0.7 * (fx - wide_x)
    tight_y = wide_y + 0.7 * (fy - wide_y)
    wide, tight = at(wide_x, wide_y, 1.0), at(tight_x, tight_y, zoom_total)
    if motion == "zoom-out":
        return {"framing": "cover", "motion": "zoom-out", "start": tight, "end": wide}
    return {"framing": "cover", "motion": "zoom-in", "start": wide, "end": tight}


def interpolate(path: dict, p: float) -> tuple[float, float, float]:
    """진행도 p(0~1)에서의 (cx, cy, zoom). 확대는 지수적으로 바꿔 속도가 일정하게 느껴지게 한다."""
    (x0, y0, z0), (x1, y1, z1) = path["start"], path["end"]
    p = float(np.clip(p, 0.0, 1.0))
    z = z0 * (z1 / z0) ** p
    if abs(z1 - z0) > 1e-6:
        # 확대 중에는 화면상 이동 속도가 고르게 느껴지도록 창 크기 변화에 맞춰 중심을 옮긴다.
        q = (1 / z0 - 1 / z) / (1 / z0 - 1 / z1)
    else:
        q = p
    return x0 + (x1 - x0) * q, y0 + (y1 - y0) * q, z


def affine(path: dict, p: float, src_w: int, src_h: int, out_w: int, out_h: int) -> np.ndarray:
    """원본(src) 픽셀을 출력 화면으로 보내는 2x3 변환 행렬."""
    cx, cy, z = interpolate(path, p)
    if path["framing"] == "fit-blur":
        s = min(out_w / src_w, out_h / src_h) * z

        def offset(center: float, src: int, out: int) -> float:
            size = src * s
            if size <= out:
                return (out - size) / 2  # 화면보다 작으면 가운데에 둔다
            return float(np.clip(out / 2 - center * size, out - size, 0))

        return np.array([[s, 0, offset(cx, src_w, out_w)], [0, s, offset(cy, src_h, out_h)]], dtype=np.float64)
    ww, wh = cover_window(src_w, src_h, out_w / out_h)
    win_w, win_h = ww / z * src_w, wh / z * src_h
    sx, sy = out_w / win_w, out_h / win_h
    x0, y0 = cx * src_w - win_w / 2, cy * src_h - win_h / 2
    return np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], dtype=np.float64)


def max_scale(path: dict, src_w: int, src_h: int, out_w: int, out_h: int) -> float:
    """경로 전체에서 원본이 가장 크게 확대되는 배율."""
    return max(affine(path, p, src_w, src_h, out_w, out_h)[0, 0] for p in (0.0, 1.0))
