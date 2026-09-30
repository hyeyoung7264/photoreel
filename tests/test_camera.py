import numpy as np
import pytest

from photoreel import camera

FLAT = np.full((24, 24), 1 / 576).tolist()
MOTIONS = ["zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down", "hold"]
SIZES = [(3200, 2133), (2400, 3200), (1080, 1920), (4000, 1000), (1000, 1000)]
ASPECTS = [9 / 16, 1.0, 16 / 9]


def window(path, p, w, h, aspect):
    cx, cy, z = camera.interpolate(path, p)
    ww, wh = camera.cover_window(w, h, aspect)
    return cx - ww / z / 2, cy - wh / z / 2, cx + ww / z / 2, cy + wh / z / 2


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("aspect", ASPECTS)
@pytest.mark.parametrize("motion", MOTIONS)
def test_cover_window_never_leaves_the_photo(size, aspect, motion):
    w, h = size
    path = camera.plan_path(w, h, aspect, motion, "cover", 3.0, 0.05, 0.08, (0.8, 0.2), FLAT, [])
    for p in np.linspace(0, 1, 11):
        x0, y0, x1, y1 = window(path, p, w, h, aspect)
        assert x0 >= -1e-4 and y0 >= -1e-4 and x1 <= 1 + 1e-4 and y1 <= 1 + 1e-4


@pytest.mark.parametrize("motion", MOTIONS)
def test_faces_stay_inside_the_frame(motion):
    w, h, aspect = 2400, 3200, 9 / 16
    faces = [{"x": 0.62, "y": 0.30, "w": 0.12, "h": 0.12}, {"x": 0.40, "y": 0.34, "w": 0.10, "h": 0.10}]
    assert camera.choose_framing(w, h, aspect, FLAT, faces)[0] == "cover"
    path = camera.plan_path(w, h, aspect, motion, "cover", 4.0, 0.06, 0.09, (0.55, 0.36), FLAT, faces)
    bx0, by0, bx1, by1 = camera.face_box(faces)
    for p in np.linspace(0, 1, 11):
        x0, y0, x1, y1 = window(path, p, w, h, aspect)
        assert x0 <= bx0 + 1e-4 and y0 <= by0 + 1e-4 and x1 >= bx1 - 1e-4 and y1 >= by1 - 1e-4


def test_wide_group_photo_is_not_cropped_through_faces():
    faces = [{"x": 0.05 + 0.18 * i, "y": 0.4, "w": 0.08, "h": 0.12} for i in range(5)]
    framing, _ = camera.choose_framing(3200, 2133, 9 / 16, FLAT, faces)
    assert framing == "fit-blur"
    path = camera.plan_path(3200, 2133, 9 / 16, "zoom-in", framing, 3.0, 0.03, 0.05, (0.5, 0.45), FLAT, faces)
    bx0, _, bx1, _ = camera.face_box(faces)
    for p in (0.0, 0.5, 1.0):
        m = camera.affine(path, p, 3200, 2133, 1080, 1920)
        assert m[0, 0] * bx0 * 3200 + m[0, 2] >= -1 and m[0, 0] * bx1 * 3200 + m[0, 2] <= 1081


def test_pan_without_room_becomes_zoom():
    path = camera.plan_path(1080, 1920, 9 / 16, "pan-up", "cover", 3.0, 0.03, 0.0001, (0.5, 0.5), FLAT, [])
    assert path["motion"] in ("zoom-in", "pan-up")
    x0, y0, x1, y1 = window(path, 1.0, 1080, 1920, 9 / 16)
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def test_affine_maps_window_to_output():
    path = camera.plan_path(3200, 2133, 9 / 16, "zoom-in", "cover", 3.0, 0.03, 0.05, (0.5, 0.5), FLAT, [])
    x0, y0, x1, y1 = window(path, 0.3, 3200, 2133, 9 / 16)
    m = camera.affine(path, 0.3, 3200, 2133, 1080, 1920)
    assert np.allclose(m @ [x0 * 3200, y0 * 2133, 1], [0, 0], atol=0.05)
    assert np.allclose(m @ [x1 * 3200, y1 * 2133, 1], [1080, 1920], atol=0.05)


def test_best_center_finds_the_salient_side():
    sal = np.zeros((24, 24))
    sal[10:14, 18:23] = 1
    cx, cy, mass = camera.best_center(sal, 0.375, 1.0)
    assert cx > 0.7 and mass > 0.9
