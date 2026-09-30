from datetime import datetime, timedelta

import numpy as np

from photoreel import planner
from photoreel.models import AUTO_DURATION

FLAT = np.full((24, 24), 1 / 576).tolist()


def photo(pid, *, scene="nature", light="day", taken=None, size=(3200, 2133), faces=None, sharp=500.0, scale="wide"):
    faces = faces or []
    return {
        "id": pid, "filename": f"{pid}.jpg", "width": size[0], "height": size[1], "taken_at": taken,
        "sharpness": sharp, "brightness": 0.5, "warmth": 0.0, "colorfulness": 0.3, "faces": faces,
        "face_area": sum(f["w"] * f["h"] for f in faces), "focus": [0.5, 0.5], "saliency": FLAT,
        "tags": {"scene": scene, "scene_conf": 0.9, "scene_top": [[scene, 0.9]], "light": light, "scale": scale},
        "methods": {"faces": "yunet", "semantic": "clip"},
    }


def embs(photos, seed=0, near=None):
    rng = np.random.default_rng(seed)
    out = {}
    for p in photos:
        v = rng.standard_normal(32)
        out[p["id"]] = (v / np.linalg.norm(v)).astype(np.float32)
    for a, b in near or []:
        v = out[a] + 0.05 * rng.standard_normal(32)
        out[b] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def order(board):
    return [s.photo_id for s in board.scenes if s.included]


def test_chronological_when_photos_have_timestamps():
    t0 = datetime(2026, 5, 1, 9)
    photos = [photo(f"p{i}", taken=(t0 + timedelta(minutes=10 * (5 - i))).isoformat()) for i in range(6)]
    board = planner.build(photos, embs(photos), "담백한 기록")
    assert board.order_mode == "chronological"
    assert order(board) == ["p5", "p4", "p3", "p2", "p1", "p0"]
    assert "촬영 시각" in board.order_basis


def test_content_order_without_timestamps_is_labelled_as_not_factual():
    photos = [
        photo("night", scene="night", light="night"),
        photo("food", scene="food", scale="close"),
        photo("dusk", scene="sunset", light="dusk"),
        photo("aerial", scene="aerial"),
        photo("market", scene="market"),
    ]
    board = planner.build(photos, embs(photos), "")
    got = order(board)
    assert board.order_mode == "content"
    assert got[0] == "aerial"  # 넓은 전경으로 시작
    assert got[-2:] == ["dusk", "night"]  # 해질녘 → 밤으로 끝
    assert "실제 촬영 순서와 다를 수 있습니다" in board.order_basis


def test_near_duplicates_are_excluded_with_a_reason_and_can_be_seen():
    photos = [photo("a", sharp=900), photo("b", sharp=300), photo("c"), photo("d")]
    board = planner.build(photos, embs(photos, near=[("a", "b")]), "")
    excluded = [s for s in board.scenes if not s.included]
    assert [s.photo_id for s in excluded] == ["b"]
    assert "거의 같은 사진" in excluded[0].exclude_reason and "a.jpg" in excluded[0].exclude_reason


def test_auto_duration_stays_in_default_range_and_trims_redundant_photos():
    photos = [photo(f"p{i}") for i in range(30)]
    board = planner.build(photos, embs(photos), "신나는 하이라이트")
    assert AUTO_DURATION[0] <= board.total_duration <= AUTO_DURATION[1] + 0.01
    assert any("길이에 맞추려고" in s.exclude_reason for s in board.scenes if not s.included)


def test_requested_duration_is_followed_not_a_fixed_format():
    photos = [photo(f"p{i}") for i in range(12)]
    board = planner.build(photos, embs(photos), "잔잔하게 50초")
    assert 44 <= board.total_duration <= 56
    wide = planner.build(photos, embs(photos), "", aspect="16:9")
    assert (wide.output.width, wide.output.height) == (1920, 1080)


def test_few_photos_gives_short_video_with_a_note():
    photos = [photo("a"), photo("b")]
    board = planner.build(photos, embs(photos), "")
    assert board.total_duration < AUTO_DURATION[0]
    assert any("짧습니다" in n for n in board.notes)


def test_scene_starts_fall_on_beats():
    photos = [photo(f"p{i}") for i in range(8)]
    board = planner.build(photos, embs(photos), "경쾌하게")
    beat = 60 / board.music.bpm
    for s in board.included():
        assert abs(s.start / beat - round(s.start / beat)) < 1e-3
    assert board.included()[0].transition == "cut"


def test_keep_preserves_user_order_and_exclusions_when_style_changes():
    photos = [photo(f"p{i}") for i in range(5)]
    e = embs(photos)
    keep = [("p3", True), ("p0", True), ("p4", False), ("p1", True), ("p2", False, "p0.jpg와(과) 거의 같은 사진")]
    board = planner.build(photos, e, "잔잔하게", style_id="cinematic", keep=keep)
    assert board.style == "cinematic" and board.order_mode == "manual"
    assert order(board) == ["p3", "p0", "p1"]
    excluded = {s.photo_id: s.exclude_reason for s in board.scenes if not s.included}
    assert excluded == {"p4": "직접 제외함", "p2": "p0.jpg와(과) 거의 같은 사진"}  # 자동 제외 이유가 사라지지 않음


def test_no_music_and_unsupported_requests_surface_in_plan():
    photos = [photo(f"p{i}") for i in range(4)]
    board = planner.build(photos, embs(photos), "음악 없이 픽사 애니메이션처럼")
    assert board.music.mode == "none"
    assert board.unsupported and board.engine == "edit"


def test_faces_are_never_cropped_by_default_framing():
    faces = [{"x": 0.05 + 0.18 * i, "y": 0.4, "w": 0.08, "h": 0.12} for i in range(5)]
    photos = [photo("group", scene="people", faces=faces), photo("b"), photo("c")]
    board = planner.build(photos, embs(photos), "")
    assert next(s for s in board.scenes if s.photo_id == "group").framing == "fit-blur"


def test_slower_pace_request_actually_lengthens_scenes():
    photos = [photo(f"p{i}") for i in range(11)]
    e = embs(photos)
    normal = planner.build(photos, e, "필름 같은 추억")
    slow = planner.build(photos, e, "필름 같은 추억, 천천히")
    assert slow.included()[1].duration > normal.included()[1].duration
    assert len(slow.included()) == len(normal.included())


def test_chapter_transitions_are_not_back_to_back():
    photos = [photo(f"p{i}", scene=sc) for i, sc in enumerate(
        ["aerial", "sea", "nature", "market", "food", "person", "street", "sign", "plant"])]
    board = planner.build(photos, embs(photos), "영화 같은 분위기")
    kinds = [s.transition for s in board.included()]
    assert "fade-black" in kinds
    assert all(not (a == b == "fade-black") for a, b in zip(kinds, kinds[1:]))
