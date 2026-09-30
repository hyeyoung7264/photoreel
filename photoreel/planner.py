"""사진 분석 결과와 콘셉트로 영상 구성안을 만든다 (로컬 규칙 기반).

사진에서 확인할 수 있는 것만 근거로 삼는다: 촬영 시각, 보이는 내용의 종류, 빛 상태, 서로의 유사도.
촬영 시각이 없을 때의 순서는 '보기 좋은 배열'일 뿐 실제 일어난 순서가 아니며, 구성안에 그렇게 적는다.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from . import camera, concept as concept_mod, semantic
from .models import ASPECTS, AUTO_DURATION, DEFAULT_ASPECT, Music, Output, Scene, Storyboard
from .styles import GRADES, STYLES, Style

# 내용 종류를 크게 묶은 것. 촬영 시각이 없을 때 장면 묶음(챕터)의 기준이 된다.
GROUPS = [
    ("scenery", "풍경", {"aerial", "nature", "sea", "road", "plant"}),
    ("people", "인물", {"person", "people", "animal"}),
    ("town", "거리·장소", {"street", "market", "sign", "vehicle", "indoor", "object"}),
    ("food", "음식", {"food"}),
    ("evening", "해질녘·밤", {"sunset", "night"}),
]
GROUP_NAME = {k: name for k, name, _ in GROUPS}
OUTDOOR = {"aerial", "nature", "sea", "road", "street", "sunset", "night"}
DUP_THRESHOLD = {"clip": 0.90, "color-hist": 0.985}
CHAPTER_GAP_MIN = 45  # 촬영 시각이 이만큼 벌어지면 다른 묶음으로 본다.


def _group(photo: dict) -> str:
    tags = photo.get("tags") or {}
    scene, light = tags.get("scene"), tags.get("light")
    if light in ("dusk", "night") and (scene in OUTDOOR or scene is None):
        return "evening"
    if photo.get("face_area", 0) >= 0.01 and scene not in ("food",):
        return "people"
    for key, _, members in GROUPS:
        if scene in members:
            return key
    return "town" if scene else "scenery"


def _scene_name(photo: dict) -> str:
    scene = (photo.get("tags") or {}).get("scene")
    return semantic.SCENE_NAMES.get(scene, "사진")


def _sim(embs: dict[str, np.ndarray], a: str, b: str) -> float:
    return float(embs[a] @ embs[b])


def _find_duplicates(photos: list[dict], embs: dict[str, np.ndarray]) -> dict[str, str]:
    """거의 같은 사진 쌍에서 덜 선명한 쪽을 찾는다. {제외할 id: 남길 id}"""
    drop: dict[str, str] = {}
    for i, a in enumerate(photos):
        for b in photos[i + 1 :]:
            if a["id"] in drop or b["id"] in drop:
                continue
            method = a["methods"]["semantic"]
            if _sim(embs, a["id"], b["id"]) < DUP_THRESHOLD.get(method, 0.95):
                continue
            keep, lose = (a, b) if a["sharpness"] >= b["sharpness"] else (b, a)
            drop[lose["id"]] = keep["id"]
    return drop


def _parse_time(p: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(p["taken_at"]) if p.get("taken_at") else None
    except ValueError:
        return None


def _chain(ids: list[str], embs: dict[str, np.ndarray], start: str | None) -> list[str]:
    """비슷한 사진이 이어지도록 가까운 것부터 차례로 잇는다."""
    rest = list(ids)
    out: list[str] = []
    cur = start
    while rest:
        nxt = rest[0] if cur is None else max(rest, key=lambda i: _sim(embs, cur, i))
        rest.remove(nxt)
        out.append(nxt)
        cur = nxt
    return out


def _opener_score(p: dict) -> float:
    tags = p.get("tags") or {}
    top = dict((k, v) for k, v in tags.get("scene_top", []))
    score = 1.5 * top.get("aerial", 0) + top.get("nature", 0) + top.get("sea", 0) + 0.6 * top.get("street", 0)
    score += 0.5 if tags.get("scale") == "wide" else 0.0
    score += 0.4 * p.get("colorfulness", 0) + 0.3 * p.get("brightness", 0)
    score -= 3.0 * p.get("face_area", 0)
    return score


def _order_by_content(photos: list[dict], embs) -> tuple[list[str], dict[str, int], dict[str, str]]:
    by_id = {p["id"]: p for p in photos}
    groups: dict[str, list[str]] = {}
    for p in photos:
        groups.setdefault(_group(p), []).append(p["id"])

    day = [p for p in photos if _group(p) != "evening"]
    opener = max(day or photos, key=_opener_score)["id"]
    first_group = _group(by_id[opener])
    sequence = [first_group] + [k for k, _, _ in GROUPS if k != first_group and k != "evening"] + ["evening"]
    if first_group == "evening":
        sequence = ["evening"] + [k for k, _, _ in GROUPS if k != "evening"]

    order: list[str] = []
    chapter: dict[str, int] = {}
    reason: dict[str, str] = {}
    chapter_no = 0
    for key in sequence:
        ids = groups.get(key)
        if not ids:
            continue
        if key == "evening":
            # 해질녘 다음에 밤. 같은 빛 상태끼리는 비슷한 것끼리 잇는다.
            dusk = [i for i in ids if by_id[i]["tags"].get("light") != "night"]
            night = [i for i in ids if by_id[i]["tags"].get("light") == "night"]
            chained = _chain(dusk, embs, order[-1] if order else None) + _chain(night, embs, dusk[-1] if dusk else None)
        elif key == first_group and opener in ids:
            rest = [i for i in ids if i != opener]
            chained = [opener] + _chain(rest, embs, opener)
        else:
            chained = _chain(ids, embs, order[-1] if order else None)
        for i in chained:
            chapter[i] = chapter_no
            if key == "evening":
                reason[i] = f"{semantic.LIGHT_NAMES.get(by_id[i]['tags'].get('light'), '어두운')} 빛의 사진이라 끝부분에 배치"
            else:
                reason[i] = f"‘{GROUP_NAME[key]}’ 사진끼리 묶음 ({_scene_name(by_id[i])}(으)로 분류)"
        order += chained
        chapter_no += 1
    reason[opener] = f"{_scene_name(by_id[opener])}: 넓게 보이는 장면이라 시작에 배치"
    return order, chapter, reason


def _order_by_time(photos: list[dict]) -> tuple[list[str], dict[str, int], dict[str, str]]:
    # 촬영 시각이 없는 사진은 올린 순서상 바로 앞 사진 뒤에 붙인다.
    times: dict[str, datetime] = {}
    last: datetime | None = None
    for p in photos:
        t = _parse_time(p)
        if t is not None:
            last = t
        times[p["id"]] = t or last or datetime.min
    indexed = sorted(range(len(photos)), key=lambda i: (times[photos[i]["id"]], i))
    order = [photos[i]["id"] for i in indexed]
    by_id = {p["id"]: p for p in photos}
    chapter, reason = {}, {}
    no = 0
    prev: datetime | None = None
    for pid in order:
        t = _parse_time(by_id[pid])
        if t and prev and ((t - prev).total_seconds() > CHAPTER_GAP_MIN * 60 or t.date() != prev.date()):
            no += 1
        chapter[pid] = no
        reason[pid] = f"촬영 시각 순 ({t:%m-%d %H:%M})" if t else "촬영 시각이 없어 올린 순서상 앞 사진 뒤에 둠"
        prev = t or prev
    return order, chapter, reason


def _fit_beats(n: int, style: Style, pace: float, target: float | None, extra_beats: int) -> tuple[int, int, str]:
    """장면당 박자 수와 담을 수 있는 장면 수. (beats, max_scenes, 설명)"""
    beat = style.beat_sec
    lo, hi = (target * 0.92, target * 1.08) if target else AUTO_DURATION
    beats = int(np.clip(round(style.beats * pace), style.min_beats, style.max_beats))

    def total(b: int, count: int) -> float:
        return (b * count + extra_beats) * beat

    while beats > style.min_beats and total(beats, n) > hi:
        beats -= 1
    while beats < style.max_beats and total(beats + 1, n) <= hi and total(beats, n) < lo:
        beats += 1
    max_scenes = n
    while max_scenes > 1 and total(beats, max_scenes) > hi:
        max_scenes -= 1
    note = ""
    if total(beats, max_scenes) < lo:
        note = f"사진 수가 적어 영상이 {'목표' if target else '기본 범위(15~30초)'}보다 짧습니다."
    return beats, max_scenes, note


def _trim_to(ids: list[str], limit: int, embs, protected: set[str]) -> list[str]:
    """담을 수 있는 수를 넘으면 다른 사진과 가장 비슷한(겹치는) 것부터 뺀다."""
    keep = list(ids)
    dropped: list[str] = []
    while len(keep) > limit:
        candidates = [i for i in keep if i not in protected] or keep

        def redundancy(i: str) -> float:
            return max((_sim(embs, i, j) for j in keep if j != i), default=0.0)

        worst = max(candidates, key=redundancy)
        keep.remove(worst)
        dropped.append(worst)
    return dropped


def _assign_motion(idx: int, count: int, photo: dict, framing: str, aspect: float, state: dict) -> str:
    """장면마다 움직임을 고른다. 같은 방향이 연달아 나오지 않게 번갈아 쓴다."""
    faces = photo["faces"]
    if framing == "fit-blur":
        state["zoom"] = not state.get("zoom", False)
        return "zoom-in" if state["zoom"] else "zoom-out"
    ww, wh = camera.cover_window(photo["width"], photo["height"], aspect)
    _, _, mass = camera.best_center(photo["saliency"], ww, wh)
    can_pan_x, can_pan_y = ww < 0.8, wh < 0.8
    if idx == count - 1 and count > 1:
        return "zoom-out"  # 마지막은 물러나며 끝낸다.
    if faces:
        return "zoom-in"
    if (can_pan_x or can_pan_y) and (mass < 0.5 or state.get("last") != "pan"):
        state["last"] = "pan"
        state["dir"] = not state.get("dir", False)
        if can_pan_x:
            return "pan-right" if state["dir"] else "pan-left"
        return "pan-up" if state["dir"] else "pan-down"
    state["last"] = "zoom"
    if idx == 0:
        return "zoom-in"
    state["zoom"] = not state.get("zoom", False)
    return "zoom-in" if state["zoom"] else "zoom-out"


def build(
    photos: list[dict],
    embs: dict[str, np.ndarray],
    concept: str = "",
    *,
    title: str = "",
    aspect: str = DEFAULT_ASPECT,
    target_duration: float | None = None,
    music: bool = True,
    style_id: str | None = None,
    keep: list[tuple[str, bool]] | None = None,
) -> Storyboard:
    """구성안을 만든다. keep 을 주면 그 순서·제외 상태를 유지하고 스타일 관련 값만 다시 정한다."""
    if not photos:
        raise ValueError("사진이 없습니다.")
    reading = concept_mod.interpret(concept)
    if style_id and style_id in STYLES:
        style = STYLES[style_id]
        style_reason = "직접 고른 스타일"
    else:
        style = STYLES[reading.style]
        style_reason = (
            "콘셉트에서 분위기를 읽어 고름"
            if reading.style_matched
            else "콘셉트에서 분위기 단서를 찾지 못해 기본 스타일을 씀"
        )
    target = target_duration or reading.duration
    use_music = music and reading.music is not False
    width, height = ASPECTS.get(aspect, ASPECTS[DEFAULT_ASPECT])
    ratio = width / height
    by_id = {p["id"]: p for p in photos}
    notes: list[str] = []
    excluded: dict[str, str] = {}

    if keep is not None:
        known = [(pid, inc) for pid, inc in keep if pid in by_id]
        order = [pid for pid, inc in known if inc]
        excluded = {pid: "직접 제외함" for pid, inc in known if not inc}
        for p in photos:  # 구성안을 만든 뒤에 추가된 사진
            if p["id"] not in order and p["id"] not in excluded:
                order.append(p["id"])
        chapter = {pid: 0 for pid in order}
        reason = {pid: "직접 정한 순서" for pid in order}
        order_mode, order_basis = "manual", "직접 정한 순서를 유지했습니다."
    else:
        for lose, keep_id in _find_duplicates(photos, embs).items():
            excluded[lose] = f"{by_id[keep_id]['filename']}와(과) 거의 같은 사진이라 제외 (더 선명한 쪽을 남김)"
        candidates = [p for p in photos if p["id"] not in excluded]
        timed = sum(1 for p in candidates if p.get("taken_at"))
        span_ok = timed >= max(2, 0.8 * len(candidates))
        if reading.order == "upload":
            order = [p["id"] for p in candidates]
            chapter = {pid: 0 for pid in order}
            reason = {pid: "올린 순서 그대로" for pid in order}
            order_mode, order_basis = "upload", "요청에 따라 올린 순서를 그대로 썼습니다."
        elif span_ok:
            order, chapter, reason = _order_by_time(candidates)
            order_mode = "chronological"
            order_basis = f"사진에 기록된 촬영 시각 순으로 배열했습니다 ({timed}/{len(candidates)}장에 시각 정보 있음)."
        else:
            order, chapter, reason = _order_by_content(candidates, embs)
            order_mode = "content"
            order_basis = (
                f"촬영 시각 정보가 부족해({timed}/{len(candidates)}장) 사진에 보이는 내용과 빛 상태로 배열했습니다. "
                "실제 촬영 순서와 다를 수 있습니다."
            )
            if reading.order == "chronological":
                notes.append("시간순을 요청했지만 사진에 촬영 시각 정보가 부족해 내용 기준으로 배열했습니다.")

    extra = 3 + (1 if title else 0)  # 시작·끝 장면을 조금 더 길게
    beats, max_scenes, short_note = _fit_beats(len(order), style, reading.pace, target, extra)
    if short_note:
        notes.append(short_note)
    if keep is None and len(order) > max_scenes:
        protected = {pid for pid in order if (by_id[pid].get("tags") or {}).get("scene") in reading.emphasis}
        protected |= {order[0]}
        for pid in _trim_to(order, max_scenes, embs, protected):
            order.remove(pid)
            excluded[pid] = "길이에 맞추려고 제외 (비슷한 장면이 더 있음)"

    scenes: list[Scene] = []
    state: dict = {}
    count = len(order)
    for idx, pid in enumerate(order):
        p = by_id[pid]
        framing, framing_reason = camera.choose_framing(p["width"], p["height"], ratio, p["saliency"], p["faces"])
        motion = _assign_motion(idx, count, p, framing, ratio, state)
        n_beats = beats
        if idx == 0:
            n_beats += 1 + (1 if title else 0)
        if idx == count - 1 and count > 1:
            n_beats += 2
        if (p.get("tags") or {}).get("scene") in reading.emphasis and 0 < idx < count - 1:
            n_beats += 1
        path = camera.plan_path(
            p["width"], p["height"], ratio, motion, framing, n_beats * style.beat_sec,
            style.zoom_rate, style.pan_rate, tuple(p["focus"]), p["saliency"], p["faces"],
        )
        if idx == 0:
            transition = "cut"
        elif chapter.get(pid, 0) != chapter.get(order[idx - 1], 0):
            transition = style.chapter_transition
        else:
            transition = style.transition
        text = reason.get(pid, "")
        if idx == count - 1 and count > 1:
            text += " · 마지막 장면이라 길게 보여 주고 어둡게 마무리"
        scenes.append(
            Scene(
                photo_id=pid, beats=n_beats, motion=path["motion"], framing=framing, transition=transition,
                chapter=chapter.get(pid, 0), reason=text, framing_reason=framing_reason,
            )
        )
    for pid, why in excluded.items():
        scenes.append(Scene(photo_id=pid, included=False, beats=beats, exclude_reason=why, transition=style.transition))

    grade = reading.grade or style.grade
    if use_music:
        notes.append("음악은 앱이 직접 합성한 음원입니다 (외부 음원 미사용). 장면 전환을 박자에 맞췄습니다.")
    else:
        notes.append("음악 없이 만듭니다.")
    methods = {p["methods"]["semantic"] for p in photos}
    if "color-hist" in methods:
        notes.append("사진 내용 분류 모델을 쓰지 못해 색 분포만으로 비교했습니다. 순서와 중복 판단이 부정확할 수 있습니다.")

    board = Storyboard(
        concept=concept,
        title=title.strip(),
        style=style.id,
        style_reason=style_reason,
        grade=grade if grade in GRADES else style.grade,
        output=Output(aspect=aspect if aspect in ASPECTS else DEFAULT_ASPECT, width=width, height=height),
        music=Music(mode="synth" if use_music else "none", mood=style.music_mood, bpm=style.bpm),
        order_mode=order_mode,
        order_basis=order_basis,
        signals=reading.signals,
        unsupported=reading.unsupported,
        notes=notes,
        scenes=scenes,
        target_duration=target,
    )
    return board.finalize()
