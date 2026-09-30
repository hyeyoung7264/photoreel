"""자유 문장으로 쓴 콘셉트를 편집 설정으로 읽는다.

언어 모델이 아니라 단어 규칙이다. 그래서 무엇을 읽어 냈고 무엇을 못 읽었는지를
그대로 돌려주어 화면에 보여 준다. 읽지 못한 요청을 반영한 척하지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .styles import DEFAULT_STYLE, STYLES

STYLE_WORDS = {
    "calm": [
        "잔잔", "감성", "차분", "힐링", "편안", "따뜻", "포근", "여유", "조용", "평화", "고요", "몽글", "아련",
        "서정", "나른", "calm", "soft", "gentle", "peaceful", "cozy", "warm", "relax",
    ],
    "upbeat": [
        "신나", "경쾌", "활기", "발랄", "에너지", "역동", "빠른", "빠르게", "하이라이트", "브이로그", "vlog",
        "흥겨", "즐거", "유쾌", "힙", "트렌디", "릴스", "쇼츠", "신명", "통통", "upbeat", "energetic", "fun",
        "fast", "dynamic", "highlight",
    ],
    "cinematic": [
        "영화", "시네마", "웅장", "다큐", "드라마틱", "장엄", "진지", "묵직", "고급", "세련", "cinematic", "epic",
        "dramatic", "movie",
    ],
    "nostalgic": [
        "추억", "레트로", "필름", "빈티지", "옛날", "아날로그", "그리운", "그리움", "회상", "향수", "nostalg",
        "retro", "vintage", "memory", "memories",
    ],
    "clean": [
        "깔끔", "담백", "심플", "미니멀", "기록", "정리", "단순", "있는 그대로", "clean", "simple", "minimal",
    ],
}
PACE_FAST = ["빠르게", "빠른", "짧게", "속도감", "템포 빠", "fast", "quick"]
PACE_SLOW = ["느리게", "느린", "천천히", "여유롭게", "길게", "slow"]
ORDER_TIME = ["시간순", "시간 순", "순서대로", "찍은 순서", "촬영 순", "날짜순", "날짜 순", "chronolog"]
ORDER_UPLOAD = ["올린 순서", "업로드 순", "upload order"]
GRADE_BW = ["흑백", "모노톤", "black and white", "monochrome"]
MUSIC_OFF = ["음악 없이", "음악없이", "무음", "음악 빼", "음악은 빼", "no music", "without music"]
EMPHASIS = {
    "인물": (["인물", "사람", "얼굴", "표정", "people", "portrait"], ["person", "people"]),
    "풍경": (["풍경", "경치", "자연", "landscape", "scenery"], ["nature", "sea", "aerial", "sunset"]),
    "음식": (["음식", "먹방", "맛집", "먹은", "food"], ["food"]),
    "동물": (["반려", "강아지", "고양이", "댕댕", "냥이", "pet", "dog", "cat"], ["animal"]),
}
# 생성 모델이 있어야만 가능한 요청. 현재 엔진에서는 적용되지 않는다는 것을 알려 준다.
NEEDS_GENERATION = [
    (
        ["애니메이션", "애니", "만화", "지브리", "픽사", "웹툰", "일러스트", "그림체", "수채화", "유화", "캐릭터",
         "anime", "cartoon", "pixar", "ghibli"],
        "그림체를 바꾸는 스타일 변환",
    ),
    (
        ["움직이게", "살아 움직", "살아움직", "움직이는 영상", "걷는", "파도가 치", "바람에 흔들", "립싱크",
         "말하는", "춤추"],
        "사진 속 장면에 실제 움직임을 만들어 넣는 생성",
    ),
    (["내레이션", "나레이션", "목소리", "자막", "voiceover", "narration"], "내레이션·자막 만들기"),
]


@dataclass
class Reading:
    style: str = DEFAULT_STYLE
    style_matched: bool = False
    pace: float = 1.0
    order: str | None = None
    grade: str | None = None
    music: bool | None = None
    duration: float | None = None
    emphasis: list[str] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def note(self, found: str, meaning: str) -> None:
        self.signals.append({"found": found, "meaning": meaning})


def _first(text: str, words: list[str]) -> str | None:
    hits = [(text.find(w), w) for w in words if w in text]
    return min(hits)[1] if hits else None


def interpret(concept: str) -> Reading:
    text = (concept or "").lower().strip()
    r = Reading()

    best: tuple[int, int, str, list[str]] | None = None
    for style_id, words in STYLE_WORDS.items():
        found = [w for w in words if w in text]
        if not found:
            continue
        pos = min(text.find(w) for w in found)
        key = (-len(found), pos, style_id, found)
        if best is None or key < best:
            best = key
    if best:
        r.style, r.style_matched = best[2], True
        r.note(", ".join(best[3]), f"스타일 → {STYLES[r.style].name}")

    if w := _first(text, MUSIC_OFF):
        r.music = False
        r.note(w, "음악 없이 만들기")
    if w := _first(text, GRADE_BW):
        r.grade = "bw"
        r.note(w, "색감 → 흑백")
    if w := _first(text, ORDER_TIME):
        r.order = "chronological"
        r.note(w, "순서 → 촬영 시각 순")
    elif w := _first(text, ORDER_UPLOAD):
        r.order = "upload"
        r.note(w, "순서 → 올린 순서 그대로")
    if w := _first(text, PACE_SLOW):
        r.pace = 1.35
        r.note(w, "호흡 → 장면을 더 길게")
    elif (w := _first(text, PACE_FAST)) and r.style != "upbeat":
        r.pace = 0.75
        r.note(w, "호흡 → 장면을 더 짧게")

    m = re.search(r"(\d+)\s*분(?:\s*(\d+)\s*초)?", text)
    if m:
        r.duration = int(m.group(1)) * 60 + int(m.group(2) or 0)
    elif m := re.search(r"(\d+)\s*(?:초|sec|seconds?)", text):
        r.duration = int(m.group(1))
    if r.duration is not None:
        r.duration = float(min(120, max(5, r.duration)))
        r.note(m.group(0), f"길이 → 약 {r.duration:.0f}초")

    for label, (words, keys) in EMPHASIS.items():
        if w := _first(text, words):
            r.emphasis.extend(k for k in keys if k not in r.emphasis)
            r.note(w, f"강조 → {label} 사진을 우선하고 조금 더 길게")

    for words, what in NEEDS_GENERATION:
        if w := _first(text, words):
            r.unsupported.append(f"‘{w}’: {what}은(는) 생성 모델이 필요해 현재 엔진에서는 적용되지 않았습니다.")
    return r
