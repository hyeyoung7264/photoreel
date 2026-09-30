"""스타일 프리셋. 장면 길이·움직임·전환·색감·음악을 한 묶음으로 정해 영상 전체의 일관성을 만든다.

여기 있는 것은 전부 '편집' 범위의 설정이다. 사진 내용을 새로 그리거나 바꾸지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MOTIONS = {
    "zoom-in": "천천히 다가가기",
    "zoom-out": "천천히 멀어지기",
    "pan-right": "오른쪽으로 훑기",
    "pan-left": "왼쪽으로 훑기",
    "pan-up": "위로 훑기",
    "pan-down": "아래로 훑기",
    "hold": "고정",
}
TRANSITIONS = {
    "cut": "바로 전환",
    "crossfade": "부드럽게 겹치기",
    "fade-black": "어두워졌다 밝아지기",
    "push-left": "옆으로 밀기",
    "push-up": "위로 밀기",
}
FRAMINGS = {
    "cover": "화면 가득 채우기 (일부 잘림)",
    "fit-blur": "넓게 보이기 (남는 곳은 흐린 배경)",
}
GRADES = {
    "neutral": "원본 색 그대로",
    "warm": "따뜻한 색감",
    "vivid": "선명한 색감",
    "cine": "차분한 영화 색감",
    "film": "바랜 필름 색감",
    "bw": "흑백",
}
MUSIC_MOODS = {
    "calm": "잔잔한",
    "upbeat": "경쾌한",
    "cinematic": "웅장한",
    "lofi": "나른한",
    "light": "가벼운",
}


@dataclass(frozen=True)
class Style:
    id: str
    name: str
    description: str
    bpm: int
    beats: int  # 장면 하나의 기본 박자 수
    min_beats: int
    max_beats: int
    transition: str
    transition_sec: float
    chapter_transition: str  # 장면 묶음이 바뀔 때
    zoom_rate: float  # 초당 확대 비율
    pan_rate: float  # 초당 이동량 (화면 폭 대비)
    grade: str
    grain: float
    vignette: float
    music_mood: str

    @property
    def beat_sec(self) -> float:
        return 60.0 / self.bpm

    def public(self) -> dict:
        d = asdict(self)
        d["summary"] = (
            f"장면당 약 {self.beats * self.beat_sec:.1f}초 · {TRANSITIONS[self.transition]} · "
            f"{GRADES[self.grade]} · {MUSIC_MOODS[self.music_mood]} 음악({self.bpm}BPM)"
        )
        return d


STYLES: dict[str, Style] = {
    s.id: s
    for s in [
        Style(
            id="clean",
            name="담백한 기록",
            description="과한 효과 없이 사진을 또렷하게 보여 준다.",
            bpm=100, beats=4, min_beats=3, max_beats=6,
            transition="crossfade", transition_sec=0.4, chapter_transition="crossfade",
            zoom_rate=0.030, pan_rate=0.055,
            grade="neutral", grain=0.0, vignette=0.0, music_mood="light",
        ),
        Style(
            id="calm",
            name="잔잔한 감성",
            description="느린 호흡과 부드러운 전환, 따뜻한 색감.",
            bpm=76, beats=4, min_beats=3, max_beats=6,
            transition="crossfade", transition_sec=0.9, chapter_transition="crossfade",
            zoom_rate=0.022, pan_rate=0.040,
            grade="warm", grain=0.0, vignette=0.12, music_mood="calm",
        ),
        Style(
            id="upbeat",
            name="경쾌한 하이라이트",
            description="박자에 맞춘 빠른 컷과 큰 움직임, 선명한 색감.",
            bpm=116, beats=4, min_beats=2, max_beats=6,
            transition="cut", transition_sec=0.0, chapter_transition="push-left",
            zoom_rate=0.055, pan_rate=0.085,
            grade="vivid", grain=0.0, vignette=0.0, music_mood="upbeat",
        ),
        Style(
            id="cinematic",
            name="영화 같은 분위기",
            description="느리게 다가가는 움직임, 어둡게 넘어가는 전환, 차분한 색감.",
            bpm=84, beats=4, min_beats=3, max_beats=6,
            transition="crossfade", transition_sec=0.7, chapter_transition="fade-black",
            zoom_rate=0.028, pan_rate=0.045,
            grade="cine", grain=0.015, vignette=0.22, music_mood="cinematic",
        ),
        Style(
            id="nostalgic",
            name="필름 같은 추억",
            description="바랜 색과 필름 입자, 여유 있는 호흡.",
            bpm=88, beats=4, min_beats=3, max_beats=6,
            transition="crossfade", transition_sec=0.6, chapter_transition="fade-black",
            zoom_rate=0.025, pan_rate=0.045,
            grade="film", grain=0.035, vignette=0.18, music_mood="lofi",
        ),
    ]
}
DEFAULT_STYLE = "clean"
