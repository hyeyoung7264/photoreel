"""영상 구성안(스토리보드) 자료 구조. 화면에서 고쳐 보낸 구성안도 이 모델로 검증한다."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .styles import FRAMINGS, GRADES, MOTIONS, MUSIC_MOODS, STYLES, TRANSITIONS

ASPECTS = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}
DEFAULT_ASPECT = "9:16"
AUTO_DURATION = (15.0, 30.0)  # 길이를 지정하지 않았을 때의 범위 (첫 검증용 기본값)


class Scene(BaseModel):
    photo_id: str
    included: bool = True
    beats: int = Field(4, ge=1, le=16)
    duration: float = 0.0
    start: float = 0.0
    motion: str = "zoom-in"
    framing: str = "cover"
    transition: str = "crossfade"
    chapter: int = 0
    reason: str = ""
    framing_reason: str = ""
    exclude_reason: str = ""

    def check(self) -> None:
        if self.motion not in MOTIONS:
            raise ValueError(f"알 수 없는 움직임: {self.motion}")
        if self.framing not in FRAMINGS:
            raise ValueError(f"알 수 없는 화면 맞춤: {self.framing}")
        if self.transition not in TRANSITIONS:
            raise ValueError(f"알 수 없는 전환: {self.transition}")


class Output(BaseModel):
    aspect: str = DEFAULT_ASPECT
    width: int = 1080
    height: int = 1920
    fps: int = Field(30, ge=12, le=60)


class Music(BaseModel):
    mode: Literal["synth", "none"] = "synth"
    mood: str = "light"
    bpm: int = Field(100, ge=50, le=180)


class Storyboard(BaseModel):
    version: int = 1
    concept: str = ""
    title: str = ""
    style: str = "clean"
    style_reason: str = ""
    grade: str = "neutral"
    output: Output = Output()
    music: Music = Music()
    engine: str = "edit"
    planner: str = "local-rules"
    order_mode: str = "content"
    order_basis: str = ""
    signals: list[dict] = []
    unsupported: list[str] = []
    notes: list[str] = []
    scenes: list[Scene] = []
    total_duration: float = 0.0
    target_duration: float | None = None

    def included(self) -> list[Scene]:
        return [s for s in self.scenes if s.included]

    def finalize(self) -> "Storyboard":
        """화면에서 고친 값을 검증하고 길이·시작 시각을 다시 계산한다."""
        if self.style not in STYLES:
            raise ValueError(f"알 수 없는 스타일: {self.style}")
        if self.grade not in GRADES:
            raise ValueError(f"알 수 없는 색감: {self.grade}")
        if self.music.mood not in MUSIC_MOODS:
            raise ValueError(f"알 수 없는 음악 분위기: {self.music.mood}")
        if self.output.aspect in ASPECTS:
            self.output.width, self.output.height = ASPECTS[self.output.aspect]
        if not (240 <= self.output.width <= 3840 and 240 <= self.output.height <= 3840):
            raise ValueError("출력 크기가 범위를 벗어남")
        beat = 60.0 / self.music.bpm
        # 제외한 장면은 뒤로 모아 둔다 (다시 넣기 쉽게).
        self.scenes = self.included() + [s for s in self.scenes if not s.included]
        t = 0.0
        for s in self.scenes:
            s.check()
            if s.included:
                s.duration = round(s.beats * beat, 4)
                s.start = round(t, 4)
                t += s.duration
            else:
                s.duration, s.start = 0.0, 0.0
        self.total_duration = round(t, 3)
        return self
