"""배경 음악을 직접 합성한다.

외부 음원을 쓰지 않는다. 사인파·노이즈로 만든 소리를 흔한 화음 진행에 얹은 것이라
제3자의 저작권이 걸릴 것이 없다. 박자 위치를 정확히 알기 때문에 장면 전환을 박자에 맞출 수 있다.
음색은 단순하다. 상용 음원을 대신할 품질은 아니다.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
TAU = 2 * np.pi

# 화음 진행: (근음, 화음 구성음) MIDI 번호. 한 마디(4박)에 화음 하나.
PROGRESSIONS = {
    "calm": [(48, [60, 64, 67, 71]), (45, [57, 60, 64, 67]), (41, [57, 60, 64, 65]), (43, [59, 62, 64, 67])],
    "light": [(48, [60, 64, 67]), (45, [57, 60, 64]), (41, [57, 60, 65]), (43, [59, 62, 67])],
    "upbeat": [(45, [57, 60, 64]), (41, [57, 60, 65]), (48, [60, 64, 67]), (43, [59, 62, 67])],
    "cinematic": [(45, [57, 60, 64]), (41, [53, 57, 60]), (48, [55, 60, 64]), (43, [55, 59, 62])],
    "lofi": [(50, [60, 62, 65, 69]), (43, [59, 62, 65, 69]), (48, [59, 60, 64, 67]), (45, [57, 60, 64, 67])],
}


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n: int, attack: float, release: float, decay: float | None = None) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.minimum(1.0, t / max(attack, 1e-4))
    if decay:
        e = e * np.exp(-t / decay)
    rel = min(n, int(release * SR))
    if rel > 1:
        e[-rel:] *= np.linspace(1.0, 0.0, rel)
    return e


def _tone(
    freq: float,
    dur: float,
    partials: list[tuple[float, float]],
    attack: float = 0.005,
    decay: float | None = None,
    release: float = 0.05,
    detune: float = 0.0,
) -> np.ndarray:
    n = max(2, int(dur * SR))
    t = np.arange(n) / SR
    out = np.zeros(n)
    for mult, amp in partials:
        f = freq * mult
        if f < SR * 0.45:
            out += amp * np.sin(TAU * f * t)
            if detune:
                out += amp * np.sin(TAU * f * (1 + detune) * t + 1.3)
    return out * _env(n, attack, release, decay)


SINE = [(1, 1.0)]
SOFT = [(1, 1.0), (2, 0.25), (3, 0.08)]
PLUCK = [(1, 1.0), (2, 0.4), (3, 0.15), (4, 0.08)]
MARIMBA = [(1, 1.0), (4, 0.25), (10, 0.04)]
EPIANO = [(1, 1.0), (2, 0.3), (3, 0.12), (5, 0.04)]
STRINGS = [(k, 1.0 / k**1.4) for k in range(1, 9)]
BASS = [(1, 1.0), (2, 0.35), (3, 0.12)]


def _kick(rng, punch: float = 1.0) -> np.ndarray:
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    f = 46 + 90 * punch * np.exp(-t / 0.028)
    return np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.11)


def _noise_hit(rng, dur: float, tau: float, bright: bool) -> np.ndarray:
    n = int(dur * SR)
    x = rng.standard_normal(n + 8)
    if bright:
        x = np.diff(np.diff(x))[:n] * 0.5  # 고역만 남긴다
    else:
        x = np.convolve(np.diff(x), np.ones(6) / 6, mode="same")[:n] * 1.6
    t = np.arange(n) / SR
    return x * np.exp(-t / tau) * np.minimum(1.0, t / 0.001)


class _Track:
    def __init__(self, seconds: float, seed: int) -> None:
        self.n = int(seconds * SR) + SR
        self.dry = np.zeros((self.n, 2))
        self.wet = np.zeros((self.n, 2))
        self.rng = np.random.default_rng(seed)

    def add(self, sig: np.ndarray, at: float, gain: float = 1.0, pan: float = 0.0, reverb: float = 0.0) -> None:
        i = int(round(at * SR))
        if i >= self.n or i < 0:
            return
        sig = sig[: self.n - i]
        theta = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        stereo = np.stack([sig * np.cos(theta), sig * np.sin(theta)], axis=1) * gain
        self.dry[i : i + len(sig)] += stereo
        if reverb:
            self.wet[i : i + len(sig)] += stereo * reverb

    def mix(self) -> np.ndarray:
        out = self.dry.copy()
        taps = [(0.023, 0.42), (0.037, 0.36), (0.053, 0.31), (0.071, 0.27), (0.097, 0.22), (0.131, 0.18),
                (0.167, 0.14), (0.211, 0.11), (0.263, 0.08), (0.331, 0.06), (0.419, 0.04)]
        for k, (delay, g) in enumerate(taps):
            d = int(delay * SR)
            src = self.wet[:-d] if k % 2 == 0 else self.wet[:-d, ::-1]  # 좌우를 번갈아 넓게
            out[d:] += src * g
        return out


def _build(mood: str, seconds: float, bpm: int, seed: int) -> np.ndarray:
    beat = 60.0 / bpm
    bars = int(np.ceil(seconds / (4 * beat))) + 1
    prog = PROGRESSIONS.get(mood, PROGRESSIONS["light"])
    tr = _Track(bars * 4 * beat, seed)
    rng = tr.rng

    for bar in range(bars):
        t0 = bar * 4 * beat
        root, chord = prog[bar % len(prog)]
        build_up = min(1.0, 0.55 + 0.45 * bar / max(1, bars - 2))

        if mood == "calm":
            for i, m in enumerate(chord):
                tr.add(_tone(_hz(m), 4 * beat + 0.8, SOFT, attack=0.5, release=1.0, detune=0.003), t0, 0.075,
                       pan=(i - 1.5) * 0.3, reverb=0.7)
            tr.add(_tone(_hz(root), 4 * beat + 0.4, SINE, attack=0.2, release=0.8), t0, 0.16)
            pattern = [0, 2, 1, 3, 2, 1, 3, 2]
            for k, idx in enumerate(pattern):
                m = chord[idx % len(chord)] + 12
                tr.add(_tone(_hz(m), 1.4, PLUCK, decay=0.45, release=0.2), t0 + k * beat / 2,
                       0.07 if k % 2 else 0.10, pan=0.35 if k % 2 else -0.35, reverb=0.8)

        elif mood == "light":
            for k in range(8):
                m = chord[[0, 1, 2, 1, 0, 2, 1, 2][k] % len(chord)] + 12
                tr.add(_tone(_hz(m), 0.7, MARIMBA, decay=0.22, release=0.1), t0 + k * beat / 2,
                       0.16 if k % 2 == 0 else 0.11, pan=-0.3 if k % 2 == 0 else 0.3, reverb=0.45)
            for i, m in enumerate(chord):
                tr.add(_tone(_hz(m), 4 * beat + 0.3, SOFT, attack=0.3, release=0.6, detune=0.002), t0, 0.035,
                       pan=(i - 1) * 0.4, reverb=0.5)
            for b in (0, 2):
                tr.add(_kick(rng, 0.7), t0 + b * beat, 0.42)
                tr.add(_tone(_hz(root - 12), beat * 1.6, BASS, attack=0.01, release=0.15), t0 + b * beat, 0.22)
            for k in range(8):
                if k % 2 == 1:
                    tr.add(_noise_hit(rng, 0.05, 0.014, True), t0 + k * beat / 2, 0.05, pan=0.2)

        elif mood == "upbeat":
            for b in range(4):
                tr.add(_kick(rng, 1.0), t0 + b * beat, 0.62)
                if b in (1, 3):
                    tr.add(_noise_hit(rng, 0.16, 0.05, False), t0 + b * beat, 0.30, reverb=0.25)
            for k in range(8):
                tr.add(_noise_hit(rng, 0.045, 0.012, True), t0 + k * beat / 2, 0.075 if k % 2 else 0.04, pan=0.25)
                octave = 0 if k % 2 == 0 else 12
                tr.add(_tone(_hz(root - 12 + octave), beat * 0.42, BASS, attack=0.004, release=0.05),
                       t0 + k * beat / 2, 0.25)
            for k in (1, 3, 5, 7):  # 엇박 화음
                for i, m in enumerate(chord):
                    tr.add(_tone(_hz(m + 12), 0.28, PLUCK, decay=0.12, release=0.05), t0 + k * beat / 2, 0.06,
                           pan=(i - 1) * 0.5, reverb=0.3)
            for k, idx in enumerate([0, 2, 1, 2, 0, 2, 1, 3]):
                m = chord[idx % len(chord)] + 24
                tr.add(_tone(_hz(m), 0.3, MARIMBA, decay=0.12, release=0.05), t0 + k * beat / 2, 0.07,
                       pan=-0.4, reverb=0.5)

        elif mood == "cinematic":
            for i, m in enumerate(chord):
                tr.add(_tone(_hz(m), 4 * beat + 1.0, STRINGS, attack=0.9, release=1.2, detune=0.004), t0,
                       0.045 * build_up, pan=(i - 1) * 0.5, reverb=0.8)
                tr.add(_tone(_hz(m - 12), 4 * beat + 1.0, SOFT, attack=0.9, release=1.2, detune=0.003), t0,
                       0.04 * build_up, pan=-(i - 1) * 0.4, reverb=0.6)
            tr.add(_tone(_hz(root - 12), 4 * beat + 0.6, BASS, attack=0.3, release=0.9), t0, 0.22)
            if bar % 2 == 0:
                tr.add(_kick(rng, 0.5) * 1.0, t0, 0.6, reverb=0.5)
            for k, idx in enumerate([0, 2, 1, 2]):
                m = chord[idx % len(chord)] + 12
                tr.add(_tone(_hz(m), 1.6, EPIANO, decay=0.6, release=0.3), t0 + k * beat, 0.10 * build_up,
                       pan=0.2, reverb=0.9)

        else:  # lofi
            swing = beat * 0.08
            for at in (0.0, 1.5 * beat):
                for i, m in enumerate(chord):
                    sig = _tone(_hz(m), 1.5 * beat + 0.5, EPIANO, decay=0.9, release=0.3)
                    trem = 1 + 0.12 * np.sin(TAU * 4.5 * np.arange(len(sig)) / SR)
                    tr.add(sig * trem, t0 + at, 0.075, pan=(i - 1.5) * 0.3, reverb=0.5)
            tr.add(_tone(_hz(root - 12), 2 * beat, BASS, attack=0.01, release=0.2), t0, 0.26)
            tr.add(_tone(_hz(root - 12), 1.2 * beat, BASS, attack=0.01, release=0.2), t0 + 2.5 * beat, 0.2)
            for at in (0.0, 2.5 * beat):
                tr.add(_kick(rng, 0.6), t0 + at, 0.5)
            for b in (1, 3):
                tr.add(_noise_hit(rng, 0.2, 0.06, False), t0 + b * beat, 0.2, reverb=0.3)
            for k in range(8):
                tr.add(_noise_hit(rng, 0.04, 0.010, True), t0 + k * beat / 2 + (swing if k % 2 else 0), 0.035, pan=-0.2)

    out = tr.mix()
    if mood == "lofi":  # 은은한 바닥 잡음
        out += (rng.standard_normal(out.shape) * 0.0025)
    return out


def synthesize(path: Path, seconds: float, mood: str, bpm: int, seed: int = 7) -> dict:
    """seconds 길이의 WAV를 만든다. 첫 박은 0초에 있다."""
    audio = _build(mood, seconds + 0.5, bpm, seed)
    n = int(round(seconds * SR))
    audio = audio[:n]
    fade_in, fade_out = int(0.25 * SR), int(min(2.0, seconds / 4) * SR)
    audio[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    audio[-fade_out:] *= (np.linspace(1, 0, fade_out) ** 1.5)[:, None]
    rms = float(np.sqrt((audio**2).mean()) + 1e-9)
    audio *= 10 ** (-16.5 / 20) / rms  # 평균 음량을 맞춘다
    audio = np.tanh(audio * 1.2) / 1.2  # 피크를 부드럽게 누른다
    peak = float(np.abs(audio).max())
    if peak > 0.89:
        audio *= 0.89 / peak
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return {
        "seconds": round(n / SR, 3),
        "bpm": bpm,
        "mood": mood,
        "peak": round(float(np.abs(audio).max()), 3),
        "rms_db": round(20 * np.log10(float(np.sqrt((audio**2).mean())) + 1e-9), 1),
        "source": "앱 내장 합성 (외부 음원 미사용)",
    }
