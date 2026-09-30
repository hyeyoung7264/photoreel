import subprocess
import wave

import numpy as np

from photoreel import music, planner, render
from photoreel.config import ffmpeg_exe
from photoreel.engines import ENGINES, ClipContext, EngineUnavailable, VideoFileClip
from photoreel.models import Output
from photoreel.styles import STYLES


def probe(path) -> str:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr


def small_board(project, concept, **kw):
    board = planner.build(project.photos(), project.embeddings(), concept, **kw)
    board.output = Output(aspect="custom", width=240, height=426, fps=12)
    for s in board.scenes:
        s.beats = 2
    return board.finalize()


def do_render(project, board, tmp_path, name="v.mp4"):
    photos = {p["id"]: p for p in project.photos()}
    proxies = {pid: project.proxy_path(pid) for pid in photos}
    out = tmp_path / name
    stats = render.render(board, photos, proxies, out, tmp_path / "work")
    return out, stats


def test_render_produces_playable_mp4_with_audio(project_with_photos, tmp_path):
    board = small_board(project_with_photos, "영화 같은 분위기", title="테스트 제목")
    out, stats = do_render(project_with_photos, board, tmp_path)
    info = probe(out)
    assert "Video: h264" in info and "yuv420p" in info and "240x426" in info
    assert "Audio: aac" in info
    assert abs(stats["duration"] - board.total_duration) < 0.1
    assert stats["engine"]["uses_generative_model"] is False and stats["cost_usd"] == 0.0
    assert out.read_bytes()[4:8] == b"ftyp"


def test_render_without_music_has_no_audio_stream(project_with_photos, tmp_path):
    board = small_board(project_with_photos, "음악 없이 담백하게")
    out, stats = do_render(project_with_photos, board, tmp_path)
    assert "Audio:" not in probe(out) and stats["audio"] is None


def test_every_transition_and_grade_renders(project_with_photos, tmp_path):
    board = small_board(project_with_photos, "")
    kinds = ["cut", "crossfade", "fade-black", "push-left", "push-up"]
    for s, k in zip(board.included(), kinds):
        s.transition = k
    for s, framing in zip(board.included(), ["cover", "fit-blur", "cover", "fit-blur", "cover"]):
        s.framing = framing
    board.grade = "film"
    board.style = "nostalgic"  # 입자·비네트 경로
    out, stats = do_render(project_with_photos, board.finalize(), tmp_path)
    assert stats["frames"] == round(board.total_duration * 12)


def test_blend_endpoints():
    a = np.full((8, 6, 3), 200, np.uint8)
    b = np.full((8, 6, 3), 50, np.uint8)
    for kind in ["crossfade", "push-left", "push-up"]:
        assert np.array_equal(render.blend(a, b, kind, 0.0), a)
        assert np.array_equal(render.blend(a, b, kind, 1.0), b)
    assert render.blend(a, b, "fade-black", 0.5).max() == 0


def test_grades_keep_shape_and_bw_is_gray():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)
    for g in ["neutral", "warm", "vivid", "cine", "film", "bw"]:
        out = render.apply_grade(img, g)
        assert out.shape == img.shape and out.dtype == np.uint8
    bw = render.apply_grade(img, "bw")
    assert np.array_equal(bw[..., 0], bw[..., 1]) and np.array_equal(bw[..., 1], bw[..., 2])


def test_music_length_level_and_beat(tmp_path):
    for mood, bpm in [("calm", 76), ("upbeat", 116), ("cinematic", 84), ("lofi", 88), ("light", 100)]:
        info = music.synthesize(tmp_path / f"{mood}.wav", 8.0, mood, bpm)
        with wave.open(str(tmp_path / f"{mood}.wav")) as w:
            assert w.getnchannels() == 2 and w.getframerate() == 44100
            assert abs(w.getnframes() / 44100 - 8.0) < 0.01
        assert info["peak"] <= 0.9 and -24 < info["rms_db"] < -12


def test_unavailable_engines_refuse_instead_of_faking(project_with_photos):
    ctx = ClipContext(image=np.zeros((10, 10, 3), np.uint8), analysis={}, motion="zoom-in", framing="cover",
                      seconds=2, style=STYLES["clean"], out_w=240, out_h=426, fps=12)
    for eid in ("generate", "stylize"):
        assert ENGINES[eid].available is False and ENGINES[eid].uses_generative_model is True
        try:
            ENGINES[eid].make_clip(ctx)
        except EngineUnavailable:
            continue
        raise AssertionError("연결되지 않은 엔진이 클립을 만들어서는 안 된다")


def test_video_file_clip_lets_an_external_clip_replace_a_scene(project_with_photos, tmp_path):
    """생성 엔진이 mp4를 돌려주는 경우를 가정: 로컬에서 만든 영상을 장면 클립으로 읽을 수 있어야 한다."""
    board = small_board(project_with_photos, "음악 없이")
    out, _ = do_render(project_with_photos, board, tmp_path)
    clip = VideoFileClip(out, 240, 426, 12, 1.0)
    assert clip.frame(0.0).shape == (426, 240, 3) and clip.frame(1.0).shape == (426, 240, 3)
    assert len(clip.frames) == 12
