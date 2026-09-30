"""완성된 영상에서 장면별 프레임을 뽑아 한 장으로 붙인다 (눈으로 확인하는 용도).

    uv run python scripts/contact_sheet.py data/projects/<id>/renders/<n> [out.jpg] [--points mid|edges|trans]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photoreel.config import ffmpeg_exe  # noqa: E402


def grab(video: Path, t: float, w: int, h: int) -> Image.Image:
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return Image.fromarray(np.frombuffer(raw, np.uint8).reshape(h, w, 3))


def main() -> None:
    rdir = Path(sys.argv[1])
    mode = sys.argv[sys.argv.index("--points") + 1] if "--points" in sys.argv else "mid"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else rdir / f"sheet_{mode}.jpg"
    board = json.loads((rdir / "storyboard.json").read_text(encoding="utf-8"))
    scenes = [s for s in board["scenes"] if s["included"]]
    ow, oh = board["output"]["width"], board["output"]["height"]
    th = 480
    tw = round(th * ow / oh / 2) * 2
    times: list[tuple[float, str]] = []
    for i, s in enumerate(scenes):
        a, b = s["start"], s["start"] + s["duration"]
        if mode == "mid":
            times.append(((a + b) / 2, f"{i + 1} {s['motion']}"))
        elif mode == "edges":
            times += [(a + 0.55, f"{i + 1} start"), (b - 0.55, f"{i + 1} end")]
        else:
            if i > 0:
                times.append((a, f"->{i + 1} {s['transition']}"))
    cols = min(len(times), 8 if tw < 400 else 4)
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 6), rows * (th + 22)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (t, label) in enumerate(times):
        x, y = (k % cols) * (tw + 6), (k // cols) * (th + 22)
        sheet.paste(grab(rdir / "video.mp4", t, tw, th), (x, y + 18))
        d.text((x + 2, y + 3), f"{t:.1f}s {label}", fill="black")
    sheet.save(out, quality=88)
    print(out)


if __name__ == "__main__":
    main()
