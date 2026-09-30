"""실제 브라우저로 전체 흐름을 한 번 돌려 본다: 올리기 → 콘셉트 → 구성안 고치기 → 만들기 → 재생 → 내려받기.

서버가 떠 있어야 한다 (uv run python -m photoreel).

    PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu22.04-x64 uv run python scripts/e2e_browser.py samples/jeju out/e2e
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765/"


def main() -> int:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    concept = sys.argv[3] if len(sys.argv) > 3 else "잔잔하고 따뜻한 분위기의 여행 기록"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    report: dict = {"photos": len(files), "concept": concept, "steps": []}

    def step(name: str, **info) -> None:
        report["steps"].append({"step": name, **info})
        print(f"[{name}] {json.dumps(info, ensure_ascii=False)}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 1400}, accept_downloads=True)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(URL)
        page.evaluate("localStorage.clear()")
        page.goto(URL)
        page.wait_for_selector("#examples .chip")

        t0 = time.time()
        page.set_input_files("#file", files)
        page.wait_for_function(f"document.querySelectorAll('#thumbs .thumb').length === {len(files)}", timeout=180_000)
        step("upload", seconds=round(time.time() - t0, 1), status=page.inner_text("#upload-status"))
        page.screenshot(path=str(out / "1_upload.png"), full_page=True)

        page.fill("#concept", concept)
        page.click("#btn-plan")
        page.wait_for_selector("#scenes .scene")
        scenes_before = page.locator("#scenes .scene").count()
        step("plan", scenes=scenes_before, excluded=page.locator("#excluded .excluded").count(),
             total=page.inner_text("#plan-total"))
        page.screenshot(path=str(out / "2_plan.png"), full_page=True)

        # 간단 수정: 세 번째 장면을 빼고, 두 번째 장면을 맨 앞으로 옮긴다.
        first_before = page.locator("#scenes .scene .why b").nth(0).inner_text()
        second_before = page.locator("#scenes .scene .why b").nth(1).inner_text()
        page.locator("#scenes .scene").nth(2).get_by_text("빼기").click()
        page.locator("#scenes .scene").nth(1).get_by_title("앞으로").click()
        page.wait_for_function("document.getElementById('save-state').textContent === '저장됨'")
        first_after = page.locator("#scenes .scene .why b").nth(0).inner_text()
        ok_edit = first_after == second_before and page.locator("#scenes .scene").count() == scenes_before - 1
        step("edit", ok=ok_edit, first_before=first_before, first_after=first_after,
             scenes=page.locator("#scenes .scene").count(), total=page.inner_text("#plan-total"))
        page.screenshot(path=str(out / "3_edited.png"), full_page=True)

        # 생성 방식 고르기: 사용 가능한 엔진만 선택되고, 연결 안 된 엔진은 눌러도 바뀌지 않아야 한다.
        pid = page.evaluate("location.hash.slice(1)")

        def saved_engine() -> str:
            page.wait_for_function("document.getElementById('save-state').textContent === '저장됨'")
            return page.evaluate(f"fetch('/api/projects/{pid}').then(r => r.json()).then(j => j.plan.engine)")

        def engine_card(name: str):
            return page.locator(".engine b").filter(has_text=re.compile("^" + re.escape(name)))

        engine_card("입체감 있는 움직임").click()
        picked = saved_engine()
        engine_card("사진에 움직임 생성").click()
        still = page.evaluate(f"fetch('/api/projects/{pid}').then(r => r.json()).then(j => j.plan.engine)")
        engine_card("원본 사진 편집").click()
        back = saved_engine()
        step("engine", picked=picked, after_click_unavailable=still, back=back)
        ok_engine = (picked, still, back) == ("parallax", "parallax", "edit")

        t0 = time.time()
        page.click("#btn-render")
        page.wait_for_selector("#result video", timeout=600_000)
        step("render", seconds=round(time.time() - t0, 1))
        page.wait_for_function("(() => { const v = document.querySelector('#result video'); return v && v.readyState >= 1; })()",
                               timeout=60_000)
        played = page.evaluate(
            """async () => {
                const v = document.querySelector('#result video');
                v.muted = true;
                await v.play();
                await new Promise((r) => setTimeout(r, 2500));
                const t = v.currentTime; v.pause();
                return { duration: v.duration, width: v.videoWidth, height: v.videoHeight, played: t, error: v.error && v.error.message };
            }"""
        )
        step("play", **played)
        with page.expect_download() as dl:
            page.get_by_text("MP4 내려받기").click()
        target = out / "downloaded.mp4"
        dl.value.save_as(str(target))
        step("download", filename=dl.value.suggested_filename, bytes=target.stat().st_size)
        page.screenshot(path=str(out / "4_result.png"), full_page=True)
        report["project"] = page.evaluate("location.hash.slice(1)")
        report["browser_errors"] = errors
        browser.close()

    ok = (
        ok_edit
        and ok_engine
        and played["duration"] > 5
        and played["played"] > 1
        and target.stat().st_size > 100_000
        and not errors
    )
    report["ok"] = ok
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("결과:", "성공" if ok else "실패", "| 브라우저 오류:", errors or "없음")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
