"""공개 라이선스 테스트 사진 세트를 내려받는다 (Wikimedia Commons).

사용자 본인의 사진이 아니다. 파이프라인을 실제 사진으로 돌려 보기 위한 대체 세트이며,
저작자·라이선스는 samples/jeju/CREDITS.md 에 기록된다. 사진 파일은 저장소에 커밋하지 않는다.

    uv run python scripts/fetch_sample_photos.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "photoreel-prototype/0.1 (local prototype test set download)"}
API = "https://commons.wikimedia.org/w/api.php"
OUT = Path(__file__).resolve().parent.parent / "samples" / "jeju"

# 파일명 앞 번호는 일부러 내용과 무관하게 섞었다. 업로드(파일명) 순서가 곧 좋은 순서가
# 되지 않도록 해서, 구성안이 순서를 실제로 다시 짜는지 확인하기 위함이다.
PHOTOS = [
    ("IMG_2041", "File:Jeju dongmun market 4.JPG"),
    ("IMG_2107", "File:Jeju Island sunset1.jpg"),
    ("IMG_2113", "File:Seongsan Ilchulbong 03.jpg"),
    ("IMG_2188", "File:Korean BBQ.jpg"),
    ("IMG_2203", "File:Seongsan Ilchulbong from the air.jpg"),
    ("IMG_2240", "File:Jeju Olle trail markers.jpg"),
    ("IMG_2296", "File:Haenyo 8101.jpg"),
    ("IMG_2311", "File:Jeju Island sunset2.jpg"),
    ("IMG_2352", "File:Hyeopjae Beach Scenery.jpg"),
    ("IMG_2377", "File:Jeju Dongmun Traditional Market 01.jpg"),
    ("IMG_2410", "File:Seongsan-ri town at blue hour seen from Seongsan Ilchulbong volcano Jeju Island South Korea.jpg"),
    ("IMG_2466", "File:Jeju Olle Route 14.jpg"),
]


def fetch(url: str, timeout: int = 120) -> bytes:
    """429(요청 과다)를 포함한 일시 오류는 간격을 늘려 가며 다시 시도한다."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            wait = 10 * (attempt + 1)
            print(f"  재시도 {attempt + 1} ({wait}초 뒤): {e}")
            time.sleep(wait)
    raise AssertionError("unreachable")


def api(params: dict) -> dict:
    return json.loads(fetch(API + "?" + urllib.parse.urlencode(params), timeout=60))


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    credits = [
        "# 테스트 사진 출처",
        "",
        "Wikimedia Commons에서 받은 공개 라이선스 사진이다. 사용자 본인의 사진이 아니며,",
        "프로토타입의 로컬 테스트에만 사용한다. 재배포할 때는 아래 라이선스 조건을 따라야 한다.",
        "",
        "| 파일 | 원본 | 저작자 | 라이선스 |",
        "| --- | --- | --- | --- |",
    ]
    failed = 0
    for name, title in PHOTOS:
        data = api(
            {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": 3000,
                "iiextmetadatafilter": "LicenseShortName|Artist",
            }
        )
        page = next(iter(data["query"]["pages"].values()))
        if "imageinfo" not in page:
            print(f"실패: {title} (찾을 수 없음)")
            failed += 1
            continue
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        dest = OUT / f"{name}.jpg"
        if not dest.exists():
            try:
                dest.write_bytes(fetch(info.get("thumburl") or info["url"]))
            except Exception as e:  # noqa: BLE001
                print(f"실패: {title}: {e}")
                failed += 1
                continue
            time.sleep(2)
        artist = strip_html(meta.get("Artist", {}).get("value", "?"))
        lic = strip_html(meta.get("LicenseShortName", {}).get("value", "?"))
        credits.append(f"| {dest.name} | [{title[5:]}]({info['descriptionurl']}) | {artist} | {lic} |")
        print(f"{dest.name}  <-  {title}  [{lic}]")
    (OUT / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(f"\n{len(PHOTOS) - failed}/{len(PHOTOS)}장 저장: {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
