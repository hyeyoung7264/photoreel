# photoreel

여러 장의 사진과 콘셉트 문장으로 하나의 짧은 영상을 만드는 **검증용 프로토타입**입니다.
여행 전용이 아닙니다. 일상·가족·반려동물·행사 사진에도 같은 흐름을 씁니다.

사진 올리기 → 콘셉트 입력 → 구성안 확인·수정 → 만들기 → 재생·MP4 내려받기

모든 처리는 이 컴퓨터 안에서 이뤄집니다. 사진을 외부 서비스로 보내지 않고, 유료 API를 호출하지 않습니다.

## 실행

필요한 것: [uv](https://docs.astral.sh/uv/) (Python 3.12는 uv가 알아서 준비합니다). ffmpeg는 따로 설치하지 않아도 됩니다 (`imageio-ffmpeg`에 들어 있음).

```bash
cd ~/photoreel && uv sync && uv run python -m photoreel
```

브라우저에서 <http://localhost:8765> 를 엽니다 (WSL에서 실행해도 Windows 브라우저로 열 수 있습니다).
처음 켤 때 모델·글꼴(약 700MB)을 `assets_cache/`에 내려받습니다. 받지 못한 항목은 단순한 방식으로 대체되고, 화면과 결과에 그 사실이 표시됩니다.

화면 없이 폴더째 만들 수도 있습니다.

```bash
uv run python -m photoreel.cli ~/내사진폴더 --concept "잔잔하고 따뜻한 분위기의 여행 기록" --out out/mine.mp4
```

옵션: `--title "제목"`, `--aspect 9:16|4:5|1:1|16:9`, `--seconds 20`, `--style clean|calm|upbeat|cinematic|nostalgic`, `--engine edit|parallax`, `--no-music`

테스트용 공개 라이선스 사진 받기: `uv run python scripts/fetch_sample_photos.py` (출처는 `samples/*/CREDITS.md`)

## 영상이 만들어지는 방식 (정확히)

| 단계 | 실제로 하는 일 | AI 모델 |
| --- | --- | --- |
| 사진 분석 | EXIF 촬영 시각·방향, 선명도·밝기·색, 얼굴 위치, 시선이 갈 영역, 내용 분류(풍경/바다/음식/인물 등)와 사진 간 유사도 | 얼굴 검출 YuNet, 내용 분류 CLIP ViT-B/32 (분류용, 생성 아님) |
| 구성안 | 촬영 시각이 있으면 시간순, 없으면 내용 묶음 + 빛 상태(낮→해질녘→밤)로 배열. 거의 같은 사진 제외, 길이에 맞춰 박자 단위로 장면 길이 결정, 장면별 움직임·화면 맞춤·전환 지정 | 없음 (규칙 기반) |
| 콘셉트 해석 | 문장에서 분위기·속도·순서·길이·흑백·음악 여부·강조 대상 단어를 찾아 설정으로 바꿈. 생성 모델이 필요한 요청(애니메이션풍, 움직이게 등)은 "반영하지 못함"으로 표시 | 없음 (단어 규칙) |
| 영상 (기본: 원본 사진 편집) | 원본을 자르고 천천히 확대·이동, 전환(겹치기/어두워졌다 밝아지기/밀기/컷), 색보정, 제목, 페이드 | **없음. 장면을 새로 그리지 않음** |
| 영상 (선택: 입체감 있는 움직임, 실험) | 위 편집에 더해 깊이 추정으로 가까운 것과 먼 것을 다른 속도로 이동(시차) | 깊이 추정 Depth Anything V2 Small (생성 아님) |
| 음악 | 앱이 사인파·노이즈로 직접 합성한 음원. 외부 음원 없음. 장면 경계를 박자에 맞춤 | 없음 |

"사진 속 장면에 움직임 생성(image-to-video)"과 "그림체 변환"은 엔진 자리만 있고 **연결되지 않았습니다**. 외부 서비스의 계정·유료 호출 승인과 사진 외부 전송 동의가 필요하기 때문입니다. 화면에서도 "아직 연결 안 됨"으로 표시되며, 선택해도 실행을 거부합니다.

## 구조

```
photoreel/
  analyze.py   사진 분석           semantic.py  CLIP 분류·임베딩       depth.py  깊이 추정
  concept.py   콘셉트 문장 해석     planner.py   구성안(순서·길이·연출)  camera.py 자르기·확대·이동 경로
  engines.py   장면 클립 엔진(편집 / 시차 / 생성·변환 자리)             render.py 프레임 합성 → ffmpeg
  music.py     음악 합성           store.py     프로젝트 파일 저장       server.py 웹 API
  web/index.html  화면              cli.py       폴더 → 영상
scripts/  fetch_sample_photos.py, evaluate.py(수치 점검), contact_sheet.py(장면별 프레임), e2e_browser.py(브라우저 전체 흐름)
tests/    pytest (합성 이미지로 실행, 네트워크 불필요)
```

새 엔진은 `engines.py`에 `make_clip`만 구현하면 됩니다. 렌더러는 장면마다 `frame(p)`를 주는 클립만 요구하므로, 외부 생성 서비스가 돌려준 mp4는 `VideoFileClip`으로 감싸 그대로 끼울 수 있습니다 (테스트로 확인).

## 테스트

```bash
uv run pytest -q
```

브라우저 전체 흐름 (서버를 켠 상태에서). Ubuntu 20.04에서는 Playwright가 기본 브라우저를 지원하지 않아 플랫폼을 지정해 설치합니다.

```bash
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu22.04-x64 uv run playwright install chromium-headless-shell
```

```bash
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu22.04-x64 uv run python scripts/e2e_browser.py samples/jeju out/e2e
```

결과 점검과 평가 기록은 [docs/EVALUATION.md](docs/EVALUATION.md)에 있습니다.
