"""로컬 CLIP 모델로 사진 내용을 분류하고 임베딩을 만든다.

이 모델은 사진을 '이해'(분류·유사도 계산)하는 데만 쓴다. 영상이나 이미지를 생성하지 않으며,
사진은 이 컴퓨터 밖으로 나가지 않는다. 모델 파일이 없으면 색 분포 기반 특징으로 대체된다.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

from . import assets

log = logging.getLogger(__name__)

# (내부 키, 화면 표시 이름, CLIP 프롬프트). 표시 이름은 '보이는 것'만 말한다.
# '행사', '가족'처럼 사진만으로는 알 수 없는 사건·관계는 일부러 넣지 않았다.
SCENE_LABELS = [
    ("aerial", "넓은 전경", "an aerial view photo of a landscape from above"),
    ("nature", "자연 풍경", "a landscape photo of mountains, fields or nature scenery"),
    ("sea", "바다·해변", "a photo of a beach or the sea"),
    ("street", "거리·건물", "a photo of a city street or buildings"),
    ("market", "시장·상점", "a photo of a market, shop or stall"),
    ("food", "음식", "a close-up photo of food or drink"),
    ("person", "인물", "a portrait photo of one person"),
    ("people", "여러 사람", "a photo of a group of people"),
    ("animal", "동물", "a photo of a pet animal, a dog or a cat"),
    ("indoor", "실내", "an indoor photo of a room"),
    ("sign", "표지·글자", "a photo of a sign, signpost or text"),
    ("plant", "꽃·식물", "a close-up photo of flowers or plants"),
    ("road", "길", "a photo of a road, path or trail"),
    ("vehicle", "탈것", "a photo of a car, train, airplane or boat"),
    ("sunset", "노을", "a photo of a sunset or sunrise sky"),
    ("night", "야경", "a night photo with lights"),
    ("object", "사물", "a close-up photo of an object"),
]
LIGHT_LABELS = [
    ("day", "낮", "a photo taken in bright daytime"),
    ("dusk", "해질녘", "a photo taken at sunset or dusk"),
    ("night", "밤", "a photo taken at night"),
]
SCALE_LABELS = [
    ("wide", "넓은 장면", "a wide shot showing a whole scene"),
    ("medium", "중간 거리", "a medium shot of a subject"),
    ("close", "가까운 장면", "a close-up shot of a detail"),
]

SCENE_NAMES = {k: name for k, name, _ in SCENE_LABELS}
LIGHT_NAMES = {k: name for k, name, _ in LIGHT_LABELS}
SCALE_NAMES = {k: name for k, name, _ in SCALE_LABELS}

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class Clip:
    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        vision = assets.ensure("clip_vision")
        text = assets.ensure("clip_text")
        tok = assets.ensure("clip_tokenizer")
        if not (vision and text and tok):
            raise RuntimeError("CLIP 모델 파일을 준비하지 못함")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.vision = ort.InferenceSession(str(vision), opts, providers=providers)
        self.text = ort.InferenceSession(str(text), opts, providers=providers)
        self.tokenizer = Tokenizer.from_file(str(tok))
        self.tokenizer.no_padding()
        self._vision_out = self._pick_output(self.vision, "image_embeds")
        self._text_out = self._pick_output(self.text, "text_embeds")
        self._text_inputs = [i.name for i in self.text.get_inputs()]
        self._label_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _pick_output(session, preferred: str) -> str:
        names = [o.name for o in session.get_outputs()]
        return preferred if preferred in names else names[0]

    def embed_image(self, img: Image.Image) -> np.ndarray:
        im = img.convert("RGB")
        w, h = im.size
        s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))
        x = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
        x = x.transpose(2, 0, 1)[None]
        out = self.vision.run([self._vision_out], {self.vision.get_inputs()[0].name: x})[0][0]
        return out / (np.linalg.norm(out) + 1e-8)

    def embed_text(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text).ids[:77]
        feed = {"input_ids": np.array([ids], dtype=np.int64)}
        if "attention_mask" in self._text_inputs:
            feed["attention_mask"] = np.ones((1, len(ids)), dtype=np.int64)
        out = self.text.run([self._text_out], feed)[0][0]
        return out / (np.linalg.norm(out) + 1e-8)

    def _label_matrix(self, labels) -> np.ndarray:
        key = "|".join(k for k, _, _ in labels)
        if key not in self._label_cache:
            self._label_cache[key] = np.stack([self.embed_text(p) for _, _, p in labels])
        return self._label_cache[key]

    def classify(self, emb: np.ndarray, labels) -> list[tuple[str, float]]:
        logits = 100.0 * self._label_matrix(labels) @ emb
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)
        return [(labels[i][0], float(p[i])) for i in order]


_clip: Clip | None = None
_clip_failed = False
_lock = threading.Lock()


def get_clip() -> Clip | None:
    """CLIP을 쓸 수 있으면 돌려준다. 한 번 실패하면 이후에는 대체 방식으로 간다."""
    global _clip, _clip_failed
    if _clip or _clip_failed:
        return _clip
    with _lock:
        if _clip or _clip_failed:
            return _clip
        try:
            _clip = Clip()
        except Exception as e:  # noqa: BLE001
            log.warning("CLIP을 쓸 수 없어 색 분포 기반 특징으로 대체: %s", e)
            _clip_failed = True
    return _clip


def color_embedding(rgb: np.ndarray) -> np.ndarray:
    """CLIP이 없을 때 쓰는 단순한 특징: HSV 히스토그램 + 저해상도 밝기 배치."""
    import cv2

    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
    layout = (gray.astype(np.float32).flatten() / 255.0 - 0.5) * 0.15
    v = np.concatenate([np.sqrt(hist), layout]).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)
