"""Identify an item from a photo: with Gemini, or with built-in demo answers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .catalog import UNKNOWN_ID, describe_for_model, item_ids
from .result import Observation, parse_observation

DEFAULT_MODEL = "gemini-3.5-flash-lite"

PROMPT = f"""你是資源回收分類助理。請看照片中最主要的那一件物品，從下面清單選出最符合的一項。

{describe_for_model()}

只根據照片中看得到的東西判斷，不要猜測看不到的部分。

- object_name：用繁體中文寫出這是什麼東西，例如「眼鏡」「寶特瓶」，不超過 10 字。看不出來就留空。
- photo_clear：照片是否清楚到足以認出物品。模糊、太暗、物品太小或被遮住時填 false。
- 物品看得清楚、但不屬於清單中任何一項時（例如眼鏡、雨傘），選 {UNKNOWN_ID}，photo_clear 填 true，並寫出 object_name。
- 照片不清楚，或有多件不同物品難以判斷主角時，選 {UNKNOWN_ID}，photo_clear 填 false，並降低 confidence。
- description：用繁體中文一句話描述你看到的物品，不超過 30 字。"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string", "enum": item_ids() + [UNKNOWN_ID]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "description": {"type": "string"},
        "object_name": {"type": "string"},
        "photo_clear": {"type": "boolean"},
        "has_liquid": {"type": "boolean"},
        "has_food_residue": {"type": "boolean"},
        "has_label": {"type": "boolean"},
        "is_broken": {"type": "boolean"},
    },
    "required": [
        "item_id",
        "confidence",
        "description",
        "object_name",
        "photo_clear",
        "has_liquid",
        "has_food_residue",
        "has_label",
        "is_broken",
    ],
}


class Classifier(Protocol):
    def identify(self, image: bytes, mime_type: str) -> Observation: ...


class ClassifierError(RuntimeError):
    """Identification failed.

    The message is short, plain Chinese and safe to show to anyone.
    `detail` keeps the original technical error for troubleshooting.
    """

    def __init__(self, message: str, detail: str = "", busy: bool = False):
        super().__init__(message)
        self.detail = detail
        self.busy = busy


# Tried in order when the chosen model is overloaded.
BACKUP_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
RETRY_WAIT_SECONDS = 1.5


class GeminiClassifier:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        backups: tuple[str, ...] = BACKUP_MODELS,
        client=None,
        sleep=None,
    ):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        if sleep is None:
            import time

            sleep = time.sleep
        self._client = client
        self._sleep = sleep
        self.model = model
        self.backups = tuple(m for m in dict.fromkeys(backups) if m != model)

    def identify(self, image: bytes, mime_type: str) -> Observation:
        return self.identify_detailed(image, mime_type)[0]

    def identify_detailed(self, image: bytes, mime_type: str) -> tuple[Observation, str]:
        """Like identify, and also says which model answered."""
        # When Google is overloaded: wait and retry the chosen model once, then try the backups.
        attempts = [self.model, self.model, *self.backups]
        last_exc: Exception | None = None

        for index, model in enumerate(attempts):
            if index == 1:
                self._sleep(RETRY_WAIT_SECONDS)
            try:
                response = self._call(model, image, mime_type)
            except Exception as exc:  # network, quota, bad key, unknown model, overload
                if _is_busy(exc):
                    last_exc = exc
                    continue
                raise _to_error(exc) from exc

            text = getattr(response, "text", None)
            if not text:
                raise ClassifierError("AI 沒有回傳結果，可能是照片被安全機制擋下。請換一張照片再試。")
            return parse_observation(text), model

        raise _to_error(last_exc) from last_exc

    def _call(self, model: str, image: bytes, mime_type: str):
        from google.genai import types

        return self._client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=image, mime_type=mime_type), PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
                temperature=0.1,
            ),
        )


def _is_busy(exc: Exception) -> bool:
    """Google is temporarily overloaded (HTTP 503), so trying again or elsewhere may work."""
    lowered = str(exc).lower()
    return any(s in lowered for s in ("503", "unavailable", "overloaded", "high demand"))


def _to_error(exc: Exception | None) -> ClassifierError:
    detail = str(exc or "")[:2000]
    lowered = detail.lower()
    if exc is not None and _is_busy(exc):
        return ClassifierError("Google 的 AI 伺服器現在太忙，請過一會兒再試。", detail, busy=True)
    if "api key" in lowered or "permission" in lowered or "401" in lowered or "403" in lowered:
        return ClassifierError("API 金鑰無效或沒有權限。請確認 GEMINI_API_KEY 設定正確。", detail)
    if "429" in lowered or "quota" in lowered or "resource_exhausted" in lowered:
        return ClassifierError("今天的 AI 使用次數已達上限，請稍後再試，或先改用示範模式。", detail, busy=True)
    if "404" in lowered or "not found" in lowered:
        return ClassifierError("找不到指定的模型。請確認 GEMINI_MODEL 設定的名稱是否正確。", detail)
    if "timeout" in lowered or "timed out" in lowered:
        return ClassifierError("AI 太久沒有回應，請再試一次。", detail, busy=True)
    return ClassifierError("AI 暫時無法辨識，請再試一次。", detail)


@dataclass(frozen=True)
class Sample:
    id: str
    title: str
    path: Path
    answer: Observation


class DemoClassifier:
    """Returns prepared answers for the built-in sample photos.

    Used when no API key is set, so anyone can try the app. Photos that are not
    one of the samples are reported as not recognized.
    """

    def __init__(self, samples: list[Sample]):
        self.samples = samples
        self._by_hash = {_digest(s.path.read_bytes()): s for s in samples if s.path.exists()}

    def identify(self, image: bytes, mime_type: str) -> Observation:
        sample = self._by_hash.get(_digest(image))
        if sample is None:
            return Observation(
                item_id=UNKNOWN_ID,
                confidence=0.0,
                description="示範模式只認得內建的範例照片",
            )
        return sample.answer


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def default_samples() -> list[Sample]:
    def s(file: str, title: str, **answer) -> Sample:
        return Sample(id=file.split(".")[0], title=title, path=SAMPLES_DIR / file, answer=Observation(**answer))

    return [
        s("pet_bottle.png", "還沒喝完的寶特瓶",
          item_id="pet_bottle", confidence=0.93, description="一支貼著標籤、裡面還有一些水的寶特瓶",
          has_liquid=True, has_label=True),
        s("tetra_pak.png", "附吸管的鋁箔包",
          item_id="tetra_pak", confidence=0.9, description="一個插著吸管的鋁箔包飲料"),
        s("paper_container.png", "吃完的紙便當盒",
          item_id="paper_container", confidence=0.86, description="一個還有少量飯菜殘渣的紙便當盒",
          has_food_residue=True),
        s("metal_can.png", "鋁罐",
          item_id="metal_can", confidence=0.95, description="一個空的鋁罐飲料"),
        s("battery.png", "乾電池",
          item_id="battery", confidence=0.91, description="兩顆三號乾電池"),
        s("tissue.png", "用過的衛生紙",
          item_id="tissue", confidence=0.84, description="一團揉皺的衛生紙"),
    ]
