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
照片模糊、物品太小，或有多件不同物品難以判斷時，選 {UNKNOWN_ID} 並降低 confidence。
description 用繁體中文一句話描述你看到的物品，不超過 30 字。"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string", "enum": item_ids() + [UNKNOWN_ID]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "description": {"type": "string"},
        "has_liquid": {"type": "boolean"},
        "has_food_residue": {"type": "boolean"},
        "has_label": {"type": "boolean"},
        "is_broken": {"type": "boolean"},
    },
    "required": [
        "item_id",
        "confidence",
        "description",
        "has_liquid",
        "has_food_residue",
        "has_label",
        "is_broken",
    ],
}


class Classifier(Protocol):
    def identify(self, image: bytes, mime_type: str) -> Observation: ...


class ClassifierError(RuntimeError):
    """Identification failed. The message is safe to show to the person."""


class GeminiClassifier:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, client=None):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model

    def identify(self, image: bytes, mime_type: str) -> Observation:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=image, mime_type=mime_type), PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_SCHEMA,
                    temperature=0.1,
                ),
            )
        except Exception as exc:  # network, quota, bad key, unknown model
            raise ClassifierError(_explain(exc)) from exc

        text = getattr(response, "text", None)
        if not text:
            raise ClassifierError("AI 沒有回傳結果，可能是照片被安全機制擋下。請換一張照片再試。")
        return parse_observation(text)


def _explain(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "api key" in lowered or "permission" in lowered or "401" in lowered or "403" in lowered:
        return "API 金鑰無效或沒有權限。請確認 GEMINI_API_KEY 設定正確。"
    if "429" in lowered or "quota" in lowered or "resource_exhausted" in lowered:
        return "已達到 API 使用上限，請稍後再試，或先改用示範模式。"
    if "404" in lowered or "not found" in lowered:
        return "找不到指定的模型。請確認 GEMINI_MODEL 設定的名稱是否正確。"
    return f"呼叫 AI 時發生錯誤：{message[:160]}"


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
