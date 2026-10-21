"""Turn a model's identification into advice a person can act on."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .catalog import BINS, UNKNOWN_ID, Item, get_item


@dataclass(frozen=True)
class Observation:
    """What the model saw. Advice is derived from this plus the catalog."""

    item_id: str
    confidence: float
    description: str = ""
    object_name: str = ""
    photo_clear: bool = True
    has_liquid: bool = False
    has_food_residue: bool = False
    has_label: bool = False
    is_broken: bool = False


@dataclass(frozen=True)
class Advice:
    observation: Observation
    item: Item | None
    bin: str | None
    bin_name: str
    steps: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    low_confidence: bool = False
    unlisted: bool = False

    @property
    def recognized(self) -> bool:
        return self.item is not None


LOW_CONFIDENCE = 0.6


class ParseError(ValueError):
    """The model's reply could not be read as an identification."""


def parse_observation(raw: str | dict) -> Observation:
    """Read the model's JSON reply. Unknown item ids become 'unknown'."""
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError("AI 回傳的內容不是有效的 JSON。") from exc
    else:
        data = raw

    if not isinstance(data, dict):
        raise ParseError("AI 回傳的內容格式不對。")

    item_id = str(data.get("item_id", UNKNOWN_ID)).strip()
    if get_item(item_id) is None:
        item_id = UNKNOWN_ID

    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    def flag(key: str) -> bool:
        value = data.get(key, False)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    return Observation(
        item_id=item_id,
        confidence=confidence,
        description=str(data.get("description", "")).strip()[:200],
        object_name=str(data.get("object_name", "")).strip()[:40],
        photo_clear=flag("photo_clear") if "photo_clear" in data else True,
        has_liquid=flag("has_liquid"),
        has_food_residue=flag("has_food_residue"),
        has_label=flag("has_label"),
        is_broken=flag("is_broken"),
    )


def build_advice(obs: Observation) -> Advice:
    item = get_item(obs.item_id)
    if item is None:
        if obs.photo_clear and obs.object_name:
            # The model saw the object clearly; it just isn't one we have rules for.
            return Advice(
                observation=obs,
                item=None,
                bin=None,
                bin_name="分類表沒有收錄",
                steps=(
                    "詢問當地清潔隊或環保局該怎麼處理",
                    "或到「分類表」分頁找材質相近的品項參考",
                ),
                unlisted=True,
            )
        return Advice(
            observation=obs,
            item=None,
            bin=None,
            bin_name="看不清楚",
            steps=(
                "把物品放在單純的背景前，讓它佔滿大半個畫面",
                "在光線充足的地方，換個角度再拍一次",
            ),
        )

    warnings: list[str] = []
    if obs.has_liquid:
        warnings.append("裡面看起來還有液體，請先倒空。")
    if obs.has_food_residue:
        warnings.append("看起來還有食物殘渣，請先清掉、沖一下。")
    if obs.has_label and item.id == "pet_bottle":
        warnings.append("瓶身還有標籤，記得撕下來。")
    if obs.is_broken and item.id == "glass_bottle":
        warnings.append("玻璃已經破了，請用報紙包好並寫上「碎玻璃」。")
    if obs.is_broken and item.id == "lighting":
        warnings.append("燈管已經破了，請小心碎片，用厚紙或報紙包好再交給回收車。")

    return Advice(
        observation=obs,
        item=item,
        bin=item.bin,
        bin_name=BINS[item.bin]["name"],
        steps=item.prep,
        warnings=tuple(warnings),
        low_confidence=obs.confidence < LOW_CONFIDENCE,
    )
