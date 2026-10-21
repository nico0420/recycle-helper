"""Sorting rules for common household items in Taiwan.

The AI model only identifies *which* item it sees and *what state* it is in.
Where the item goes and how to prepare it come from this table, so the advice
stays consistent and can be reviewed line by line.

Rules differ between cities. The app tells people to check their local
environmental protection bureau for the final word.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Bin categories
RECYCLE = "recycle"
SPECIAL = "special"
TRASH = "trash"
FOOD = "food"

BINS: dict[str, dict[str, str]] = {
    RECYCLE: {"name": "資源回收", "hint": "交給資源回收車或回收點"},
    SPECIAL: {"name": "特殊回收", "hint": "單獨收集，送到指定回收點"},
    TRASH: {"name": "一般垃圾", "hint": "交給垃圾車"},
    FOOD: {"name": "廚餘", "hint": "瀝乾後倒進廚餘桶"},
}

BIN_ORDER = [RECYCLE, SPECIAL, TRASH, FOOD]


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    bin: str
    group: str
    prep: tuple[str, ...]
    mistake: str
    examples: tuple[str, ...] = field(default_factory=tuple)


ITEMS: tuple[Item, ...] = (
    Item(
        id="pet_bottle",
        name="寶特瓶",
        bin=RECYCLE,
        group="塑膠類",
        prep=("倒空瓶內液體", "簡單沖洗", "撕下瓶身標籤", "瓶蓋另外回收"),
        mistake="瓶子裡還有飲料就丟進回收，會弄髒其他回收物。",
        examples=("礦泉水瓶", "茶飲瓶", "汽水瓶"),
    ),
    Item(
        id="plastic_container",
        name="塑膠容器",
        bin=RECYCLE,
        group="塑膠類",
        prep=("清除食物殘渣", "簡單沖洗", "瀝乾"),
        mistake="便當盒裡還有菜渣或醬汁，回收時會被當成垃圾。",
        examples=("塑膠便當盒", "手搖飲杯", "布丁杯"),
    ),
    Item(
        id="metal_can",
        name="鐵鋁罐",
        bin=RECYCLE,
        group="金屬類",
        prep=("倒空罐內液體", "簡單沖洗", "可以壓扁節省空間"),
        mistake="罐子沒倒乾淨，殘留的飲料會發臭、招蟲。",
        examples=("鋁罐飲料", "罐頭", "奶粉罐"),
    ),
    Item(
        id="glass_bottle",
        name="玻璃瓶",
        bin=RECYCLE,
        group="玻璃類",
        prep=("倒空瓶內液體", "簡單沖洗", "取下瓶蓋另外回收"),
        mistake="破掉的玻璃直接丟，容易割傷清潔人員。請用報紙包好並寫上「碎玻璃」。",
        examples=("醬油瓶", "酒瓶", "玻璃罐"),
    ),
    Item(
        id="tetra_pak",
        name="鋁箔包",
        bin=RECYCLE,
        group="紙容器類",
        prep=("抽出吸管", "倒空", "簡單沖洗", "壓扁"),
        mistake="把鋁箔包當成一般紙類。它內層有鋁箔和塑膠，要放在紙容器類。",
        examples=("鋁箔包飲料", "牛奶紙盒", "豆漿紙盒"),
    ),
    Item(
        id="paper_container",
        name="紙杯與紙餐盒",
        bin=RECYCLE,
        group="紙容器類",
        prep=("剩下的食物倒進廚餘", "簡單沖洗", "疊好"),
        mistake="和報紙、紙箱放在一起。紙杯、紙餐盒有淋膜，屬於紙容器類。",
        examples=("紙便當盒", "紙杯", "紙碗"),
    ),
    Item(
        id="paper",
        name="紙類",
        bin=RECYCLE,
        group="紙類",
        prep=("攤平或拆開", "保持乾燥", "綁好或放進紙袋"),
        mistake="混進用過的衛生紙或沾到油汙的紙，會讓整批紙降低價值。",
        examples=("報紙", "紙箱", "影印紙"),
    ),
    Item(
        id="styrofoam",
        name="保麗龍",
        bin=RECYCLE,
        group="保麗龍類",
        prep=("清除食物殘渣", "撕掉膠帶與貼紙"),
        mistake="沾滿油汙或食物的保麗龍沒有清乾淨就回收。",
        examples=("保麗龍碗", "家電緩衝材"),
    ),
    Item(
        id="battery",
        name="廢乾電池",
        bin=SPECIAL,
        group="電池",
        prep=("單獨收集，不要和其他垃圾混在一起", "送到便利商店或量販店的回收點"),
        mistake="丟進一般垃圾。電池含有重金屬，混入垃圾可能汙染環境。",
        examples=("三號電池", "四號電池", "鈕扣電池"),
    ),
    Item(
        id="tissue",
        name="用過的衛生紙",
        bin=TRASH,
        group="一般垃圾",
        prep=("直接丟一般垃圾",),
        mistake="當成紙類回收。用過的衛生紙纖維太短又不衛生，無法再製。",
        examples=("衛生紙", "面紙", "濕紙巾"),
    ),
    Item(
        id="food_waste",
        name="廚餘",
        bin=FOOD,
        group="廚餘",
        prep=("瀝乾水分", "依縣市規定分生廚餘、熟廚餘", "不要連塑膠袋一起倒"),
        mistake="連同塑膠袋一起倒進廚餘桶。",
        examples=("果皮", "剩菜", "茶葉渣"),
    ),
)

UNKNOWN_ID = "unknown"

_BY_ID = {item.id: item for item in ITEMS}


def get_item(item_id: str) -> Item | None:
    return _BY_ID.get(item_id)


def item_ids() -> list[str]:
    return [item.id for item in ITEMS]


def describe_for_model() -> str:
    """A compact list of the items the model may choose from."""
    lines = []
    for item in ITEMS:
        examples = "、".join(item.examples)
        lines.append(f"- {item.id}: {item.name}（例如 {examples}）")
    lines.append(f"- {UNKNOWN_ID}: 看不出來，或不屬於以上任何一項")
    return "\n".join(lines)
