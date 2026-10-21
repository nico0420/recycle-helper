import json
from types import SimpleNamespace

import pytest

from recycle_helper.catalog import BINS, ITEMS, UNKNOWN_ID, describe_for_model, get_item
from recycle_helper.classifier import (
    RESPONSE_SCHEMA,
    ClassifierError,
    DemoClassifier,
    GeminiClassifier,
    default_samples,
)
from recycle_helper.result import Observation, ParseError, build_advice, parse_observation
from recycle_helper.stats import Stats


# Catalog

def test_catalog_ids_are_unique_and_bins_valid():
    ids = [item.id for item in ITEMS]
    assert len(ids) == len(set(ids))
    assert UNKNOWN_ID not in ids
    for item in ITEMS:
        assert item.bin in BINS
        assert item.prep, item.id
        assert item.mistake, item.id


def test_model_prompt_and_schema_list_every_item():
    listing = describe_for_model()
    enum = RESPONSE_SCHEMA["properties"]["item_id"]["enum"]
    for item in ITEMS:
        assert item.id in listing
        assert item.id in enum
    assert UNKNOWN_ID in enum


# Parsing the model's reply

def test_parse_valid_reply():
    obs = parse_observation(json.dumps({
        "item_id": "pet_bottle", "confidence": 0.9, "description": "寶特瓶",
        "has_liquid": True, "has_food_residue": False, "has_label": True, "is_broken": False,
    }))
    assert obs.item_id == "pet_bottle"
    assert obs.has_liquid and obs.has_label


def test_parse_fenced_json():
    obs = parse_observation('```json\n{"item_id": "metal_can", "confidence": 0.8}\n```')
    assert obs.item_id == "metal_can"


def test_parse_invented_item_becomes_unknown():
    obs = parse_observation({"item_id": "spaceship", "confidence": 0.99})
    assert obs.item_id == UNKNOWN_ID


def test_parse_clamps_confidence_and_reads_string_flags():
    obs = parse_observation({"item_id": "battery", "confidence": 7, "has_liquid": "false", "is_broken": "true"})
    assert obs.confidence == 1.0
    assert obs.has_liquid is False
    assert obs.is_broken is True


def test_parse_bad_confidence_is_zero():
    assert parse_observation({"item_id": "paper", "confidence": "high"}).confidence == 0.0


def test_parse_rejects_non_json():
    with pytest.raises(ParseError):
        parse_observation("I think it's a bottle")
    with pytest.raises(ParseError):
        parse_observation("[1, 2]")


# Advice

def test_advice_comes_from_catalog():
    advice = build_advice(Observation(item_id="tetra_pak", confidence=0.9))
    assert advice.bin == "recycle"
    assert advice.steps == get_item("tetra_pak").prep
    assert not advice.low_confidence


def test_advice_warnings_follow_what_was_seen():
    advice = build_advice(Observation(item_id="pet_bottle", confidence=0.9, has_liquid=True, has_label=True))
    assert any("倒空" in w for w in advice.warnings)
    assert any("標籤" in w for w in advice.warnings)


def test_label_warning_only_for_bottles():
    advice = build_advice(Observation(item_id="metal_can", confidence=0.9, has_label=True))
    assert advice.warnings == ()


def test_low_confidence_is_flagged():
    assert build_advice(Observation(item_id="paper", confidence=0.4)).low_confidence


def test_unknown_item_gives_retry_advice():
    advice = build_advice(Observation(item_id=UNKNOWN_ID, confidence=0.1))
    assert not advice.recognized
    assert advice.bin is None


# Classifiers

def test_demo_classifier_answers_every_sample():
    samples = default_samples()
    demo = DemoClassifier(samples)
    for sample in samples:
        assert sample.path.exists(), sample.path
        obs = demo.identify(sample.path.read_bytes(), "image/png")
        assert obs == sample.answer
        assert get_item(obs.item_id) is not None


def test_demo_classifier_does_not_guess_other_photos():
    obs = DemoClassifier(default_samples()).identify(b"not a sample", "image/png")
    assert obs.item_id == UNKNOWN_ID


def _fake_client(text=None, error=None):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(text=text)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, calls


def test_gemini_classifier_sends_image_and_schema():
    reply = json.dumps({"item_id": "glass_bottle", "confidence": 0.88, "description": "玻璃瓶",
                        "has_liquid": False, "has_food_residue": False, "has_label": False, "is_broken": True})
    client, calls = _fake_client(text=reply)
    obs = GeminiClassifier(api_key="x", model="test-model", client=client).identify(b"img", "image/jpeg")
    assert obs.item_id == "glass_bottle" and obs.is_broken
    call = calls[0]
    assert call["model"] == "test-model"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == RESPONSE_SCHEMA
    assert call["contents"][0].inline_data.data == b"img"


@pytest.mark.parametrize("message, expected", [
    ("401 API key not valid", "金鑰"),
    ("429 RESOURCE_EXHAUSTED", "上限"),
    ("404 models/foo is not found", "模型"),
    ("connection reset", "錯誤"),
])
def test_gemini_errors_are_explained(message, expected):
    client, _ = _fake_client(error=RuntimeError(message))
    with pytest.raises(ClassifierError, match=expected):
        GeminiClassifier(api_key="x", client=client).identify(b"img", "image/jpeg")


def test_gemini_empty_reply_is_an_error():
    client, _ = _fake_client(text="")
    with pytest.raises(ClassifierError):
        GeminiClassifier(api_key="x", client=client).identify(b"img", "image/jpeg")


# Stats

def test_stats_track_guesses_and_misses():
    stats = Stats()
    bottle = build_advice(Observation(item_id="pet_bottle", confidence=0.9))
    pack = build_advice(Observation(item_id="tetra_pak", confidence=0.9))
    battery = build_advice(Observation(item_id="battery", confidence=0.9))
    stats.add(bottle, "recycle")
    stats.add(pack, "trash")
    stats.add(pack, "trash")
    stats.add(battery, None)

    assert stats.total == 4
    assert stats.guess_count == 3
    assert stats.accuracy == pytest.approx(1 / 3)
    assert stats.most_missed() == [("鋁箔包", 2)]
    assert stats.by_bin()["資源回收"] == 3
    assert stats.by_bin()["特殊回收"] == 1


def test_stats_accuracy_none_without_guesses():
    assert Stats().accuracy is None
