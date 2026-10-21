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
    ("connection reset", "再試一次"),
])
def test_gemini_errors_are_explained(message, expected):
    client, _ = _fake_client(error=RuntimeError(message))
    with pytest.raises(ClassifierError, match=expected) as info:
        GeminiClassifier(api_key="x", client=client, sleep=lambda s: None).identify(b"img", "image/jpeg")
    assert info.value.detail == message


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


# Two kinds of "not recognized"

def test_clear_photo_of_unlisted_item():
    obs = parse_observation({"item_id": "unknown", "confidence": 0.9, "object_name": "眼鏡", "photo_clear": True})
    advice = build_advice(obs)
    assert advice.unlisted
    assert advice.bin_name == "分類表沒有收錄"
    assert any("清潔隊" in step for step in advice.steps)
    assert not any("再拍" in step for step in advice.steps)


def test_unclear_photo_suggests_retake():
    obs = parse_observation({"item_id": "unknown", "confidence": 0.2, "object_name": "", "photo_clear": False})
    advice = build_advice(obs)
    assert not advice.unlisted
    assert advice.bin_name == "看不清楚"
    assert any("再拍" in step for step in advice.steps)


def test_unclear_flag_wins_even_with_a_name():
    advice = build_advice(Observation(item_id=UNKNOWN_ID, confidence=0.3, object_name="杯子", photo_clear=False))
    assert advice.bin_name == "看不清楚"


def test_reply_without_photo_clear_is_treated_as_clear():
    assert parse_observation({"item_id": "paper", "confidence": 0.9}).photo_clear is True


def test_unlisted_items_do_not_count_toward_guess_accuracy():
    stats = Stats()
    glasses = build_advice(Observation(item_id=UNKNOWN_ID, confidence=0.9, object_name="眼鏡"))
    stats.add(glasses, None)
    assert stats.accuracy is None


def test_schema_requires_new_fields():
    assert "object_name" in RESPONSE_SCHEMA["required"]
    assert "photo_clear" in RESPONSE_SCHEMA["required"]


# New catalog items

@pytest.mark.parametrize("item_id, bin_id", [
    ("lighting", "special"),
    ("electronics", "recycle"),
    ("disc", "recycle"),
    ("plastic_bag", "recycle"),
    ("clothing", "recycle"),
    ("ceramic", "trash"),
    ("medicine", "trash"),
])
def test_new_items_have_expected_bins(item_id, bin_id):
    assert get_item(item_id).bin == bin_id


def test_broken_lamp_warning():
    advice = build_advice(Observation(item_id="lighting", confidence=0.9, is_broken=True))
    assert any("碎片" in w for w in advice.warnings)



# Overload: retry, backups, and never showing raw error text

def _reply(item_id="metal_can"):
    return json.dumps({"item_id": item_id, "confidence": 0.9, "description": "", "object_name": "",
                       "photo_clear": True, "has_liquid": False, "has_food_residue": False,
                       "has_label": False, "is_broken": False})


BUSY = "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}"


def _scripted_client(outcomes):
    """Each call takes the next outcome: an exception to raise, or reply text."""
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs["model"])
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)

    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)), calls


def test_busy_then_ok_retries_same_model_after_waiting():
    client, calls = _scripted_client([RuntimeError(BUSY), _reply()])
    waits = []
    clf = GeminiClassifier(api_key="x", model="gemini-3.5-flash-lite", client=client, sleep=waits.append)
    obs, used = clf.identify_detailed(b"img", "image/jpeg")
    assert obs.item_id == "metal_can"
    assert calls == ["gemini-3.5-flash-lite", "gemini-3.5-flash-lite"]
    assert used == "gemini-3.5-flash-lite"
    assert waits == [1.5]


def test_still_busy_moves_to_backup_model():
    client, calls = _scripted_client([RuntimeError(BUSY), RuntimeError(BUSY), _reply()])
    clf = GeminiClassifier(api_key="x", model="gemini-3.5-flash-lite", client=client, sleep=lambda s: None)
    obs, used = clf.identify_detailed(b"img", "image/jpeg")
    assert calls == ["gemini-3.5-flash-lite", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    assert used == "gemini-3.1-flash-lite"


def test_backups_skip_the_chosen_model():
    clf = GeminiClassifier(api_key="x", model="gemini-3.8-flash", client=object(), sleep=lambda s: None)
    assert clf.backups == ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
    clf = GeminiClassifier(api_key="x", model="gemini-3.5-flash-lite", client=object(), sleep=lambda s: None)
    assert clf.backups == ("gemini-3.1-flash-lite",)


def test_all_busy_gives_short_message_and_keeps_detail():
    client, calls = _scripted_client([RuntimeError(BUSY)])
    clf = GeminiClassifier(api_key="x", model="gemini-3.5-flash-lite", client=client, sleep=lambda s: None)
    with pytest.raises(ClassifierError) as info:
        clf.identify(b"img", "image/jpeg")
    err = info.value
    assert len(calls) == 3
    assert err.busy
    assert "太忙" in str(err)
    assert "{" not in str(err) and "503" not in str(err)
    assert "high demand" in err.detail


def test_non_busy_error_is_not_retried():
    client, calls = _scripted_client([RuntimeError("401 API key not valid")])
    with pytest.raises(ClassifierError):
        GeminiClassifier(api_key="x", client=client, sleep=lambda s: None).identify(b"img", "image/jpeg")
    assert len(calls) == 1


@pytest.mark.parametrize("raw", [BUSY, "401 API key not valid", "429 quota", "boom {'x': 1}", "Request timed out"])
def test_messages_never_contain_raw_text(raw):
    client, _ = _scripted_client([RuntimeError(raw)])
    with pytest.raises(ClassifierError) as info:
        GeminiClassifier(api_key="x", client=client, sleep=lambda s: None).identify(b"img", "image/jpeg")
    message = str(info.value)
    assert "{" not in message and "'" not in message
    assert all(ord(ch) > 127 or ch in " .,_" or ch.isalnum() for ch in message)
