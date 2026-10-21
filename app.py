"""回收小幫手：拍一張照片，告訴你怎麼分類、丟之前要做什麼。"""

from __future__ import annotations

import html
import io
import os

import streamlit as st
from PIL import Image, ImageOps

from recycle_helper.catalog import BIN_ORDER, BINS, ITEMS
from recycle_helper.classifier import (
    DEFAULT_MODEL,
    ClassifierError,
    DemoClassifier,
    GeminiClassifier,
    default_samples,
)
from recycle_helper.result import ParseError, build_advice
from recycle_helper.stats import Stats

st.set_page_config(page_title="回收小幫手", page_icon="♻️", layout="centered")

BIN_COLORS = {"recycle": "#2E7D5B", "special": "#B26A00", "trash": "#5B6670", "food": "#7A5C2E"}

st.markdown(
    """
    <style>
      .bin-card { border-radius: 14px; padding: 18px 20px; color: #fff; margin: 4px 0 12px; }
      .bin-card .label { font-size: 14px; opacity: .85; margin: 0; }
      .bin-card .name { font-size: 32px; font-weight: 700; line-height: 1.3; margin: 2px 0 0; }
      .bin-card .hint { font-size: 15px; opacity: .9; margin: 4px 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def setting(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file
        value = None
    return value or os.environ.get(name)


@st.cache_resource
def demo_classifier() -> DemoClassifier:
    return DemoClassifier(default_samples())


@st.cache_resource
def gemini_classifier(api_key: str, model: str) -> GeminiClassifier:
    return GeminiClassifier(api_key=api_key, model=model)


def shrink(data: bytes, max_side: int = 1280) -> tuple[bytes, str]:
    """Rotate per EXIF, cap the size, and re-encode as JPEG to keep requests small."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=88)
    return out.getvalue(), "image/jpeg"


if "stats" not in st.session_state:
    st.session_state.stats = Stats()
if "result" not in st.session_state:
    st.session_state.result = None

api_key = setting("GEMINI_API_KEY")
model = setting("GEMINI_MODEL") or DEFAULT_MODEL
samples = default_samples()

st.title("♻️ 回收小幫手")
st.write("拍一張照片，告訴你它該丟哪裡、丟之前要做什麼。")

with st.sidebar:
    st.header("設定")
    if api_key:
        mode = st.radio("辨識方式", ["AI 辨識", "示範模式"], help="示範模式使用內建範例與預先準備的結果，不需要 API 金鑰。")
    else:
        mode = "示範模式"
        st.info("尚未設定 GEMINI_API_KEY，目前使用示範模式：只能辨識內建的範例照片。")
    guess_mode = st.toggle("先猜猜看再看答案", value=True, help="先選你會丟哪一桶，再看 AI 判斷的結果。")
    if mode == "AI 辨識":
        st.caption(f"模型：{model}")

tab_scan, tab_stats, tab_table = st.tabs(["辨識", "統計", "分類表"])

with tab_scan:
    image_bytes: bytes | None = None
    source_key = None

    if mode == "AI 辨識":
        upload_tab, camera_tab, sample_tab = st.tabs(["上傳照片", "拍照", "範例照片"])
        with upload_tab:
            uploaded = st.file_uploader("選一張回收物的照片", type=["jpg", "jpeg", "png", "webp", "heic"])
        with camera_tab:
            shot = st.camera_input("對準要丟的東西")
        with sample_tab:
            sample_choice = st.radio("或用範例照片試試", [s.title for s in samples], index=None)
        if shot is not None:
            image_bytes, source_key = shot.getvalue(), f"camera-{shot.file_id}"
        elif uploaded is not None:
            image_bytes, source_key = uploaded.getvalue(), f"upload-{uploaded.file_id}"
        elif sample_choice:
            sample = next(s for s in samples if s.title == sample_choice)
            image_bytes, source_key = sample.path.read_bytes(), f"sample-{sample.id}"
    else:
        sample_choice = st.radio("選一張範例照片", [s.title for s in samples], index=None, horizontal=True)
        if sample_choice:
            sample = next(s for s in samples if s.title == sample_choice)
            image_bytes, source_key = sample.path.read_bytes(), f"sample-{sample.id}"

    if image_bytes is None:
        st.caption("選好照片後，這裡會顯示辨識結果。")
    else:
        st.image(image_bytes, width=320)

        guess = None
        if guess_mode:
            guess_name = st.radio(
                "你會把它丟到哪裡？",
                [BINS[b]["name"] for b in BIN_ORDER],
                index=None,
                horizontal=True,
                key=f"guess-{source_key}",
            )
            guess = next((b for b in BIN_ORDER if BINS[b]["name"] == guess_name), None)

        ready = guess is not None or not guess_mode
        label = "看答案" if guess_mode else "開始辨識"
        if st.button(label, type="primary", disabled=not ready, key=f"run-{source_key}"):
            classifier = gemini_classifier(api_key, model) if mode == "AI 辨識" else demo_classifier()
            try:
                with st.spinner("辨識中…"):
                    if mode == "AI 辨識":
                        data, mime = shrink(image_bytes)
                    else:
                        data, mime = image_bytes, "image/png"
                    advice = build_advice(classifier.identify(data, mime))
            except (ClassifierError, ParseError) as exc:
                st.session_state.result = None
                st.error(str(exc))
            else:
                st.session_state.stats.add(advice, guess if advice.recognized else None)
                st.session_state.result = {"key": source_key, "advice": advice, "guess": guess}

        result = st.session_state.result
        if result and result["key"] == source_key:
            advice = result["advice"]
            guess = result["guess"]

            if advice.recognized:
                color = BIN_COLORS[advice.bin]
                st.markdown(
                    f"""<div class="bin-card" style="background:{color}">
                          <p class="label">{advice.item.name}・{advice.item.group}</p>
                          <p class="name">{advice.bin_name}</p>
                          <p class="hint">{BINS[advice.bin]["hint"]}</p>
                        </div>""",
                    unsafe_allow_html=True,
                )
                if guess:
                    if guess == advice.bin:
                        st.success("答對了！")
                    else:
                        st.error(f"你選了「{BINS[guess]['name']}」，正確是「{advice.bin_name}」。")

                obs = advice.observation
                caption = f"信心 {obs.confidence:.0%}"
                if obs.description:
                    caption = f"{obs.description}　{caption}"
                if mode == "示範模式":
                    caption += "　（示範模式：預先準備的結果）"
                st.caption(caption)
                if advice.low_confidence:
                    st.warning("AI 不太確定這個判斷，建議換個角度再拍一次，或對照分類表確認。")

                for warning in advice.warnings:
                    st.warning(warning)

                st.subheader("丟之前")
                st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(advice.steps, 1)))
                st.info(f"常見錯誤：{advice.item.mistake}")
            else:
                obs = advice.observation
                if advice.unlisted:
                    label = f"AI 認出這是「{html.escape(obs.object_name)}」"
                    hint = "目前的分類表還沒有這個品項，所以不提供分類建議。"
                else:
                    label = html.escape(obs.description) if obs.description else "照片中的物品看不太清楚"
                    hint = "AI 沒辦法從這張照片認出物品。"
                st.markdown(
                    f"""<div class="bin-card" style="background:{BIN_COLORS['trash']}">
                          <p class="label">{label}</p>
                          <p class="name">{advice.bin_name}</p>
                          <p class="hint">{hint}</p>
                        </div>""",
                    unsafe_allow_html=True,
                )
                if guess:
                    st.caption("這次沒有標準答案，所以不列入猜對率。")
                st.markdown("\n".join(f"- {step}" for step in advice.steps))

    st.caption("各縣市的回收規定可能不同，實際做法請以當地環保局公告為準。")

with tab_stats:
    stats: Stats = st.session_state.stats
    if stats.total == 0:
        st.write("還沒有辨識紀錄。到「辨識」分頁試一張照片吧。")
    else:
        col1, col2 = st.columns(2)
        col1.metric("已辨識", f"{stats.total} 件")
        accuracy = stats.accuracy
        col2.metric("猜對率", "—" if accuracy is None else f"{accuracy:.0%}", help=f"共猜了 {stats.guess_count} 次")

        st.subheader("各類別件數")
        st.bar_chart(stats.by_bin(), color="#2E7D5B", horizontal=True, sort=False)

        missed = stats.most_missed()
        st.subheader("最常分錯")
        if missed:
            st.markdown("\n".join(f"- {name}：{count} 次" for name, count in missed))
        else:
            st.write("目前沒有分錯的紀錄。")

        if st.button("清除紀錄"):
            st.session_state.stats = Stats()
            st.session_state.result = None
            st.rerun()
        st.caption("紀錄只存在這次的瀏覽視窗，關掉後就會清除。")

with tab_table:
    st.write("AI 只負責辨認是什麼東西，分類方式和處理步驟都依照這張表。")
    for bin_id in BIN_ORDER:
        items = [item for item in ITEMS if item.bin == bin_id]
        if not items:
            continue
        st.subheader(BINS[bin_id]["name"])
        st.table(
            {
                "品項": [item.name for item in items],
                "類別": [item.group for item in items],
                "例如": ["、".join(item.examples) for item in items],
                "丟之前": ["；".join(item.prep) for item in items],
            }
        )
