"""
Explainable Medical VLM
MedGemma + Visual Prototype Retrieval — research demo interface.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from gradio_client import Client, handle_file

from src.retrieval import (
    load_retrieval_artifacts,
    find_similar_embeddings,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Medical VLM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
HF_SPACE = "mdjisanmashrafi/medgemma-medical-backend"
MAX_CHAT_CONTEXT_TURNS = 3  # user+assistant pairs kept as context for follow-ups


# ============================================================
# VISUAL DESIGN — CUSTOM CSS
# ============================================================

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #FFFFFF;
            --bg-secondary: #F6F7F9;
            --border: #E5E8EC;
            --text-primary: #191C22;
            --text-secondary: #6B7280;
            --accent: #2E5FE8;
            --accent-soft: rgba(46, 95, 232, 0.08);
            --success: #1F9D6B;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: "Inter", -apple-system, "Segoe UI", sans-serif;
        }

        [data-testid="stHeader"] { background-color: transparent; }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1140px;
        }

        h1, h2, h3, h4, p, span, label, .stMarkdown {
            color: var(--text-primary);
        }

        /* Page header */
        .app-title {
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin: 0;
        }
        .app-subtitle {
            color: var(--accent);
            font-size: 0.95rem;
            font-weight: 500;
            margin: 0.2rem 0 0.4rem 0;
        }
        .app-desc {
            color: var(--text-secondary);
            font-size: 0.88rem;
            margin: 0 0 1.6rem 0;
        }
        .app-header-rule {
            border: none;
            border-top: 1px solid var(--border);
            margin: 0 0 0.4rem 0;
        }

        /* Numbered section headers */
        .section-header {
            display: flex;
            align-items: baseline;
            gap: 0.65rem;
            margin: 2.4rem 0 1rem 0;
        }
        .section-number {
            color: var(--accent);
            font-weight: 700;
            font-size: 0.92rem;
            font-variant-numeric: tabular-nums;
        }
        .section-title {
            font-size: 1.12rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Cards */
        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.15rem 1.3rem;
        }

        .answer-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            padding: 1rem 1.2rem;
            font-size: 0.96rem;
            line-height: 1.6;
        }
        .answer-card .tag {
            color: var(--accent);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            display: block;
            margin-bottom: 0.4rem;
        }

        /* Status pill */
        .status-pill {
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 500;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background-color: var(--accent-soft);
            color: var(--accent);
        }
        .status-pill.ready { background-color: var(--accent-soft); color: var(--accent); }
        .status-pill.done { background-color: rgba(31, 157, 107, 0.1); color: var(--success); }

        /* Similar-case gallery cards */
        .case-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }
        .case-card .case-body {
            padding: 0.6rem 0.75rem 0.8rem 0.75rem;
        }
        .case-rank {
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }
        .case-sim {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .case-proto {
            font-size: 0.8rem;
            color: var(--accent);
            font-weight: 600;
        }

        /* Buttons */
        .stButton > button {
            background-color: var(--accent);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.1rem;
            transition: opacity 0.15s ease;
        }
        .stButton > button:hover { opacity: 0.88; color: #FFFFFF; }

        /* Metrics */
        [data-testid="stMetric"] {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 1rem;
        }
        [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.75rem !important; }
        [data-testid="stMetricValue"] { color: var(--text-primary) !important; }

        /* Chat */
        [data-testid="stChatMessage"] {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--bg-secondary);
            border: 1px dashed var(--border);
            border-radius: 10px;
        }

        hr { border-color: var(--border); }

        .footer-note {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.78rem;
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(number: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-number">{number}</span>'
        f'<span class="section-title">{title}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

def ensure_session_defaults() -> None:
    defaults = {
        "image_hash": None,
        "chat_history": [],
        "query_embedding": None,
        "retrieval_results": None,
        "recommended_prototype": None,
        "last_error": None,
        "last_error_detail": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_image_specific_state() -> None:
    st.session_state["chat_history"] = []
    st.session_state["query_embedding"] = None
    st.session_state["retrieval_results"] = None
    st.session_state["recommended_prototype"] = None
    st.session_state["last_error"] = None
    st.session_state["last_error_detail"] = None


def get_image_hash(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def set_error(message: str, exc: Exception = None) -> None:
    st.session_state["last_error"] = message
    st.session_state["last_error_detail"] = exc


def render_pending_error() -> None:
    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])
        if st.session_state["last_error_detail"] is not None:
            with st.expander("Technical details"):
                st.exception(st.session_state["last_error_detail"])
        st.session_state["last_error"] = None
        st.session_state["last_error_detail"] = None


# ============================================================
# ARTIFACTS + BACKEND CLIENT
# ============================================================

@st.cache_resource
def load_artifacts():
    return load_retrieval_artifacts()


@st.cache_resource
def load_medgemma_client() -> Client:
    hf_token = st.secrets["HF_TOKEN"]
    return Client(HF_SPACE, token=hf_token)


def save_temp_image(image: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    image.save(tmp.name, format="PNG")
    return tmp.name


class BackendError(Exception):
    pass


def call_analyze_image(image: Image.Image, question: str) -> str:
    client = load_medgemma_client()
    image_path = save_temp_image(image)
    try:
        result = client.predict(
            handle_file(image_path),
            question,
            api_name="/analyze_image",
        )
        return str(result).strip()
    except Exception as exc:
        raise BackendError("MedGemma analysis is currently unavailable.") from exc
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass


def call_embed_image(image: Image.Image) -> np.ndarray:
    client = load_medgemma_client()
    image_path = save_temp_image(image)
    try:
        result = client.predict(
            handle_file(image_path),
            api_name="/embed_image",
        )
        return np.asarray(result, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise BackendError("Visual embedding generation failed.") from exc
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass


def build_question_with_context(question: str, history: list) -> str:
    if not history:
        return question

    recent = history[-(MAX_CHAT_CONTEXT_TURNS * 2):]
    lines = [
        f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
        for turn in recent
    ]
    context_block = "\n".join(lines)
    return (
        "Previous conversation:\n"
        f"{context_block}\n\n"
        f"Current question: {question}"
    )


def run_retrieval(
    image: Image.Image,
    visual_embeddings: np.ndarray,
    valid_indices: np.ndarray,
    cluster_labels: np.ndarray,
) -> None:
    query_embedding_vector = call_embed_image(image)

    if query_embedding_vector.shape[0] != visual_embeddings.shape[1]:
        raise BackendError(
            "The returned embedding has an unexpected dimension "
            f"({query_embedding_vector.shape[0]} vs "
            f"{visual_embeddings.shape[1]})."
        )

    results = find_similar_embeddings(
        query_embedding_vector,
        visual_embeddings,
        valid_indices,
        cluster_labels,
        top_k=5,
    )

    st.session_state["query_embedding"] = query_embedding_vector
    st.session_state["retrieval_results"] = results
    st.session_state["recommended_prototype"] = results[0]["prototype_id"]


# ============================================================
# REPRESENTATIVE IMAGE LOOKUP (single source of truth)
# ============================================================

def get_prototype_representative_path(prototype_id: int):
    """Return the first available representative image for a prototype.

    This is the single lookup used everywhere a prototype needs a
    representative visual, so the same image is never sourced two
    different ways in two different sections.
    """
    folder = ROOT / "representative_images" / f"P{prototype_id:02d}"
    if not folder.exists():
        return None
    for rank in range(1, 6):
        candidate = folder / f"representative_{rank}.png"
        if candidate.exists():
            return candidate
    return None


# ============================================================
# APP START
# ============================================================

inject_custom_css()
ensure_session_defaults()

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error("Failed to load retrieval artifacts.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

visual_embeddings = artifacts["visual_embeddings"]
valid_indices = artifacts["valid_indices"]
cluster_labels = artifacts["cluster_labels"]
prototype_summary = artifacts["prototype_summary"]


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="app-title">Explainable Medical VLM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">MedGemma · Visual Retrieval · Prototype Reasoning</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-desc">Medical visual question answering with example-based, '
    "prototype-driven retrieval explanations.</div>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="app-header-rule">', unsafe_allow_html=True)


# ============================================================
# 01 — MEDICAL IMAGE
# ============================================================

section_header("01", "Medical Image")

uploaded_file = st.file_uploader(
    "Upload medical image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        '<div class="card" style="text-align:center; color:var(--text-secondary);">'
        "Upload a medical image to begin.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="footer-note">Research prototype — not for clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

current_hash = get_image_hash(uploaded_file)
if st.session_state["image_hash"] != current_hash:
    st.session_state["image_hash"] = current_hash
    reset_image_specific_state()

image = Image.open(uploaded_file).convert("RGB")

img_col, meta_col = st.columns([1.3, 1])

with img_col:
    st.image(image, use_container_width=True)

with meta_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Resolution", f"{image.width} × {image.height}")
    st.metric("Case database", f"{len(visual_embeddings):,} cases")
    has_analysis = len(st.session_state["chat_history"]) > 0
    has_retrieval = st.session_state["retrieval_results"] is not None
    status_label = "Ready for analysis"
    status_class = "ready"
    if has_analysis and has_retrieval:
        status_label = "Analysis and retrieval complete"
        status_class = "done"
    elif has_analysis:
        status_label = "Analysis complete"
        status_class = "done"
    st.markdown(
        f'<span class="status-pill {status_class}">{status_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 02 — MODEL ANALYSIS
# ============================================================

section_header("02", "Model Analysis")
st.markdown('<div class="card">', unsafe_allow_html=True)

question = st.text_input(
    "Question",
    value="What findings are visible in this image?",
    label_visibility="collapsed",
)

analyze_clicked = st.button("Analyze Image", key="analyze_btn")

if analyze_clicked and question.strip():
    with st.spinner("Analyzing image..."):
        try:
            contextual_question = build_question_with_context(
                question.strip(), st.session_state["chat_history"]
            )
            answer = call_analyze_image(image, contextual_question)
            st.session_state["chat_history"].append(
                {"role": "user", "content": question.strip()}
            )
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
        except BackendError as exc:
            set_error(str(exc), exc.__cause__)

last_turn = next(
    (m for m in reversed(st.session_state["chat_history"]) if m["role"] == "assistant"),
    None,
)
if last_turn:
    st.markdown(
        f'<div class="answer-card"><span class="tag">MEDGEMMA</span>{last_turn["content"]}</div>',
        unsafe_allow_html=True,
    )
else:
    st.caption("No analysis yet.")

st.markdown("</div>", unsafe_allow_html=True)
render_pending_error()


# ============================================================
# 03 — SIMILAR CASES  (single primary visual gallery)
# ============================================================

section_header("03", "Similar Cases")
st.markdown('<div class="card">', unsafe_allow_html=True)

retrieve_clicked = st.button("Find Similar Cases", key="retrieve_btn")

if retrieve_clicked:
    with st.spinner("Searching similar cases..."):
        try:
            run_retrieval(image, visual_embeddings, valid_indices, cluster_labels)
        except BackendError as exc:
            set_error(str(exc), exc.__cause__)

retrieval_results = st.session_state["retrieval_results"]
recommended_prototype = st.session_state["recommended_prototype"]

if retrieval_results:
    cols = st.columns(len(retrieval_results))
    for col, result in zip(cols, retrieval_results):
        with col:
            rep_path = get_prototype_representative_path(result["prototype_id"])
            st.markdown('<div class="case-card">', unsafe_allow_html=True)
            if rep_path is not None:
                st.image(Image.open(rep_path).convert("RGB"), use_container_width=True)
            st.markdown(
                f'<div class="case-body">'
                f'<div class="case-rank">RANK {result["rank"]:02d}</div>'
                f'<div class="case-sim">{result["similarity"] * 100:.1f}%</div>'
                f'<div class="case-proto">P{result["prototype_id"]:02d}</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )
else:
    st.caption("Run retrieval to view the top visually similar cases.")

st.markdown("</div>", unsafe_allow_html=True)
render_pending_error()


# ============================================================
# 04 — PROTOTYPE EXPLANATION (text/metadata only — no image repeat)
# ============================================================

section_header("04", "Prototype Explanation")
st.markdown('<div class="card">', unsafe_allow_html=True)

if recommended_prototype is not None:
    summary_row = prototype_summary[
        prototype_summary["prototype_id"] == recommended_prototype
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Prototype", f"P{recommended_prototype:02d}")

    num_cases = None
    if not summary_row.empty:
        row = summary_row.iloc[0]
        num_cases = int(row["num_images"])
        c2.metric("Cases in Prototype", num_cases)
        if "representative_dataset_index" in row:
            c3.metric("Representative Index", int(row["representative_dataset_index"]))

    st.caption(
        f"The uploaded image's closest match belongs to prototype "
        f"P{recommended_prototype:02d}, a cluster of "
        f"{num_cases if num_cases is not None else '—'} "
        f"visually related cases in the retrieval database."
    )
else:
    st.caption("Run retrieval above to identify the associated prototype.")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 05 — EXPLAINABILITY (text-based, honest scope)
# ============================================================

section_header("05", "Explainability")
st.markdown('<div class="card">', unsafe_allow_html=True)

if recommended_prototype is not None and retrieval_results:
    best = retrieval_results[0]
    st.markdown(
        f"The uploaded image was compared against **{len(visual_embeddings):,}** "
        f"precomputed medical visual embeddings using cosine similarity on the "
        f"MedGemma vision representation. Its closest match was assigned to "
        f"**prototype P{recommended_prototype:02d}** with a visual similarity of "
        f"**{best['similarity'] * 100:.1f}%**."
    )
    st.caption(
        "This score reflects embedding-space visual similarity, not a "
        "diagnostic confidence or clinical probability."
    )
    st.caption(
        "Localization of specific affected regions is not currently produced "
        "by this pipeline; explanations are based on retrieval and prototype "
        "membership only."
    )
else:
    st.caption("Run retrieval above to generate an image-specific explanation.")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 06 — ASK ABOUT THIS IMAGE
# ============================================================

section_header("06", "Ask About This Image")

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

follow_up = st.chat_input("Ask a follow-up question about this image")

if follow_up and follow_up.strip():
    follow_up = follow_up.strip()

    st.session_state["chat_history"].append({"role": "user", "content": follow_up})

    with st.spinner("Thinking..."):
        try:
            previous_history = st.session_state["chat_history"][:-1]
            contextual_question = build_question_with_context(follow_up, previous_history)
            answer = call_analyze_image(image, contextual_question)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
        except BackendError as exc:
            st.session_state["chat_history"].pop()
            set_error(str(exc), exc.__cause__)

    st.rerun()

render_pending_error()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-note">Research prototype — not for clinical diagnosis.</div>',
    unsafe_allow_html=True,
)
