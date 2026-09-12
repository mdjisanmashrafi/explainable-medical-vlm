from pathlib import Path

app_code = r'''import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

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
MAX_CHAT_CONTEXT_TURNS = 3


# ============================================================
# RESEARCH-GRADE UI
# ============================================================

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #ffffff;
            --surface: #f7f9fb;
            --surface-2: #fbfcfd;
            --border: #e3e8ee;
            --border-strong: #d3dbe4;
            --text: #17202a;
            --muted: #667382;
            --subtle: #8b97a5;
            --accent: #1677a8;
            --accent-soft: #edf7fb;
            --success: #287a58;
            --danger: #b33a3a;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg);
            color: var(--text);
            font-family:
                Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                Roboto, Helvetica, Arial, sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.96);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2.8rem;
            padding-bottom: 4rem;
        }

        /* ---------- Global typography ---------- */

        h1, h2, h3, h4 {
            color: var(--text) !important;
            font-weight: 650 !important;
            letter-spacing: -0.02em;
        }

        p, li, label, .stMarkdown {
            color: var(--text);
        }

        /* ---------- Header ---------- */

        .research-header {
            padding-bottom: 1.65rem;
            margin-bottom: 2.1rem;
            border-bottom: 1px solid var(--border);
        }

        .research-kicker {
            color: var(--accent);
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .research-title {
            color: var(--text);
            font-size: 2.25rem;
            line-height: 1.08;
            font-weight: 700;
            letter-spacing: -0.035em;
            margin: 0;
        }

        .research-subtitle {
            color: var(--muted);
            font-size: 0.96rem;
            margin-top: 0.55rem;
        }

        .research-description {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.55;
            max-width: 760px;
            margin-top: 0.65rem;
        }

        /* ---------- Sections ---------- */

        .section-wrap {
            margin-top: 2.2rem;
        }

        .section-heading {
            display: flex;
            align-items: baseline;
            gap: 0.7rem;
            margin-bottom: 0.85rem;
        }

        .section-number {
            color: var(--accent);
            font-size: 0.73rem;
            font-weight: 750;
            letter-spacing: 0.08em;
        }

        .section-title {
            color: var(--text);
            font-size: 1.15rem;
            font-weight: 680;
            letter-spacing: -0.015em;
        }

        .section-caption {
            color: var(--muted);
            font-size: 0.79rem;
            margin: -0.35rem 0 0.9rem 1.95rem;
        }

        /* ---------- Surfaces ---------- */

        .surface {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.15rem;
        }

        .surface-tight {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }

        .analysis-card {
            background: #ffffff;
            border: 1px solid var(--border-strong);
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 1px 2px rgba(23, 32, 42, 0.03);
        }

        .analysis-label {
            color: var(--accent);
            font-size: 0.69rem;
            font-weight: 750;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .analysis-text {
            color: var(--text);
            font-size: 0.96rem;
            line-height: 1.7;
            white-space: pre-wrap;
        }

        .empty-state {
            border: 1px dashed var(--border-strong);
            border-radius: 10px;
            padding: 1.6rem;
            text-align: center;
            color: var(--muted);
            background: var(--surface-2);
            font-size: 0.84rem;
        }

        /* ---------- Metadata ---------- */

        .meta-label {
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.2rem;
        }

        .meta-value {
            color: var(--text);
            font-size: 0.91rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.69rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .similarity-note {
            color: var(--muted);
            font-size: 0.72rem;
            line-height: 1.45;
            margin-top: 0.7rem;
        }

        /* ---------- Similar-case cards ---------- */

        .case-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            height: 100%;
        }

        .case-meta {
            padding: 0.72rem 0.8rem 0.82rem 0.8rem;
        }

        .case-rank {
            color: var(--subtle);
            font-size: 0.67rem;
            font-weight: 750;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .case-score {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 0.22rem;
        }

        .case-prototype {
            color: var(--accent);
            font-size: 0.73rem;
            font-weight: 650;
            margin-top: 0.18rem;
        }

        /* ---------- Chat ---------- */

        [data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        /* ---------- Inputs ---------- */

        .stTextInput input,
        .stTextArea textarea {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border-strong) !important;
            border-radius: 8px !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border: 1px dashed var(--border-strong);
            border-radius: 10px;
        }

        /* ---------- Buttons ---------- */

        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--border-strong);
            background: #ffffff;
            color: var(--text);
            font-weight: 650;
            min-height: 2.55rem;
            transition: border-color 0.12s ease, background 0.12s ease;
        }

        .stButton > button:hover {
            border-color: var(--accent);
            background: var(--accent-soft);
            color: var(--accent);
        }

        /* Primary buttons */
        .primary-button .stButton > button {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
        }

        .primary-button .stButton > button:hover {
            background: #126789;
            border-color: #126789;
            color: #ffffff;
        }

        /* ---------- Metrics ---------- */

        [data-testid="stMetric"] {
            background: transparent;
            border: 0;
            padding: 0.15rem 0;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.7rem !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-size: 1.2rem !important;
            font-weight: 680 !important;
        }

        /* ---------- Tables ---------- */

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        /* ---------- Footer ---------- */

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--subtle);
            text-align: center;
            font-size: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

def ensure_session_defaults() -> None:
    defaults = {
        "image_hash": None,
        "chat_history": [],
        "analysis_result": None,
        "query_embedding": None,
        "retrieval_results": None,
        "recommended_prototype": None,
        "last_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_image_specific_state() -> None:
    st.session_state["chat_history"] = []
    st.session_state["analysis_result"] = None
    st.session_state["query_embedding"] = None
    st.session_state["retrieval_results"] = None
    st.session_state["recommended_prototype"] = None
    st.session_state["last_error"] = None


def get_image_hash(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


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


# ============================================================
# BACKEND
# ============================================================

class BackendError(Exception):
    """User-facing backend exception with optional technical detail."""

    def __init__(self, message: str, technical_detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.technical_detail = technical_detail


def call_analyze_image(image: Image.Image, question: str) -> str:
    client = load_medgemma_client()
    image_path = save_temp_image(image)

    try:
        result = client.predict(
            handle_file(image_path),
            question,
            api_name="/analyze_image",
        )

        answer = str(result).strip()

        if not answer:
            raise BackendError(
                "MedGemma returned an empty response.",
                "The /analyze_image endpoint completed without returning text.",
            )

        return answer

    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(
            "MedGemma analysis failed. Please try again.",
            repr(exc),
        ) from exc
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

        embedding = np.asarray(result, dtype=np.float32).reshape(-1)

        if embedding.size == 0:
            raise BackendError(
                "Visual embedding generation returned no data.",
                "The /embed_image endpoint returned an empty embedding.",
            )

        return embedding

    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(
            "Visual embedding generation failed. Please try again.",
            repr(exc),
        ) from exc
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
            "The returned visual embedding has an unexpected dimension.",
            (
                f"Returned dimension: {query_embedding_vector.shape[0]}; "
                f"expected: {visual_embeddings.shape[1]}."
            ),
        )

    results = find_similar_embeddings(
        query_embedding_vector,
        visual_embeddings,
        valid_indices,
        cluster_labels,
        top_k=5,
    )

    if not results:
        raise BackendError(
            "No visually similar cases were found.",
            "find_similar_embeddings returned an empty result list.",
        )

    st.session_state["query_embedding"] = query_embedding_vector
    st.session_state["retrieval_results"] = results
    st.session_state["recommended_prototype"] = results[0]["prototype_id"]


# ============================================================
# REPRESENTATIVE IMAGE HELPERS
# ============================================================

def get_representative_image(
    prototype_images: pd.DataFrame,
    prototype_id: int,
    preferred_rank: Optional[int] = None,
    excluded_paths: Optional[set] = None,
) -> Optional[tuple[Any, Path]]:
    """Return one representative image for a prototype, avoiding duplicates."""

    excluded_paths = excluded_paths or set()

    subset = (
        prototype_images[
            prototype_images["prototype_id"] == prototype_id
        ]
        .sort_values("rank")
    )

    candidates = []

    if preferred_rank is not None:
        preferred = subset[subset["rank"] == preferred_rank]
        candidates.extend(list(preferred.iterrows()))

    candidates.extend(
        [
            item
            for item in subset.iterrows()
            if preferred_rank is None or item[1]["rank"] != preferred_rank
        ]
    )

    for _, row in candidates:
        rank = int(row["rank"])
        image_path = (
            ROOT
            / "representative_images"
            / f"P{prototype_id:02d}"
            / f"representative_{rank}.png"
        )

        if image_path.exists() and str(image_path) not in excluded_paths:
            return row, image_path

    return None


def get_similar_case_images(
    prototype_images: pd.DataFrame,
    retrieval_results: list,
) -> list[tuple[dict, Any, Path]]:
    """Map retrieved prototypes to unique representative images for the gallery."""

    selected = []
    used_paths: set = set()

    for result in retrieval_results:
        prototype_id = int(result["prototype_id"])

        preferred_rank = None
        if "rank" in result:
            try:
                preferred_rank = int(result["rank"])
            except (TypeError, ValueError):
                preferred_rank = None

        found = get_representative_image(
            prototype_images,
            prototype_id,
            preferred_rank=preferred_rank,
            excluded_paths=used_paths,
        )

        if found is None:
            continue

        row, image_path = found
        used_paths.add(str(image_path))
        selected.append((result, row, image_path))

    return selected


# ============================================================
# UI HELPERS
# ============================================================

def render_section(number: str, title: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-heading">
                <span class="section-number">{number}</span>
                <span class="section-title">{title}</span>
            </div>
            {f'<div class="section-caption">{caption}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error() -> None:
    error = st.session_state.get("last_error")

    if not error:
        return

    if isinstance(error, dict):
        message = error.get("message", "An unexpected error occurred.")
        technical = error.get("technical")
    else:
        message = str(error)
        technical = None

    st.error(message)

    if technical:
        with st.expander("Technical details"):
            st.code(technical)

    st.session_state["last_error"] = None


def render_analysis(answer: Optional[str]) -> None:
    if not answer:
        st.markdown(
            '<div class="empty-state">Run MedGemma analysis to generate an image-grounded response.</div>',
            unsafe_allow_html=True,
        )
        return

    safe_answer = answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-label">MedGemma · Image-grounded response</div>
            <div class="analysis-text">{safe_answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prototype_statistics(
    prototype_summary: pd.DataFrame,
    selected_prototype: int,
) -> None:
    selected = prototype_summary[
        prototype_summary["prototype_id"] == selected_prototype
    ]

    if selected.empty:
        st.caption("No summary record is available for this prototype.")
        return

    row = selected.iloc[0]

    # Core known field
    if "num_images" in row.index:
        st.metric("Cases in prototype", f"{int(row['num_images']):,}")

    # Show only real fields present in the artifact.
    excluded = {
        "prototype_id",
        "num_images",
        "id",
        "rank",
    }

    available = []
    for column in row.index:
        if column in excluded:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        if isinstance(value, (str, int, float, np.integer, np.floating, bool)):
            available.append((str(column), value))

    if available:
        with st.expander("Available prototype statistics"):
            stat_cols = st.columns(min(3, len(available)))

            for idx, (label, value) in enumerate(available):
                with stat_cols[idx % len(stat_cols)]:
                    pretty_label = str(label).replace("_", " ").title()

                    if isinstance(value, (float, np.floating)):
                        display_value = f"{float(value):.4f}"
                    else:
                        display_value = str(value)

                    st.markdown(
                        f"""
                        <div class="meta-label">{pretty_label}</div>
                        <div class="meta-value">{display_value}</div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# APP START
# ============================================================

inject_custom_css()
ensure_session_defaults()

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error("Retrieval artifacts could not be loaded.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

visual_embeddings = artifacts["visual_embeddings"]
valid_indices = artifacts["valid_indices"]
cluster_labels = artifacts["cluster_labels"]
prototype_summary = artifacts["prototype_summary"]
prototype_images = artifacts["prototype_images"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="research-header">
        <div class="research-kicker">Medical AI Research Prototype</div>
        <div class="research-title">Explainable Medical VLM</div>
        <div class="research-subtitle">
            MedGemma · Visual Retrieval · Prototype Reasoning
        </div>
        <div class="research-description">
            An image-grounded research interface combining medical VLM analysis
            with visual prototype retrieval for transparent case comparison.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 01 — MEDICAL IMAGE
# ============================================================

render_section(
    "01",
    "Medical Image",
    "Upload an image, inspect its metadata, and start image-grounded analysis.",
)

uploaded_file = st.file_uploader(
    "Upload medical image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-state">
            Upload a PNG or JPEG medical image to begin.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="footer">Research prototype · Not for clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


current_hash = get_image_hash(uploaded_file)

if st.session_state["image_hash"] != current_hash:
    st.session_state["image_hash"] = current_hash
    reset_image_specific_state()

image = Image.open(uploaded_file).convert("RGB")

image_col, control_col = st.columns([1.55, 1], gap="large")

with image_col:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="surface">', unsafe_allow_html=True)

    st.markdown(
        '<span class="status-pill">IMAGE READY</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    meta_items = [
        ("Resolution", f"{image.width} × {image.height} px"),
        ("Embedding database", f"{len(visual_embeddings):,} cases"),
        ("Embedding dimension", f"{visual_embeddings.shape[1]:,}"),
        ("Input format", uploaded_file.type or "image"),
    ]

    for label, value in meta_items:
        st.markdown(
            f"""
            <div class="meta-label">{label}</div>
            <div class="meta-value">{value}</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    analysis_question = st.text_input(
        "Analysis question",
        value="What findings are visible in this image?",
        label_visibility="collapsed",
        key="analysis_question",
    )

    st.markdown('<div class="primary-button">', unsafe_allow_html=True)
    analyze_clicked = st.button(
        "Analyze Image",
        use_container_width=True,
        key="analyze_btn",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked:
        if not analysis_question.strip():
            st.session_state["last_error"] = {
                "message": "Please enter a question before running the analysis.",
                "technical": None,
            }
        else:
            with st.spinner("Running MedGemma analysis..."):
                try:
                    contextual_question = build_question_with_context(
                        analysis_question.strip(),
                        st.session_state["chat_history"],
                    )

                    answer = call_analyze_image(
                        image,
                        contextual_question,
                    )

                    st.session_state["analysis_result"] = answer

                    st.session_state["chat_history"].append(
                        {
                            "role": "user",
                            "content": analysis_question.strip(),
                        }
                    )
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    st.session_state["last_error"] = None

                except BackendError as exc:
                    st.session_state["last_error"] = {
                        "message": exc.message,
                        "technical": exc.technical_detail,
                    }

    st.markdown("</div>", unsafe_allow_html=True)

render_error()


# ============================================================
# 02 — MODEL ANALYSIS
# ============================================================

render_section(
    "02",
    "Model Analysis",
    "The current MedGemma response for the selected image.",
)

render_analysis(st.session_state["analysis_result"])


# ============================================================
# 03 — SIMILAR CASES
# ============================================================

render_section(
    "03",
    "Similar Cases",
    "Top visual matches from the prototype retrieval system. Similarity is an embedding-space measure, not a clinical probability.",
)

retrieval_results = st.session_state["retrieval_results"]

if retrieval_results:
    gallery_items = get_similar_case_images(
        prototype_images,
        retrieval_results,
    )

    if gallery_items:
        gallery_cols = st.columns(len(gallery_items), gap="medium")

        for col, (result, row, image_path) in zip(gallery_cols, gallery_items):
            with col:
                st.markdown('<div class="case-card">', unsafe_allow_html=True)

                case_image = Image.open(image_path).convert("RGB")
                st.image(case_image, use_container_width=True)

                similarity = float(result["similarity"])
                prototype_id = int(result["prototype_id"])
                rank = int(result.get("rank", 0))

                st.markdown(
                    f"""
                    <div class="case-meta">
                        <div class="case-rank">Rank {rank:02d}</div>
                        <div class="case-score">Similarity {similarity * 100:.1f}%</div>
                        <div class="case-prototype">Prototype P{prototype_id:02d}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="empty-state">Retrieved cases were found, but no representative images are available for display.</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="empty-state">Run visual retrieval to compare this image with visually similar cases.</div>',
        unsafe_allow_html=True,
    )

if retrieval_results:
    st.markdown(
        """
        <div class="similarity-note">
            Similarity represents closeness between visual embeddings in the retrieval
            space. It should not be interpreted as diagnostic confidence, disease
            probability, or clinical certainty.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 04 — PROTOTYPE EXPLANATION
# ============================================================

render_section(
    "04",
    "Prototype Explanation",
    "Interpret the recommended prototype through the retrieval structure and existing artifact statistics.",
)

recommended_prototype = st.session_state["recommended_prototype"]

if recommended_prototype is not None and retrieval_results:
    prototype_col, stats_col = st.columns([0.85, 1.4], gap="large")

    with prototype_col:
        st.markdown('<div class="surface">', unsafe_allow_html=True)

        st.markdown(
            '<div class="meta-label">Recommended prototype</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:2rem;font-weight:700;color:#17202a;">P{int(recommended_prototype):02d}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="meta-label" style="margin-top:0.8rem;">
                Retrieval relationship
            </div>
            <div style="font-size:0.84rem;line-height:1.55;color:#667382;">
                This prototype is the cluster associated with the highest-ranked
                visual retrieval result for the current image.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with stats_col:
        st.markdown('<div class="surface">', unsafe_allow_html=True)

        render_prototype_statistics(
            prototype_summary,
            int(recommended_prototype),
        )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="empty-state">Prototype information becomes available after visual retrieval.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 05 — EXPLAINABILITY
# ============================================================

render_section(
    "05",
    "Explainability",
    "Retrieval-based explanation of how the system relates the current image to its visual prototype space.",
)

if recommended_prototype is not None and retrieval_results:
    top_result = retrieval_results[0]
    top_similarity = float(top_result["similarity"])

    e1, e2, e3 = st.columns(3, gap="large")

    with e1:
        st.markdown('<div class="surface-tight">', unsafe_allow_html=True)
        st.markdown(
            '<div class="meta-label">01 · Visual representation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="font-size:0.86rem;line-height:1.55;">
                The uploaded image is converted into a visual embedding by the
                existing MedGemma backend.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with e2:
        st.markdown('<div class="surface-tight">', unsafe_allow_html=True)
        st.markdown(
            '<div class="meta-label">02 · Prototype relationship</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size:0.86rem;line-height:1.55;">
                The embedding is compared with the existing case database and
                associated with prototype <strong>P{int(recommended_prototype):02d}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with e3:
        st.markdown('<div class="surface-tight">', unsafe_allow_html=True)
        st.markdown(
            '<div class="meta-label">03 · Retrieval evidence</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size:0.86rem;line-height:1.55;">
                The highest-ranked retrieved case has a visual similarity score
                of <strong>{top_similarity * 100:.1f}%</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="surface">
            <div class="analysis-label">Interpretation boundary</div>
            <div style="font-size:0.86rem;line-height:1.65;color:#667382;">
                This implementation provides prototype- and retrieval-based
                explainability. It does not currently generate segmentation masks,
                lesion localization maps, or highlighted affected regions.
                Those outputs can be incorporated later without changing the
                current retrieval explanation layer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        '<div class="empty-state">Run visual retrieval to generate prototype-based explainability.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 06 — FOLLOW-UP CONVERSATION
# ============================================================

render_section(
    "06",
    "Ask About This Image",
    "Continue the image-grounded conversation with MedGemma using recent conversation context.",
)

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

follow_up = st.chat_input(
    "Ask a follow-up question about this image"
)

if follow_up and follow_up.strip():
    follow_up = follow_up.strip()

    previous_history = st.session_state["chat_history"][-(
        MAX_CHAT_CONTEXT_TURNS * 2
    ):]

    contextual_question = build_question_with_context(
        follow_up,
        previous_history,
    )

    # Keep the user message visible while the request is processed.
    st.session_state["chat_history"].append(
        {
            "role": "user",
            "content": follow_up,
        }
    )

    try:
        with st.spinner("Generating response..."):
            answer = call_analyze_image(
                image,
                contextual_question,
            )

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state["analysis_result"] = answer
        st.session_state["last_error"] = None

    except BackendError as exc:
        # Remove only the just-added user message if the backend fails.
        if (
            st.session_state["chat_history"]
            and st.session_state["chat_history"][-1]["role"] == "user"
            and st.session_state["chat_history"][-1]["content"] == follow_up
        ):
            st.session_state["chat_history"].pop()

        st.session_state["last_error"] = {
            "message": exc.message,
            "technical": exc.technical_detail,
        }

    st.rerun()


render_error()


# ============================================================
# SYSTEM STATUS
# ============================================================

with st.expander("System status", expanded=False):
    status_items = [
        ("Retrieval artifacts", True),
        ("Prototype database", True),
        ("MedGemma VQA", True),
        ("Visual retrieval", retrieval_results is not None),
        ("Current image", uploaded_file is not None),
    ]

    status_cols = st.columns(len(status_items))

    for col, (label, is_on) in zip(status_cols, status_items):
        with col:
            symbol = "●" if is_on else "○"
            color = "#287a58" if is_on else "#8b97a5"

            st.markdown(
                f"""
                <div style="font-size:0.78rem;color:#667382;">
                    <span style="color:{color};font-size:0.72rem;">{symbol}</span>
                    &nbsp;{label}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Explainable Medical VLM · Research prototype · Not for clinical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
'''

out = Path("/mnt/data/app_redesigned.py")
out.write_text(app_code, encoding="utf-8")
print(f"Created: {out}")
print(f"Lines: {len(app_code.splitlines())}")
