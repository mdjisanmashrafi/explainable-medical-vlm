"""
Explainable Medical VLM
MedGemma + Visual Prototype Retrieval — research demo interface.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

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
            --bg-primary: #0A0E17;
            --bg-card: #121826;
            --bg-card-hover: #161E2E;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --text-primary: #E9EDF4;
            --text-secondary: #8A95A8;
            --accent: #4FB8C9;
            --accent-soft: rgba(79, 184, 201, 0.12);
            --success: #5FBF8A;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: "Inter", -apple-system, "Segoe UI", sans-serif;
        }

        [data-testid="stHeader"] {
            background-color: transparent;
        }

        [data-testid="stSidebar"] {
            background-color: var(--bg-card);
            border-right: 1px solid var(--border-subtle);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        h1, h2, h3, h4 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }

        p, span, label, .stMarkdown {
            color: var(--text-primary);
        }

        /* Header */
        .app-header {
            padding: 0 0 1.75rem 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 1.75rem;
        }
        .app-header h1 {
            font-size: 1.7rem;
            margin: 0 0 0.3rem 0;
        }
        .app-header .subtitle {
            color: var(--accent);
            font-size: 0.92rem;
            font-weight: 500;
        }

        /* Section headings */
        .section-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0 0 0.6rem 0;
        }

        /* Cards */
        .med-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
        }

        .answer-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            padding: 1rem 1.2rem;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .answer-card .tag {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            display: block;
            margin-bottom: 0.4rem;
        }

        /* Result row */
        .result-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.65rem 1rem;
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        .result-row .rank {
            color: var(--text-secondary);
            width: 2.4rem;
        }
        .result-row .proto {
            color: var(--accent);
            font-weight: 600;
            width: 4.5rem;
        }
        .result-row .score {
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
        }

        /* Status */
        .status-line {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.88rem;
            padding: 0.3rem 0;
        }
        .dot-on { color: var(--success); }
        .dot-off { color: var(--text-secondary); }

        /* Buttons */
        .stButton > button {
            background-color: var(--accent-soft);
            color: var(--accent);
            border: 1px solid rgba(79, 184, 201, 0.35);
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: background-color 0.15s ease;
        }
        .stButton > button:hover {
            background-color: rgba(79, 184, 201, 0.22);
            color: var(--accent);
            border-color: var(--accent);
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 0.7rem 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
            font-size: 0.75rem !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }

        /* Chat */
        [data-testid="stChatMessage"] {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 8px !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--bg-card);
            border: 1px dashed var(--border-subtle);
            border-radius: 10px;
        }

        hr {
            border-color: var(--border-subtle);
        }

        .footer-note {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.78rem;
            padding-top: 1.5rem;
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
# BACKEND CALLS
# ============================================================

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
        raise BackendError("MedGemma analysis failed.") from exc
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
        return embedding
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
            "Returned embedding has an unexpected dimension "
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
# REPRESENTATIVE IMAGE DEDUPLICATION
# ============================================================

def get_unique_representative_images(
    prototype_images: pd.DataFrame,
    prototype_id: int,
    max_images: int = 3,
):
    subset = prototype_images[
        prototype_images["prototype_id"] == prototype_id
    ].sort_values("rank")

    seen_keys = set()
    unique = []

    for _, row in subset.iterrows():
        rank = int(row["rank"])
        image_path = (
            ROOT
            / "representative_images"
            / f"P{prototype_id:02d}"
            / f"representative_{rank}.png"
        )

        if not image_path.exists():
            continue

        if "dataset_index" in row and pd.notna(row["dataset_index"]):
            key = f"idx:{int(row['dataset_index'])}"
        else:
            key = f"hash:{hashlib.md5(image_path.read_bytes()).hexdigest()}"

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique.append((row, image_path))

        if len(unique) >= max_images:
            break

    return unique


def render_representative_images(prototype_images, prototype_id, key_prefix=""):
    unique_images = get_unique_representative_images(prototype_images, prototype_id)

    if not unique_images:
        st.caption("No representative images available for this prototype.")
        return

    cols = st.columns(len(unique_images))
    for col, (row, image_path) in zip(cols, unique_images):
        with col:
            rep_image = Image.open(image_path).convert("RGB")
            st.image(rep_image, use_container_width=True)
            rank = int(row["rank"])
            caption = f"Representative {rank}"
            if "distance_to_centroid" in row and pd.notna(row["distance_to_centroid"]):
                caption += f" · d={float(row['distance_to_centroid']):.2f}"
            st.caption(caption)


# ============================================================
# APP START
# ============================================================

inject_custom_css()
ensure_session_defaults()

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error("Failed to load retrieval artifacts.")
    with st.expander("Details"):
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
    <div class="app-header">
        <h1>Explainable Medical VLM</h1>
        <div class="subtitle">MedGemma · Visual Retrieval · Prototype Reasoning</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown('<div class="section-label">Medical Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload medical image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        '<div class="med-card" style="text-align:center; color:var(--text-secondary);">'
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
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.metric("Resolution", f"{image.width} × {image.height}")
    st.metric("Case database", f"{len(visual_embeddings):,} cases")
    st.metric("Embedding dimension", f"{visual_embeddings.shape[1]:,}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MEDICAL VLM  +  VISUAL RETRIEVAL  (two columns)
# ============================================================

vlm_col, retrieval_col = st.columns(2)

# ---------------- Medical VLM ----------------
with vlm_col:
    st.markdown('<div class="section-label">Medical VLM</div>', unsafe_allow_html=True)
    st.markdown('<div class="med-card">', unsafe_allow_html=True)

    question = st.text_input(
        "Question",
        value="What findings are visible in this image?",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("Analyze", use_container_width=True, key="analyze_btn")

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
                st.session_state["last_error"] = str(exc)

    last_turn = next(
        (m for m in reversed(st.session_state["chat_history"]) if m["role"] == "assistant"),
        None,
    )
    if last_turn:
        st.markdown(
            f'<div class="answer-card"><span class="tag">MEDGEMMA</span>{last_turn["content"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Visual Retrieval ----------------
with retrieval_col:
    st.markdown('<div class="section-label">Visual Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="med-card">', unsafe_allow_html=True)

    recommended_prototype = st.session_state["recommended_prototype"]
    retrieval_results = st.session_state["retrieval_results"]

    proto_display = f"P{recommended_prototype:02d}" if recommended_prototype is not None else "—"
    sim_display = (
        f"{retrieval_results[0]['similarity']:.4f}" if retrieval_results else "—"
    )

    m1, m2 = st.columns(2)
    m1.metric("Prototype", proto_display)
    m2.metric("Similarity", sim_display)

    retrieve_clicked = st.button(
        "Find Similar Cases", use_container_width=True, key="retrieve_btn"
    )

    if retrieve_clicked:
        with st.spinner("Searching similar cases..."):
            try:
                run_retrieval(image, visual_embeddings, valid_indices, cluster_labels)
            except BackendError as exc:
                st.session_state["last_error"] = str(exc)

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])
    st.session_state["last_error"] = None


```python
# ============================================================
# FOLLOW-UP CONVERSATION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-label">Conversation</div>',
    unsafe_allow_html=True,
)

# Display existing conversation
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Follow-up question input
follow_up = st.chat_input(
    "Ask a follow-up question about this image"
)

if follow_up and follow_up.strip():

    follow_up = follow_up.strip()

    # Show the user's new question immediately
    st.session_state["chat_history"].append(
        {
            "role": "user",
            "content": follow_up,
        }
    )

    with st.spinner("Thinking..."):

        try:
            # Build context from previous conversation.
            # The current question is NOT added twice.
            previous_history = (
                st.session_state["chat_history"][:-1]
            )

            contextual_question = build_question_with_context(
                follow_up,
                previous_history,
            )

            # Send the CURRENT IMAGE + CURRENT QUESTION
            answer = call_analyze_image(
                image,
                contextual_question,
            )

            # Store MedGemma response
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            # Clear any previous error
            st.session_state["last_error"] = None

        except BackendError as exc:

            # Remove the user question if the backend failed
            st.session_state["chat_history"].pop()

            st.session_state["last_error"] = str(exc)

    # Force Streamlit to redraw the conversation
    st.rerun()


# Display backend error if one occurred
if st.session_state["last_error"]:
    st.error(
        st.session_state["last_error"]
    )
    st.session_state["last_error"] = None
```



# ============================================================
# TOP SIMILAR CASES
# ============================================================

retrieval_results = st.session_state["retrieval_results"]
recommended_prototype = st.session_state["recommended_prototype"]

if retrieval_results:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Top Similar Cases</div>', unsafe_allow_html=True)

    rows_html = ""
    for result in retrieval_results:
        rows_html += (
            '<div class="result-row">'
            f'<span class="rank">#{result["rank"]}</span>'
            f'<span class="proto">P{result["prototype_id"]:02d}</span>'
            f'<span class="score">{result["similarity"]:.4f}</span>'
            "</div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)


# ============================================================
# PROTOTYPE EXPLORER
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Prototype Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="med-card">', unsafe_allow_html=True)

prototype_options = sorted(prototype_summary["prototype_id"].unique())

if recommended_prototype is not None and recommended_prototype in prototype_options:
    default_index = prototype_options.index(recommended_prototype)
else:
    default_index = 0

selected_prototype = st.selectbox(
    "Prototype",
    prototype_options,
    index=default_index,
    format_func=lambda x: f"P{x:02d}",
    label_visibility="collapsed",
)

selected_summary = prototype_summary[
    prototype_summary["prototype_id"] == selected_prototype
]

if not selected_summary.empty:
    row = selected_summary.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Prototype", f"P{selected_prototype:02d}")
    c2.metric("Cases", int(row["num_images"]))
    if retrieval_results and selected_prototype == recommended_prototype:
        c3.metric("Similarity", f"{retrieval_results[0]['similarity']:.4f}")
    else:
        c3.metric("Similarity", "—")

st.markdown("<br>", unsafe_allow_html=True)
render_representative_images(prototype_images, selected_prototype, key_prefix="explorer")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# EXPLAINABILITY
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="med-card">', unsafe_allow_html=True)

if recommended_prototype is not None and retrieval_results:
    c1, c2, c3 = st.columns(3)
    c1.metric("Prototype", f"P{recommended_prototype:02d}")
    c2.metric("Similarity", f"{retrieval_results[0]['similarity']:.4f}")
    c3.metric("Retrieved", f"{len(retrieval_results)} cases")

    st.markdown("<br>", unsafe_allow_html=True)
    render_representative_images(prototype_images, recommended_prototype, key_prefix="explain")
else:
    st.caption("Run visual retrieval to generate an image-specific explanation.")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
st.markdown('<div class="med-card">', unsafe_allow_html=True)

status_items = [
    ("Retrieval artifacts", True),
    ("Prototype database", True),
    ("MedGemma VQA", True),
    ("Live retrieval", retrieval_results is not None),
]

status_cols = st.columns(len(status_items))
for col, (label, is_on) in zip(status_cols, status_items):
    dot_class = "dot-on" if is_on else "dot-off"
    symbol = "●" if is_on else "○"
    col.markdown(
        f'<div class="status-line"><span class="{dot_class}">{symbol}</span>{label}</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-note">Research prototype — not for clinical diagnosis.</div>',
    unsafe_allow_html=True,
)
