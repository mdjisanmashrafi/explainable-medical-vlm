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
MAX_CHAT_CONTEXT_TURNS = 3


# ============================================================
# VISUAL DESIGN — CUSTOM CSS
# ============================================================

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #FFFFFF;
            --bg-secondary: #F8F9FA;
            --border-color: #E5E7EB;
            --text-main: #111827;
            --text-muted: #6B7280;
            --accent: #2563EB; /* Restrained professional blue */
            --accent-light: #EFF6FF;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: "Inter", -apple-system, "Segoe UI", sans-serif;
        }

        /* Typography & Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-main) !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }
        
        p, span, label, .stMarkdown {
            color: var(--text-main);
        }

        .app-header {
            padding: 2rem 0 1.5rem 0;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .app-header h1 {
            font-size: 2.25rem;
            margin: 0 0 0.5rem 0;
            font-weight: 700 !important;
        }
        
        .app-header .subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            font-weight: 400;
        }

        .section-heading {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-main);
            margin: 2.5rem 0 1.25rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .section-heading span {
            color: var(--accent);
            margin-right: 0.5rem;
            font-weight: 700;
        }

        /* Cards & Containers */
        .research-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .analysis-result {
            background-color: var(--bg-primary);
            border-left: 4px solid var(--accent);
            border-top: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
            border-radius: 0 8px 8px 0;
            padding: 1.25rem 1.5rem;
            font-size: 1rem;
            line-height: 1.6;
            color: var(--text-main);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-size: 0.85rem !important;
            font-weight: 500;
        }
        [data-testid="stMetricValue"] {
            color: var(--text-main) !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
        }

        /* Image Gallery */
        .gallery-item {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem;
            text-align: center;
        }
        
        .gallery-meta {
            margin-top: 0.75rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .similarity-badge {
            background-color: var(--accent-light);
            color: var(--accent);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
        }

        /* Chat & Inputs */
        [data-testid="stChatMessage"] {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--bg-secondary);
            border: 1px dashed #D1D5DB;
            border-radius: 8px;
        }

        .stButton > button {
            background-color: var(--bg-primary);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
            background-color: var(--accent-light);
        }
        
        .stButton > button[kind="primary"] {
            background-color: var(--accent);
            color: white;
            border: none;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #1D4ED8;
            color: white;
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
        "initial_analysis": None,
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
    st.session_state["initial_analysis"] = None
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


def run_full_pipeline(
    image: Image.Image,
    visual_embeddings: np.ndarray,
    valid_indices: np.ndarray,
    cluster_labels: np.ndarray,
) -> None:
    # 1. MedGemma Initial Analysis
    base_question = "What findings are visible in this image?"
    answer = call_analyze_image(image, base_question)
    st.session_state["initial_analysis"] = answer
    
    # Pre-seed the chat history so follow-ups have context of the first answer
    st.session_state["chat_history"] = [
        {"role": "user", "content": base_question},
        {"role": "assistant", "content": answer}
    ]

    # 2. Visual Embedding & Retrieval
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


# ============================================================
# APP START
# ============================================================

inject_custom_css()
ensure_session_defaults()

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error("Failed to load retrieval artifacts. Please ensure the backend is available.")
    with st.expander("Technical Details"):
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
# SECTION 01 — MEDICAL IMAGE
# ============================================================

st.markdown('<h2 class="section-heading"><span>01</span> Medical Image</h2>', unsafe_allow_html=True)

col_upload, col_meta = st.columns([1.5, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload medical image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

if uploaded_file is None:
    st.info("Upload a medical image to begin the analysis.")
    st.stop()

current_hash = get_image_hash(uploaded_file)
if st.session_state["image_hash"] != current_hash:
    st.session_state["image_hash"] = current_hash
    reset_image_specific_state()

image = Image.open(uploaded_file).convert("RGB")

with col_upload:
    st.image(image, use_container_width=True, caption="Uploaded Image")

with col_meta:
    st.markdown('<div class="research-card">', unsafe_allow_html=True)
    st.metric("Resolution", f"{image.width} × {image.height}")
    st.metric("Case Database", f"{len(visual_embeddings):,} cases")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state["initial_analysis"] is None:
        if st.button("Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and retrieving similar cases..."):
                try:
                    run_full_pipeline(image, visual_embeddings, valid_indices, cluster_labels)
                    st.rerun()
                except BackendError as exc:
                    st.session_state["last_error"] = str(exc)

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])
    st.session_state["last_error"] = None


# Stop execution here if analysis hasn't been run yet
if st.session_state["initial_analysis"] is None:
    st.stop()


# ============================================================
# SECTION 02 — MODEL ANALYSIS
# ============================================================

st.markdown('<h2 class="section-heading"><span>02</span> Model Analysis</h2>', unsafe_allow_html=True)

st.markdown(
    f'<div class="analysis-result">{st.session_state["initial_analysis"]}</div>',
    unsafe_allow_html=True
)


# ============================================================
# SECTION 03 — SIMILAR CASES
# ============================================================

st.markdown('<h2 class="section-heading"><span>03</span> Similar Cases</h2>', unsafe_allow_html=True)

recommended_prototype = st.session_state["recommended_prototype"]
retrieval_results = st.session_state["retrieval_results"]

if retrieval_results and recommended_prototype is not None:
    unique_images = get_unique_representative_images(prototype_images, recommended_prototype, max_images=3)
    
    if not unique_images:
        st.info("No representative images available for this visual cluster.")
    else:
        # Display the primary gallery once
        cols = st.columns(len(unique_images))
        for i, (col, (row, image_path)) in enumerate(zip(cols, unique_images)):
            with col:
                st.markdown('<div class="gallery-item">', unsafe_allow_html=True)
                rep_image = Image.open(image_path).convert("RGB")
                st.image(rep_image, use_container_width=True)
                
                # Retrieve the similarity score for this rank from the retrieval results
                # Assuming retrieval_results is ordered by rank
                sim_score = retrieval_results[i]["similarity"] if i < len(retrieval_results) else retrieval_results[0]["similarity"]
                sim_percentage = f"{sim_score * 100:.1f}%"
                
                st.markdown(
                    f"""
                    <div class="gallery-meta">
                        <span>Rank #{i+1}</span>
                        <span class="similarity-badge">Similarity {sim_percentage}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SECTION 04 — PROTOTYPE EXPLANATION
# ============================================================

st.markdown('<h2 class="section-heading"><span>04</span> Prototype Explanation</h2>', unsafe_allow_html=True)

selected_summary = prototype_summary[
    prototype_summary["prototype_id"] == recommended_prototype
]

if not selected_summary.empty:
    row = selected_summary.iloc[0]
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Assigned Prototype", f"P{recommended_prototype:02d}")
    col_p2.metric("Cluster Size", f"{int(row['num_images'])} verified cases")
    col_p3.metric("Top Similarity", f"{retrieval_results[0]['similarity'] * 100:.1f}%")
else:
    st.info("Prototype statistics are currently unavailable.")


# ============================================================
# SECTION 05 — EXPLAINABILITY
# ============================================================

st.markdown('<h2 class="section-heading"><span>05</span> Explainability</h2>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="research-card">
    <p>This system utilizes <strong>Visual Prototype Retrieval</strong> to ground the Vision-Language Model's (VLM) findings. Rather than relying solely on opaque parametric memory, the query image's visual embedding is compared against a verified clinical database.</p>
    <p>The system identified the uploaded image as belonging to visual cluster <strong>P%02d</strong>. The VLM's analysis is informed by the geometric similarities between your query and the historical cases in this cluster, providing a transparent, retrieval-based explanation for its textual output without requiring manual segmentation maps.</p>
    </div>
    """ % (recommended_prototype if recommended_prototype is not None else 0),
    unsafe_allow_html=True
)


# ============================================================
# SECTION 06 — ASK ABOUT THIS IMAGE
# ============================================================

st.markdown('<h2 class="section-heading"><span>06</span> Ask About This Image</h2>', unsafe_allow_html=True)

# Display chat history (skipping the first Q&A since it's displayed in Section 02)
for i, msg in enumerate(st.session_state["chat_history"]):
    if i < 2:  # Skip the initial analysis turn
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

follow_up = st.chat_input("Ask a follow-up question (e.g., 'What are the main visual characteristics?')")

if follow_up and follow_up.strip():
    follow_up = follow_up.strip()

    # Show the user's new question immediately
    st.session_state["chat_history"].append(
        {"role": "user", "content": follow_up}
    )

    with st.spinner("Thinking..."):
        try:
            previous_history = st.session_state["chat_history"][:-1]
            contextual_question = build_question_with_context(follow_up, previous_history)
            
            answer = call_analyze_image(image, contextual_question)
            
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
            st.session_state["last_error"] = None

        except BackendError as exc:
            st.session_state["chat_history"].pop()
            st.session_state["last_error"] = str(exc)

    st.rerun()

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])
    st.session_state["last_error"] = None
