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
            --accent: #2563EB;
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

    # Note: Fetch slightly more than top 3 to allow safe filtering of exact matches and duplicates
    results = find_similar_embeddings(
        query_embedding_vector,
        visual_embeddings,
        valid_indices,
        cluster_labels,
        top_k=3, 
    )
    st.write("DEBUG RETRIEVAL:", results)

    st.session_state["query_embedding"] = query_embedding_vector
    st.session_state["retrieval_results"] = results
    
    if results:
        st.session_state["recommended_prototype"] = results[0].get("prototype_id")


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

retrieval_results = st.session_state["retrieval_results"]

if retrieval_results:
    top_unique_cases = []
    seen_hashes = set()

    for result in retrieval_results:
        sim = result.get("similarity", 0.0)
        
        # 1. Exclude the query image itself (identifying ~100% exact matches)
        if sim >= 0.999:
            continue
            
        # 2. Resolve the image path safely based on what the backend provides
        img_path = None
        
        # Case A: Backend provides direct path
        if "image_path" in result and Path(result["image_path"]).exists():
            img_path = Path(result["image_path"])
            
        # Case B: Backend provides dataset_index
        elif "dataset_index" in result:
            idx = result["dataset_index"]
            # Try finding it in prototype_images
            match = prototype_images[prototype_images["dataset_index"] == idx]
            if not match.empty:
                row = match.iloc[0]
                p_id = int(row["prototype_id"])
                r_rank = int(row["rank"])
                potential_path = ROOT / "representative_images" / f"P{p_id:02d}" / f"representative_{r_rank}.png"
                if potential_path.exists():
                    img_path = potential_path
            
            # Fallback path if it's stored in a general directory
            if not img_path:
                alt_path = ROOT / "images" / f"{idx}.png"
                if alt_path.exists():
                    img_path = alt_path
                    
        # Case C: Backend only provides prototype_id (fallback to top unique representative image)
        if not img_path and "prototype_id" in result:
            p_id = result["prototype_id"]
            subset = prototype_images[prototype_images["prototype_id"] == p_id].sort_values("rank")
            for _, row in subset.iterrows():
                r_rank = int(row["rank"])
                temp_path = ROOT / "representative_images" / f"P{p_id:02d}" / f"representative_{r_rank}.png"
                if temp_path.exists():
                    # Ensure we haven't already used this fallback image
                    temp_hash = hashlib.md5(temp_path.read_bytes()).hexdigest()
                    if temp_hash not in seen_hashes:
                        img_path = temp_path
                        break
                        
        # 3. If an image was found, check for duplicates and append
        if img_path and img_path.exists():
            file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                result["resolved_image_path"] = img_path
                top_unique_cases.append(result)
                
        # Stop once we have 3 unique cases
        if len(top_unique_cases) >= 3:
            break

    # Ensure final list is explicitly sorted from highest similarity to lowest
    top_unique_cases = sorted(top_unique_cases, key=lambda x: x.get("similarity", 0.0), reverse=True)

    if not top_unique_cases:
        st.info("No unique similar cases found.")
    else:
        cols = st.columns(len(top_unique_cases))
        for i, (col, case) in enumerate(zip(cols, top_unique_cases)):
            with col:
                st.markdown('<div class="gallery-item">', unsafe_allow_html=True)
                rep_image = Image.open(case["resolved_image_path"]).convert("RGB")
                st.image(rep_image, use_container_width=True)
                
                sim_percentage = f"{case.get('similarity', 0.0) * 100:.1f}%"
                
                st.markdown(
                    f"""
                    <div class="gallery-meta">
                        <span>Rank #{i+1}</span>
                        <span class="similarity-badge">Similarity: {sim_percentage}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SECTION 04 — PROTOTYPE EXPLANATION
# ============================================================

st.markdown('<h2 class="section-heading"><span>04</span> Prototype Explanation</h2>', unsafe_allow_html=True)

recommended_prototype = st.session_state.get("recommended_prototype")

if recommended_prototype is not None:
    selected_summary = prototype_summary[
        prototype_summary["prototype_id"] == recommended_prototype
    ]
    
    if not selected_summary.empty:
        row = selected_summary.iloc[0]
        
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Assigned Prototype", f"P{recommended_prototype:02d}")
        col_p2.metric("Cluster Size", f"{int(row['num_images'])} cases")
        
        # Display similarity between the query and the cluster centroid/top case
        top_sim = retrieval_results[0].get("similarity", 0.0) if retrieval_results else 0.0
        col_p3.metric("Prototype Similarity", f"{top_sim * 100:.1f}%")

        st.markdown(
            """
            <div class="research-card" style="margin-top: 1rem;">
            <p>This visual prototype represents a recurring pattern in the model's learned image representation. 
            It highlights structural and textural consistencies across the database, but is <strong>not</strong> a clinically validated diagnosis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Prototype statistics are currently unavailable.")


# ============================================================
# SECTION 05 — EXPLAINABILITY
# ============================================================

st.markdown('<h2 class="section-heading"><span>05</span> Explainability</h2>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="research-card">
    <p>This system utilizes <strong>Visual Prototype Retrieval</strong> to provide transparent context for the Vision-Language Model's (VLM) findings. Rather than relying solely on opaque parametric memory, the query image's visual embedding is compared against a verified clinical database.</p>
    <p>The retrieved prototype provides an external visual evidence layer that contextualizes the VLM's output by showing visually related cases from the reference database.</p>
    <p style="font-size: 0.9em; color: var(--text-muted); margin-top: 1rem;">
    <em>Note: Prototype similarity does NOT prove that MedGemma used that specific prototype when generating its answer, nor does it imply a causal medical relationship. It serves strictly as grounded visual context.</em>
    </p>
    </div>
    """,
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
