import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.retrieval import load_retrieval_artifacts


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Medical VLM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DEMO_CASES_PATH = ROOT / "demo_cases.csv"
DEMO_IMAGES_DIR = ROOT / "demo_images"
PROTOTYPE_IMAGES_DIR = ROOT / "representative_images"

MAX_SIMILAR_CASES = 3
MAX_PROTOTYPE_IMAGES = 3
MAX_CHAT_CONTEXT_TURNS = 3


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(55, 75, 100, 0.16),
                transparent 35%
            ),
            #0b0f14;
        color: #e8edf3;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #f3f6f9 !important;
    }

    p {
        color: #b8c2cd;
    }

    /* -------------------------------------------------------
       HERO
    ------------------------------------------------------- */

    .hero {
        padding: 1.8rem 2rem 1.7rem 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(24,31,40,0.96),
                rgba(13,18,25,0.96)
            );
        box-shadow: 0 15px 45px rgba(0,0,0,0.22);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        color: #f5f7fa;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #aeb9c5;
        max-width: 780px;
    }

    .hero-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 1.15rem;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.045);
        color: #cbd5df;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-green {
        border-color: rgba(90, 180, 120, 0.25);
        background: rgba(90, 180, 120, 0.08);
        color: #9ed7b2;
    }

    .badge-blue {
        border-color: rgba(100, 150, 220, 0.25);
        background: rgba(100, 150, 220, 0.08);
        color: #abc8ee;
    }

    /* -------------------------------------------------------
       DISCLAIMER
    ------------------------------------------------------- */

    .disclaimer {
        margin-top: 1rem;
        padding: 0.8rem 1rem;
        border-left: 3px solid rgba(180, 190, 205, 0.55);
        background: rgba(255,255,255,0.025);
        border-radius: 8px;
        color: #98a5b2;
        font-size: 0.78rem;
        line-height: 1.55;
    }

    /* -------------------------------------------------------
       SECTION HEADER
    ------------------------------------------------------- */

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-top: 1.9rem;
        margin-bottom: 0.9rem;
    }

    .section-number {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.09);
        color: #dbe3eb;
        font-size: 0.8rem;
        font-weight: 750;
    }

    .section-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #edf1f5;
    }

    .section-description {
        color: #8f9ba8;
        font-size: 0.82rem;
        margin-top: 0.1rem;
    }

    /* -------------------------------------------------------
       CARDS
    ------------------------------------------------------- */

    .card {
        padding: 1.15rem;
        border: 1px solid rgba(255,255,255,0.075);
        border-radius: 16px;
        background: rgba(20,26,34,0.82);
    }

    .soft-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255,255,255,0.065);
        border-radius: 14px;
        background: rgba(255,255,255,0.025);
    }

    .status-card {
        padding: 0.95rem 1rem;
        border-radius: 13px;
        border: 1px solid rgba(255,255,255,0.075);
        background: rgba(255,255,255,0.025);
    }

    .status-label {
        color: #7f8b98;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .status-value {
        margin-top: 0.25rem;
        color: #e7edf3;
        font-size: 0.92rem;
        font-weight: 650;
    }

    /* -------------------------------------------------------
       ANALYSIS
    ------------------------------------------------------- */

    .analysis-box {
        padding: 1.3rem 1.4rem;
        border-radius: 17px;
        border: 1px solid rgba(110,145,190,0.18);
        background:
            linear-gradient(
                135deg,
                rgba(50,70,95,0.20),
                rgba(20,27,36,0.82)
            );
    }

    .analysis-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8295aa;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }

    .analysis-text {
        color: #e8edf2;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    /* -------------------------------------------------------
       SIMILAR CASES
    ------------------------------------------------------- */

    .case-card {
        overflow: hidden;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.075);
        background: rgba(20,26,34,0.9);
        height: 100%;
    }

    .case-meta {
        padding: 0.8rem 0.9rem 0.9rem 0.9rem;
    }

    .case-rank {
        color: #8e9baa;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .case-title {
        color: #e8edf3;
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .case-subtitle {
        color: #84919e;
        font-size: 0.74rem;
        margin-top: 0.3rem;
    }

    .similarity {
        display: inline-block;
        margin-top: 0.55rem;
        padding: 0.3rem 0.55rem;
        border-radius: 7px;
        background: rgba(255,255,255,0.055);
        color: #b8c8d9;
        font-size: 0.72rem;
        font-weight: 650;
    }

    /* -------------------------------------------------------
       PROTOTYPE
    ------------------------------------------------------- */

    .prototype-main {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background:
            linear-gradient(
                135deg,
                rgba(28,35,44,0.95),
                rgba(17,22,29,0.95)
            );
        margin-bottom: 1rem;
    }

    .prototype-id {
        color: #e8edf3;
        font-size: 1.05rem;
        font-weight: 750;
    }

    .prototype-description {
        color: #9ca8b4;
        margin-top: 0.35rem;
        font-size: 0.84rem;
        line-height: 1.55;
    }

    .prototype-score {
        font-size: 1.55rem;
        font-weight: 750;
        color: #e5edf6;
    }

    .prototype-score-label {
        color: #7f8c99;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .evidence-caption {
        color: #798694;
        font-size: 0.73rem;
        text-align: center;
        margin-top: 0.35rem;
    }

    /* -------------------------------------------------------
       EXPLAINABILITY
    ------------------------------------------------------- */

    .evidence-title {
        color: #edf2f6;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.65rem;
    }

    .evidence-text {
        color: #a9b4bf;
        line-height: 1.65;
        font-size: 0.87rem;
    }

    .step {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(255,255,255,0.065);
        border-radius: 12px;
        background: rgba(255,255,255,0.025);
        margin-bottom: 0.65rem;
    }

    .step-number {
        display: inline-block;
        color: #7e8fa1;
        font-size: 0.7rem;
        font-weight: 750;
        margin-right: 0.5rem;
    }

    .step-title {
        color: #e0e7ed;
        font-weight: 700;
        font-size: 0.84rem;
    }

    .step-description {
        color: #8f9ca8;
        font-size: 0.77rem;
        margin-top: 0.3rem;
        line-height: 1.5;
    }

    .limitation {
        margin-top: 1rem;
        padding: 1rem 1.05rem;
        border-radius: 13px;
        border: 1px solid rgba(200,170,100,0.15);
        background: rgba(150,120,60,0.06);
        color: #9fa8b1;
        font-size: 0.77rem;
        line-height: 1.6;
    }

    .limitation-title {
        color: #c5b58c;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    /* -------------------------------------------------------
       EMPTY STATES
    ------------------------------------------------------- */

    .empty-state {
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px dashed rgba(255,255,255,0.11);
        background: rgba(255,255,255,0.018);
        text-align: center;
    }

    .empty-icon {
        font-size: 1.5rem;
        margin-bottom: 0.4rem;
    }

    .empty-title {
        color: #d9e0e6;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .empty-text {
        color: #7f8b97;
        font-size: 0.77rem;
        margin-top: 0.25rem;
    }

    /* -------------------------------------------------------
       FILE UPLOADER
    ------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        border-radius: 15px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.025);
        border: 1px dashed rgba(255,255,255,0.13);
        border-radius: 15px;
    }

    /* -------------------------------------------------------
       BUTTONS / INPUTS
    ------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.45rem;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    .stTextInput > div > div > input {
        border-radius: 10px;
    }

    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.07);
        color: #66727f;
        font-size: 0.73rem;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "image_hash": None,
    "analysis": None,
    "embedding": None,
    "retrieval_results": [],
    "prototype_id": None,
    "prototype_similarity": None,
    "demo_case": None,
    "chat_history": [],
    "error": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    return load_retrieval_artifacts()


@st.cache_data(show_spinner=False)
def load_demo_cases():
    if not DEMO_CASES_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DEMO_CASES_PATH)

    if "dataset_index" in df.columns:
        df["dataset_index"] = pd.to_numeric(
            df["dataset_index"],
            errors="coerce",
        )

    if "prototype_id" in df.columns:
        df["prototype_id"] = pd.to_numeric(
            df["prototype_id"],
            errors="coerce",
        )

    return df


def image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def safe_cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim != 1:
        a = a.reshape(-1)

    if b.ndim != 1:
        b = b.reshape(-1)

    if a.shape[0] != b.shape[0]:
        return 0.0

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def normalize_question(text):
    if text is None:
        return ""

    text = str(text).strip().lower()

    punctuation = [
        "?",
        ".",
        ",",
        "!",
        ":",
        ";",
    ]

    for char in punctuation:
        text = text.replace(char, "")

    return " ".join(text.split())


def parse_qa(qa_value):
    if qa_value is None:
        return {}

    if isinstance(qa_value, dict):
        return qa_value

    try:
        parsed = json.loads(str(qa_value))

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    return {}


def resolve_demo_image(row):
    """
    Resolve the actual image belonging to a retrieved dataset index.

    IMPORTANT:
    This intentionally does NOT use prototype representative images.
    Similar Cases should show actual retrieved reference cases.
    """

    if row is None:
        return None

    image_path_value = row.get("image_path")

    if pd.notna(image_path_value):
        candidate = ROOT / str(image_path_value)

        if candidate.exists():
            return candidate

    image_filename = row.get("image_filename")

    if pd.notna(image_filename):
        candidate = DEMO_IMAGES_DIR / str(image_filename)

        if candidate.exists():
            return candidate

    return None


def get_prototype_images(prototype_id, artifacts):
    """
    Return representative images ONLY for Prototype Evidence.
    """

    images = []

    prototype_df = artifacts.get("prototype_images")

    if prototype_df is not None and not prototype_df.empty:
        if "prototype_id" in prototype_df.columns:
            matches = prototype_df[
                prototype_df["prototype_id"].astype(str)
                == str(prototype_id)
            ]

            for _, row in matches.iterrows():

                possible_columns = [
                    "image_path",
                    "representative_image",
                    "image",
                    "filename",
                ]

                for column in possible_columns:

                    if column not in row:
                        continue

                    value = row[column]

                    if pd.isna(value):
                        continue

                    candidate = ROOT / str(value)

                    if candidate.exists():
                        images.append(candidate)
                        break

                    candidate = PROTOTYPE_IMAGES_DIR / str(value)

                    if candidate.exists():
                        images.append(candidate)
                        break

    # Fallback to prototype directory
    if not images:
        folder = PROTOTYPE_IMAGES_DIR / f"P{int(prototype_id):02d}"

        if folder.exists():
            images = sorted(
                [
                    path
                    for path in folder.iterdir()
                    if path.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                    }
                ]
            )

    # Remove duplicates while preserving order
    unique = []
    seen = set()

    for path in images:
        key = str(path.resolve())

        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique[:MAX_PROTOTYPE_IMAGES]


def get_prototype_description(prototype_id, artifacts):
    prototype_df = artifacts.get("prototype_summary")

    if prototype_df is None or prototype_df.empty:
        return "Learned visual prototype."

    if "prototype_id" not in prototype_df.columns:
        return "Learned visual prototype."

    matches = prototype_df[
        prototype_df["prototype_id"].astype(str)
        == str(prototype_id)
    ]

    if matches.empty:
        return "Learned visual prototype."

    row = matches.iloc[0]

    preferred_columns = [
        "description",
        "prototype_description",
        "summary",
        "interpretation",
    ]

    for column in preferred_columns:
        if column in row.index:
            value = row[column]

            if pd.notna(value) and str(value).strip():
                return str(value)

    return "Learned visual prototype."


def compute_prototype_similarity(
    embedding,
    prototype_id,
    artifacts,
):
    """
    Supports either:
    1. Original-space centroids, or
    2. PCA-space centroids.
    """

    centroids = artifacts.get("cluster_centroids")

    if centroids is None:
        return None

    centroids = np.asarray(centroids)

    prototype_id = int(prototype_id)

    if prototype_id < 0 or prototype_id >= len(centroids):
        return None

    centroid = centroids[prototype_id]

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    centroid = np.asarray(
        centroid,
        dtype=np.float32,
    ).reshape(-1)

    # Direct original-space comparison
    if centroid.shape[0] == embedding.shape[0]:
        return safe_cosine_similarity(
            embedding,
            centroid,
        )

    # PCA-space comparison
    pca = artifacts.get("pca_model")
    scaler = artifacts.get("pca_scaler")

    if pca is None or scaler is None:
        return None

    try:
        scaled = scaler.transform(
            embedding.reshape(1, -1)
        )

        transformed = pca.transform(scaled)[0]

        return safe_cosine_similarity(
            transformed,
            centroid,
        )

    except Exception:
        return None


def find_top_similar_cases(
    query_embedding,
    artifacts,
    demo_df,
    exclude_dataset_index=None,
    top_k=3,
):
    """
    Retrieve actual reference cases.

    The current query image is excluded from the results so that
    Similar Cases does not simply return the uploaded image itself.
    """

    embeddings = artifacts["visual_embeddings"]
    valid_indices = artifacts["valid_indices"]
    cluster_labels = artifacts["cluster_labels"]

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        return []

    if embeddings.shape[1] != query_embedding.shape[0]:
        return []

    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        return []

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    denominator = (
        embedding_norms * query_norm
    )

    denominator[denominator == 0] = 1e-12

    scores = (
        np.dot(
            embeddings,
            query_embedding,
        )
        / denominator
    )

    order = np.argsort(scores)[::-1]

    results = []
    seen_dataset_indices = set()
    seen_paths = set()

    for row_index in order:

        dataset_index = int(
            valid_indices[row_index]
        )

        # Exclude the query image itself
        if (
            exclude_dataset_index is not None
            and dataset_index
            == int(exclude_dataset_index)
        ):
            continue

        # Deduplicate dataset records
        if dataset_index in seen_dataset_indices:
            continue

        if dataset_index < 0:
            continue

        # Resolve actual case metadata
        case_rows = demo_df[
            demo_df["dataset_index"]
            == dataset_index
        ]

        if case_rows.empty:
            continue

        case_row = case_rows.iloc[0]

        actual_image = resolve_demo_image(
            case_row
        )

        if actual_image is None:
            # Do not substitute prototype images.
            continue

        image_key = str(
            actual_image.resolve()
        )

        if image_key in seen_paths:
            continue

        seen_dataset_indices.add(
            dataset_index
        )

        seen_paths.add(image_key)

        results.append(
            {
                "rank": len(results) + 1,
                "embedding_row": int(row_index),
                "dataset_index": dataset_index,
                "prototype_id": int(
                    cluster_labels[row_index]
                ),
                "similarity": float(
                    scores[row_index]
                ),
                "image_path": actual_image,
                "image_filename": case_row.get(
                    "image_filename",
                    actual_image.name,
                ),
            }
        )

        if len(results) >= top_k:
            break

    return results


def reset_analysis():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = (
            value.copy()
            if isinstance(value, list)
            else value
        )


def match_uploaded_image(
    uploaded_hash,
    demo_df,
):
    if demo_df.empty:
        return None

    if "image_hash" not in demo_df.columns:
        return None

    matches = demo_df[
        demo_df["image_hash"].astype(str)
        == str(uploaded_hash)
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def run_precomputed_demo(
    demo_case,
    artifacts,
    demo_df,
):
    """
    Run the complete precomputed pipeline for a known demo case.
    """

    if demo_case is None:
        return False

    try:
        dataset_index = int(
            demo_case["dataset_index"]
        )

        embedding_row = int(
            demo_case["embedding_row"]
        )

        embedding = np.asarray(
            artifacts["visual_embeddings"][
                embedding_row
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Model observation
        # ----------------------------------------------------

        answer = (
            demo_case.get("answer")
            or demo_case.get("analysis")
            or demo_case.get("initial_analysis")
            or "No precomputed observation is available."
        )

        # ----------------------------------------------------
        # Prototype
        # ----------------------------------------------------

        prototype_id = int(
            demo_case["prototype_id"]
        )

        prototype_similarity = (
            compute_prototype_similarity(
                embedding,
                prototype_id,
                artifacts,
            )
        )

        # ----------------------------------------------------
        # Similar cases
        # ----------------------------------------------------

        similar_cases = find_top_similar_cases(
            query_embedding=embedding,
            artifacts=artifacts,
            demo_df=demo_df,
            exclude_dataset_index=dataset_index,
            top_k=MAX_SIMILAR_CASES,
        )

        st.session_state.analysis = str(
            answer
        )

        st.session_state.embedding = embedding

        st.session_state.prototype_id = (
            prototype_id
        )

        st.session_state.prototype_similarity = (
            prototype_similarity
        )

        st.session_state.retrieval_results = (
            similar_cases
        )

        st.session_state.demo_case = (
            demo_case
        )

        st.session_state.error = None

        return True

    except Exception as exc:
        st.session_state.error = str(exc)
        return False


# ============================================================
# LOAD DATA
# ============================================================

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(
        f"Unable to load retrieval artifacts: {exc}"
    )
    st.stop()

demo_df = load_demo_cases()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🩺 Explainable Medical VLM
        </div>

        <div class="hero-subtitle">
            Prototype-grounded visual evidence for
            interpretable medical image analysis
        </div>

        <div class="hero-row">

            <span class="badge badge-blue">
                MedGemma 1.5 4B
            </span>

            <span class="badge">
                Visual Retrieval
            </span>

            <span class="badge">
                Learned Prototypes
            </span>

            <span class="badge badge-green">
                Offline Demonstration
            </span>

        </div>

        <div class="disclaimer">
            Research demonstration using precomputed model
            outputs and visual retrieval artifacts.
            Outputs are model-generated observations and are
            <b>not clinical diagnoses</b>.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

with st.expander(
    "System information",
    expanded=False,
):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    Model
                </div>
                <div class="status-value">
                    MedGemma 1.5 4B
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    Embedding
                </div>
                <div class="status-value">
                    1152-D visual space
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    Prototypes
                </div>
                <div class="status-value">
                    30 learned clusters
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    Reference cases
                </div>
                <div class="status-value">
                    1,793 images
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 01 — MEDICAL IMAGE
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-number">
            01
        </div>

        <div>
            <div class="section-title">
                Medical Image
            </div>

            <div class="section-description">
                Upload one of the prepared demonstration images.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "Choose a medical image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
    ],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    try:
        image_bytes = uploaded_file.getvalue()

        uploaded_hash = image_hash(
            image_bytes
        )

        # New image
        if (
            st.session_state.image_hash
            != uploaded_hash
        ):
            reset_analysis()

            st.session_state.image_hash = (
                uploaded_hash
            )

            demo_case = match_uploaded_image(
                uploaded_hash,
                demo_df,
            )

            if demo_case is not None:

                st.session_state.demo_case = (
                    demo_case
                )

                success = run_precomputed_demo(
                    demo_case,
                    artifacts,
                    demo_df,
                )

                if success:
                    st.success(
                        "Prepared demonstration case recognized."
                    )

            else:
                st.session_state.error = (
                    "This image is not included in the "
                    "prepared demonstration library."
                )

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        left, right = st.columns(
            [1.05, 1],
            gap="large",
        )

        with left:
            st.image(
                image,
                use_container_width=True,
            )

        with right:

            if st.session_state.demo_case:
                st.markdown(
                    """
                    <div class="soft-card">
                        <div class="status-label">
                            Case status
                        </div>
                        <div class="status-value">
                            Prepared demo case
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                case = st.session_state.demo_case

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(
                        f"""
                        <div class="status-card">
                            <div class="status-label">
                                Dataset index
                            </div>
                            <div class="status-value">
                                {int(case["dataset_index"])}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown(
                        f"""
                        <div class="status-card">
                            <div class="status-label">
                                Prototype
                            </div>
                            <div class="status-value">
                                P{int(case["prototype_id"]):02d}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            else:

                st.markdown(
                    """
                    <div class="empty-state">

                        <div class="empty-icon">
                            🔎
                        </div>

                        <div class="empty-title">
                            Image not in demonstration library
                        </div>

                        <div class="empty-text">
                            The public demo currently supports
                            prepared cases only. Add this image
                            to the precomputed artifact set to
                            analyze it.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error(
            f"Unable to process the uploaded image: {exc}"
        )


# ============================================================
# 02 — MODEL ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-number">
            02
        </div>

        <div>
            <div class="section-title">
                Model Analysis
            </div>

            <div class="section-description">
                Precomputed MedGemma observation for the selected case.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.analysis:

    st.markdown(
        """
        <div class="analysis-box">

            <div class="analysis-label">
                MedGemma observation
            </div>

        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="analysis-text">
            {st.session_state.analysis}
        </div>

        <div style="margin-top: 1rem;">

            <span class="badge">
                Precomputed
            </span>

            <span class="badge">
                Image-grounded
            </span>

            <span class="badge">
                VQA result
            </span>

        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🧠
            </div>

            <div class="empty-title">
                Waiting for a prepared image
            </div>

            <div class="empty-text">
                Upload a demonstration image above to
                display the precomputed MedGemma observation.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 03 — SIMILAR CASES
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-number">
            03
        </div>

        <div>
            <div class="section-title">
                Similar Cases
            </div>

            <div class="section-description">
                Unique reference images retrieved from the visual
                embedding space.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

results = st.session_state.retrieval_results

if results:

    st.markdown(
        """
        <div class="soft-card" style="margin-bottom: 1rem;">
            <span style="color:#8996a4;font-size:0.76rem;">
                Retrieved using cosine similarity in the
                1152-dimensional visual embedding space.
                The uploaded case itself is excluded.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(
        len(results),
        gap="medium",
    )

    for column, result in zip(
        columns,
        results,
    ):

        with column:

            image_path = result.get(
                "image_path"
            )

            if (
                image_path is not None
                and Path(image_path).exists()
            ):

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

            else:

                st.markdown(
                    """
                    <div class="empty-state">
                        <div class="empty-icon">
                            🖼️
                        </div>
                        <div class="empty-title">
                            Reference image unavailable
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            prototype_id = int(
                result["prototype_id"]
            )

            st.markdown(
                f"""
                <div class="case-meta">

                    <div class="case-rank">
                        Rank {result["rank"]}
                    </div>

                    <div class="case-title">
                        Reference Case {result["dataset_index"]}
                    </div>

                    <div class="case-subtitle">
                        Visual prototype P{prototype_id:02d}
                    </div>

                    <div class="similarity">
                        Cosine similarity
                        {result["similarity"]:.3f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

else:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🔗
            </div>

            <div class="empty-title">
                No retrieved cases yet
            </div>

            <div class="empty-text">
                Run the analysis on a prepared demonstration
                image to retrieve visually similar reference cases.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 04 — PROTOTYPE EVIDENCE
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-number">
            04
        </div>

        <div>
            <div class="section-title">
                Prototype Evidence
            </div>

            <div class="section-description">
                Representative images from the highest-ranked
                learned visual prototype.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

prototype_id = st.session_state.prototype_id

if prototype_id is not None:

    prototype_description = (
        get_prototype_description(
            prototype_id,
            artifacts,
        )
    )

    prototype_similarity = (
        st.session_state.prototype_similarity
    )

    score_text = (
        f"{prototype_similarity:.3f}"
        if prototype_similarity is not None
        else "—"
    )

    st.markdown(
        f"""
        <div class="prototype-main">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:1rem;
                flex-wrap:wrap;
            ">

                <div>

                    <div class="prototype-id">
                        Visual Prototype P{int(prototype_id):02d}
                    </div>

                    <div class="prototype-description">
                        {prototype_description}
                    </div>

                </div>

                <div style="text-align:right;">

                    <div class="prototype-score-label">
                        Prototype affinity
                    </div>

                    <div class="prototype-score">
                        {score_text}
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    prototype_images = get_prototype_images(
        prototype_id,
        artifacts,
    )

    if prototype_images:

        columns = st.columns(
            len(prototype_images),
            gap="medium",
        )

        for index, (
            column,
            image_path,
        ) in enumerate(
            zip(
                columns,
                prototype_images,
            ),
            start=1,
        ):

            with column:

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

                st.markdown(
                    f"""
                    <div class="evidence-caption">
                        Prototype representative {index}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">
                    🧩
                </div>

                <div class="empty-title">
                    Prototype representatives unavailable
                </div>

                <div class="empty-text">
                    The assigned prototype was identified,
                    but its representative images could not
                    be resolved.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🧩
            </div>

            <div class="empty-title">
                Prototype evidence not available yet
            </div>

            <div class="empty-text">
                Upload a prepared demonstration image to
                identify its learned visual prototype.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 05 — EXPLAINABILITY
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-number">
            05
        </div>

        <div>
            <div class="section-title">
                Explainability
            </div>

            <div class="section-description">
                How the prototype layer provides visual context
                for the model output.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

explain_left, explain_right = st.columns(
    [1.05, 1],
    gap="large",
)

with explain_left:

    st.markdown(
        """
        <div class="card">

            <div class="evidence-title">
                Prototype-Grounded Visual Evidence
            </div>

            <div class="evidence-text">
                The selected image is represented as a visual
                embedding and compared with a reference medical
                image database. Similar cases are retrieved from
                the embedding space, while the assigned visual
                prototype provides a compact representation of
                the image's learned visual neighborhood.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with explain_right:

    st.markdown(
        """
        <div class="card">

            <div class="step">

                <span class="step-number">
                    01
                </span>

                <span class="step-title">
                    Visual representation
                </span>

                <div class="step-description">
                    The medical image is represented in a
                    1152-dimensional visual embedding space.
                </div>

            </div>

            <div class="step">

                <span class="step-number">
                    02
                </span>

                <span class="step-title">
                    Similarity retrieval
                </span>

                <div class="step-description">
                    Nearby reference images are identified
                    using cosine similarity.
                </div>

            </div>

            <div class="step">

                <span class="step-number">
                    03
                </span>

                <span class="step-title">
                    Prototype context
                </span>

                <div class="step-description">
                    The image is associated with a learned
                    visual prototype representing its local
                    embedding neighborhood.
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="limitation">

        <div class="limitation-title">
            Interpretation boundary
        </div>

        Prototype similarity indicates visual association
        in the learned representation space. It does not
        establish a causal medical relationship, clinical
        validity, or diagnostic certainty.

        <br><br>

        The prototype retrieval process is also separate
        from MedGemma generation. Therefore, retrieval does
        not prove that MedGemma used the retrieved prototype
        when producing its response.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 06 — ASK ABOUT THIS IMAGE
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-number">
            06
        </div>

        <div>
            <div class="section-title">
                Ask About This Image
            </div>

            <div class="section-description">
                Ask a question covered by the precomputed VQA
                results for this case.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.demo_case:

    demo_case = st.session_state.demo_case

    qa_dict = parse_qa(
        demo_case.get("qa")
    )

    if qa_dict:

        question_options = list(
            qa_dict.keys()
        )

        selected_question = st.selectbox(
            "Precomputed questions",
            options=question_options,
            format_func=lambda x: str(x).capitalize(),
        )

        if st.button(
            "Show answer",
            use_container_width=False,
        ):

            answer = qa_dict.get(
                selected_question
            )

            if answer:

                st.session_state.chat_history.append(
                    {
                        "question": selected_question,
                        "answer": str(answer),
                    }
                )

                st.session_state.chat_history = (
                    st.session_state.chat_history[
                        -MAX_CHAT_CONTEXT_TURNS:
                    ]
                )

        if st.session_state.chat_history:

            st.markdown(
                """
                <div style="margin-top:1rem;">
                """,
                unsafe_allow_html=True,
            )

            for turn in reversed(
                st.session_state.chat_history
            ):

                st.markdown(
                    f"""
                    <div class="soft-card"
                         style="margin-bottom:0.7rem;">

                        <div class="status-label">
                            Question
                        </div>

                        <div style="
                            color:#dfe6ec;
                            font-size:0.88rem;
                            font-weight:650;
                            margin-top:0.25rem;
                        ">
                            {turn["question"]}
                        </div>

                        <div class="status-label"
                             style="margin-top:0.8rem;">
                            Precomputed answer
                        </div>

                        <div style="
                            color:#aeb9c4;
                            font-size:0.84rem;
                            line-height:1.6;
                            margin-top:0.25rem;
                        ">
                            {turn["answer"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">
                    💬
                </div>

                <div class="empty-title">
                    No precomputed questions available
                </div>

                <div class="empty-text">
                    This case does not contain a saved VQA
                    question-answer pair.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                💬
            </div>

            <div class="empty-title">
                Select a prepared image first
            </div>

            <div class="empty-text">
                Available precomputed VQA questions will
                appear after a demonstration case is loaded.
            </div>

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

        <b style="color:#8a96a3;">
            Explainable Medical VLM
        </b>
        <br>

        MedGemma · Visual Retrieval · Prototype-Grounded Evidence

        <br><br>

        Research prototype for interpretable medical image
        analysis. Model outputs are not clinical diagnoses.
        Prototype similarity represents learned visual
        association rather than clinical certainty.

    </div>
    """,
    unsafe_allow_html=True,
)
