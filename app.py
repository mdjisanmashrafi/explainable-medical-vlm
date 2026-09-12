import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.retrieval import load_retrieval_artifacts


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Medical VLM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent

DEMO_CASES_PATH = ROOT / "demo_cases.csv"
DEMO_IMAGES_DIR = ROOT / "demo_images"
PROTOTYPE_IMAGES_DIR = ROOT / "representative_images"

MAX_SIMILAR_CASES = 3
MAX_PROTOTYPE_IMAGES = 3
MAX_CHAT_TURNS = 3


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(70, 90, 115, 0.14),
                transparent 34%
            ),
            #0b0f14;
        color: #e8edf3;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5 {
        color: #f1f4f7 !important;
    }

    p {
        color: #aeb8c3;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        padding: 1.8rem 2rem 1.6rem 2rem;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.075);
        background:
            linear-gradient(
                135deg,
                rgba(24,31,40,0.97),
                rgba(13,18,24,0.97)
            );
        box-shadow:
            0 18px 50px rgba(0,0,0,0.24);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 760;
        letter-spacing: -0.04em;
        color: #f4f7fa;
        line-height: 1.15;
    }

    .hero-subtitle {
        max-width: 760px;
        margin-top: 0.55rem;
        color: #aeb8c3;
        font-size: 1rem;
        line-height: 1.55;
    }

    .hero-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.34rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.04);
        color: #c7d0d9;
        font-size: 0.74rem;
        font-weight: 650;
    }

    .badge-blue {
        border-color: rgba(105,150,205,0.25);
        background: rgba(105,150,205,0.08);
        color: #aec8e5;
    }

    .badge-green {
        border-color: rgba(90,180,120,0.23);
        background: rgba(90,180,120,0.075);
        color: #a6d7b6;
    }

    .disclaimer {
        margin-top: 1rem;
        padding: 0.75rem 0.9rem;
        border-left: 3px solid rgba(175,185,198,0.42);
        border-radius: 7px;
        background: rgba(255,255,255,0.025);
        color: #8f9ba7;
        font-size: 0.76rem;
        line-height: 1.55;
    }

    /* ======================================================
       SECTION HEADERS
       ====================================================== */

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: 2rem;
        margin-bottom: 0.85rem;
    }

    .section-number {
        width: 34px;
        height: 34px;
        flex: 0 0 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.065);
        border: 1px solid rgba(255,255,255,0.08);
        color: #d7e0e8;
        font-size: 0.78rem;
        font-weight: 750;
    }

    .section-title {
        color: #edf1f5;
        font-size: 1.08rem;
        font-weight: 720;
    }

    .section-description {
        margin-top: 0.1rem;
        color: #7f8b98;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    /* ======================================================
       CARDS
       ====================================================== */

    .card {
        padding: 1.15rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(19,25,33,0.84);
    }

    .soft-card {
        padding: 0.9rem 1rem;
        border-radius: 13px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.022);
    }

    .status-card {
        padding: 0.85rem 0.95rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.065);
        background: rgba(255,255,255,0.025);
    }

    .status-label {
        color: #778492;
        font-size: 0.66rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .status-value {
        margin-top: 0.25rem;
        color: #e4eaf0;
        font-size: 0.88rem;
        font-weight: 650;
    }

    /* ======================================================
       IMAGE
       ====================================================== */

    .image-frame {
        padding: 0.35rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.075);
        background: rgba(255,255,255,0.025);
    }

    /* ======================================================
       ANALYSIS
       ====================================================== */

    .analysis-box {
        padding: 1.25rem 1.35rem;
        border-radius: 17px;
        border: 1px solid rgba(105,145,190,0.17);
        background:
            linear-gradient(
                135deg,
                rgba(50,72,98,0.18),
                rgba(20,27,35,0.86)
            );
    }

    .analysis-label {
        color: #8094a9;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .analysis-text {
        color: #e9eef3;
        font-size: 1rem;
        line-height: 1.65;
    }

    /* ======================================================
       SIMILAR CASES
       ====================================================== */

    .case-card {
        height: 100%;
        overflow: hidden;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(19,25,33,0.92);
    }

    .case-image-wrapper {
        background: #080b0f;
    }

    .case-meta {
        padding: 0.78rem 0.85rem 0.9rem 0.85rem;
    }

    .case-rank {
        color: #778492;
        font-size: 0.65rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .case-title {
        margin-top: 0.22rem;
        color: #e4eaf0;
        font-size: 0.86rem;
        font-weight: 700;
    }

    .case-subtitle {
        margin-top: 0.22rem;
        color: #7f8b97;
        font-size: 0.72rem;
    }

    .similarity-pill {
        display: inline-block;
        margin-top: 0.55rem;
        padding: 0.27rem 0.5rem;
        border-radius: 7px;
        border: 1px solid rgba(255,255,255,0.065);
        background: rgba(255,255,255,0.045);
        color: #b9c8d7;
        font-size: 0.69rem;
        font-weight: 650;
    }

    /* ======================================================
       PROTOTYPE
       ====================================================== */

    .prototype-card {
        padding: 1.15rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.075);
        background:
            linear-gradient(
                135deg,
                rgba(28,35,44,0.96),
                rgba(17,22,29,0.96)
            );
    }

    .prototype-id {
        color: #e9eef3;
        font-size: 1.03rem;
        font-weight: 750;
    }

    .prototype-description {
        margin-top: 0.3rem;
        color: #929eaa;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    .prototype-score-label {
        color: #778492;
        font-size: 0.64rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .prototype-score {
        margin-top: 0.15rem;
        color: #e6edf4;
        font-size: 1.45rem;
        font-weight: 760;
    }

    .evidence-caption {
        margin-top: 0.35rem;
        color: #6f7c88;
        text-align: center;
        font-size: 0.68rem;
    }

    /* ======================================================
       EXPLAINABILITY
       ====================================================== */

    .evidence-title {
        color: #edf2f6;
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 0.55rem;
    }

    .evidence-text {
        color: #a3aeb9;
        font-size: 0.83rem;
        line-height: 1.65;
    }

    .step {
        padding: 0.82rem 0.9rem;
        margin-bottom: 0.55rem;
        border-radius: 11px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.022);
    }

    .step-number {
        color: #7d8a97;
        font-size: 0.67rem;
        font-weight: 750;
        margin-right: 0.42rem;
    }

    .step-title {
        color: #dde5eb;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .step-description {
        margin-top: 0.28rem;
        color: #858f9a;
        font-size: 0.74rem;
        line-height: 1.5;
    }

    .limitation {
        margin-top: 0.9rem;
        padding: 0.95rem 1rem;
        border-radius: 13px;
        border: 1px solid rgba(190,160,90,0.14);
        background: rgba(150,120,55,0.055);
        color: #969fa8;
        font-size: 0.74rem;
        line-height: 1.6;
    }

    .limitation-title {
        color: #c4b487;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    /* ======================================================
       EMPTY STATES
       ====================================================== */

    .empty-state {
        padding: 1.35rem;
        border-radius: 15px;
        border: 1px dashed rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.017);
        text-align: center;
    }

    .empty-icon {
        font-size: 1.35rem;
        margin-bottom: 0.3rem;
    }

    .empty-title {
        color: #d6dee5;
        font-size: 0.86rem;
        font-weight: 700;
    }

    .empty-text {
        margin-top: 0.25rem;
        color: #77838f;
        font-size: 0.74rem;
        line-height: 1.5;
    }

    /* ======================================================
       INPUTS
       ====================================================== */

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 15px !important;
        background: rgba(255,255,255,0.022) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    .stTextInput > div > div > input {
        border-radius: 10px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.065);
        color: #65717d;
        font-size: 0.7rem;
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
        st.session_state[key] = (
            value.copy()
            if isinstance(value, list)
            else value
        )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_retrieval_artifacts()


@st.cache_data(show_spinner=False)
def get_demo_cases():
    if not DEMO_CASES_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        DEMO_CASES_PATH
    )

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


try:
    artifacts = get_artifacts()
except Exception as exc:
    st.error(
        f"Unable to load retrieval artifacts: {exc}"
    )
    st.stop()

demo_df = get_demo_cases()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_hash(data):
    return hashlib.sha256(data).hexdigest()


def normalize_question(text):
    if text is None:
        return ""

    text = str(text).strip().lower()

    for character in [
        "?",
        ".",
        ",",
        "!",
        ":",
        ";",
    ]:
        text = text.replace(
            character,
            "",
        )

    return " ".join(
        text.split()
    )


def parse_qa(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    try:
        parsed = json.loads(
            str(value)
        )

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    return {}


def cosine_similarity(a, b):
    a = np.asarray(
        a,
        dtype=np.float32,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float32,
    ).reshape(-1)

    if a.shape != b.shape:
        return 0.0

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / (a_norm * b_norm)
    )


def resolve_demo_image(row):
    """
    Resolve the actual image belonging to a dataset case.

    IMPORTANT:
    This function intentionally does NOT fall back to
    representative prototype images.
    """

    if row is None:
        return None

    image_path = row.get(
        "image_path"
    )

    if (
        image_path is not None
        and not pd.isna(image_path)
    ):
        candidate = (
            ROOT / str(image_path)
        )

        if candidate.exists():
            return candidate

    filename = row.get(
        "image_filename"
    )

    if (
        filename is not None
        and not pd.isna(filename)
    ):
        candidate = (
            DEMO_IMAGES_DIR
            / str(filename)
        )

        if candidate.exists():
            return candidate

    return None


def find_demo_case_by_hash(
    file_hash,
):
    if demo_df.empty:
        return None

    if "image_hash" not in demo_df.columns:
        return None

    matches = demo_df[
        demo_df["image_hash"].astype(str)
        == str(file_hash)
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def find_case_metadata(
    dataset_index,
):
    if demo_df.empty:
        return None

    if "dataset_index" not in demo_df.columns:
        return None

    matches = demo_df[
        demo_df["dataset_index"]
        == int(dataset_index)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


# ============================================================
# PROTOTYPE FUNCTIONS
# ============================================================

def calculate_prototype_affinity(
    embedding,
    prototype_id,
):
    centroids = artifacts.get(
        "cluster_centroids"
    )

    if centroids is None:
        return None

    centroids = np.asarray(
        centroids
    )

    prototype_id = int(
        prototype_id
    )

    if (
        prototype_id < 0
        or prototype_id >= len(centroids)
    ):
        return None

    centroid = np.asarray(
        centroids[prototype_id],
        dtype=np.float32,
    ).reshape(-1)

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    # --------------------------------------------------------
    # Original embedding space
    # --------------------------------------------------------

    if (
        centroid.shape[0]
        == embedding.shape[0]
    ):
        return cosine_similarity(
            embedding,
            centroid,
        )

    # --------------------------------------------------------
    # PCA space
    # --------------------------------------------------------

    pca = artifacts.get(
        "pca_model"
    )

    scaler = artifacts.get(
        "pca_scaler"
    )

    if pca is None or scaler is None:
        return None

    try:
        scaled = scaler.transform(
            embedding.reshape(1, -1)
        )

        transformed = pca.transform(
            scaled
        )[0]

        return cosine_similarity(
            transformed,
            centroid,
        )

    except Exception:
        return None


def get_prototype_description(
    prototype_id,
):
    summary = artifacts.get(
        "prototype_summary"
    )

    if summary is None or summary.empty:
        return (
            "Learned visual prototype."
        )

    if "prototype_id" not in summary.columns:
        return (
            "Learned visual prototype."
        )

    matches = summary[
        summary["prototype_id"].astype(str)
        == str(prototype_id)
    ]

    if matches.empty:
        return (
            "Learned visual prototype."
        )

    row = matches.iloc[0]

    for column in [
        "description",
        "prototype_description",
        "summary",
        "interpretation",
    ]:

        if column not in row.index:
            continue

        value = row[column]

        if (
            pd.notna(value)
            and str(value).strip()
        ):
            return str(value)

    return (
        "Learned visual prototype."
    )


def get_prototype_images(
    prototype_id,
):
    """
    Prototype Evidence ONLY.

    These images are never used by Similar Cases.
    """

    results = []

    prototype_df = artifacts.get(
        "prototype_images"
    )

    if (
        prototype_df is not None
        and not prototype_df.empty
        and "prototype_id"
        in prototype_df.columns
    ):

        matches = prototype_df[
            prototype_df["prototype_id"].astype(str)
            == str(prototype_id)
        ]

        for _, row in matches.iterrows():

            for column in [
                "image_path",
                "representative_image",
                "image",
                "filename",
            ]:

                if column not in row.index:
                    continue

                value = row[column]

                if pd.isna(value):
                    continue

                value = str(value)

                candidate_1 = (
                    ROOT / value
                )

                candidate_2 = (
                    PROTOTYPE_IMAGES_DIR
                    / value
                )

                if candidate_1.exists():
                    results.append(
                        candidate_1
                    )
                    break

                if candidate_2.exists():
                    results.append(
                        candidate_2
                    )
                    break

    # Folder fallback
    if not results:

        folder = (
            PROTOTYPE_IMAGES_DIR
            / f"P{int(prototype_id):02d}"
        )

        if folder.exists():

            results = sorted(
                [
                    p
                    for p in folder.iterdir()
                    if p.suffix.lower()
                    in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                    }
                ]
            )

    # Deduplicate
    unique = []
    seen = set()

    for path in results:

        key = str(
            path.resolve()
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(path)

    return unique[
        :MAX_PROTOTYPE_IMAGES
    ]


# ============================================================
# SIMILAR CASE RETRIEVAL
# ============================================================

def retrieve_similar_cases(
    query_embedding,
    exclude_dataset_index=None,
    top_k=MAX_SIMILAR_CASES,
):
    """
    Retrieve ACTUAL reference cases.

    Important protections:
    - excludes the query itself
    - excludes duplicate dataset indices
    - excludes duplicate image paths
    - does not substitute prototype images
    """

    embeddings = np.asarray(
        artifacts["visual_embeddings"],
        dtype=np.float32,
    )

    valid_indices = np.asarray(
        artifacts["valid_indices"]
    )

    cluster_labels = np.asarray(
        artifacts["cluster_labels"]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    if embeddings.ndim != 2:
        return []

    if (
        embeddings.shape[1]
        != query_embedding.shape[0]
    ):
        return []

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:
        return []

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    denominator = (
        embedding_norms
        * query_norm
    )

    denominator[
        denominator == 0
    ] = 1e-12

    scores = (
        np.dot(
            embeddings,
            query_embedding,
        )
        / denominator
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    seen_dataset_indices = set()
    seen_image_paths = set()

    for row_index in ranked_indices:

        dataset_index = int(
            valid_indices[row_index]
        )

        # ----------------------------------------------------
        # Exclude exact query case
        # ----------------------------------------------------

        if (
            exclude_dataset_index
            is not None
            and dataset_index
            == int(exclude_dataset_index)
        ):
            continue

        # ----------------------------------------------------
        # Dataset-level deduplication
        # ----------------------------------------------------

        if (
            dataset_index
            in seen_dataset_indices
        ):
            continue

        metadata = find_case_metadata(
            dataset_index
        )

        if metadata is None:
            continue

        actual_image = (
            resolve_demo_image(
                metadata
            )
        )

        # ----------------------------------------------------
        # Do NOT use prototype image as fallback
        # ----------------------------------------------------

        if actual_image is None:
            continue

        image_key = str(
            actual_image.resolve()
        )

        if (
            image_key
            in seen_image_paths
        ):
            continue

        seen_dataset_indices.add(
            dataset_index
        )

        seen_image_paths.add(
            image_key
        )

        results.append(
            {
                "rank": len(results) + 1,
                "embedding_row": int(
                    row_index
                ),
                "dataset_index": dataset_index,
                "prototype_id": int(
                    cluster_labels[row_index]
                ),
                "similarity": float(
                    scores[row_index]
                ),
                "image_path": actual_image,
                "image_filename": str(
                    metadata.get(
                        "image_filename",
                        actual_image.name,
                    )
                ),
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# RESET
# ============================================================

def reset_analysis():
    st.session_state.analysis = None
    st.session_state.embedding = None
    st.session_state.retrieval_results = []
    st.session_state.prototype_id = None
    st.session_state.prototype_similarity = None
    st.session_state.demo_case = None
    st.session_state.chat_history = []
    st.session_state.error = None


# ============================================================
# RUN PRECOMPUTED PIPELINE
# ============================================================

def run_precomputed_case(
    demo_case,
):
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
        # Precomputed MedGemma observation
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
            calculate_prototype_affinity(
                embedding,
                prototype_id,
            )
        )

        # ----------------------------------------------------
        # Similar cases
        # ----------------------------------------------------

        similar_cases = (
            retrieve_similar_cases(
                query_embedding=embedding,
                exclude_dataset_index=dataset_index,
                top_k=MAX_SIMILAR_CASES,
            )
        )

        st.session_state.analysis = (
            str(answer)
        )

        st.session_state.embedding = (
            embedding
        )

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

        st.session_state.error = str(
            exc
        )

        return False


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

    c1, c2, c3, c4 = st.columns(4)

    with c1:
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

    with c2:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    Embedding
                </div>
                <div class="status-value">
                    1152-D
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
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

    with c4:
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
    "Upload a medical image",
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

        file_bytes = (
            uploaded_file.getvalue()
        )

        current_hash = (
            calculate_hash(
                file_bytes
            )
        )

        # ----------------------------------------------------
        # Only process when image changes
        # ----------------------------------------------------

        if (
            st.session_state.image_hash
            != current_hash
        ):

            reset_analysis()

            st.session_state.image_hash = (
                current_hash
            )

            demo_case = (
                find_demo_case_by_hash(
                    current_hash
                )
            )

            if demo_case is not None:

                st.session_state.demo_case = (
                    demo_case
                )

                success = (
                    run_precomputed_case(
                        demo_case
                    )
                )

                if success:
                    st.success(
                        "Prepared demonstration case recognized."
                    )

            else:

                st.session_state.error = (
                    "This image is not included "
                    "in the prepared demonstration library."
                )

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        left, right = st.columns(
            [1.05, 0.95],
            gap="large",
        )

        with left:

            st.image(
                image,
                use_container_width=True,
            )

        with right:

            if (
                st.session_state.demo_case
                is not None
            ):

                case = (
                    st.session_state.demo_case
                )

                st.markdown(
                    """
                    <div class="soft-card">

                        <div class="status-label">
                            Case status
                        </div>

                        <div class="status-value">
                            ✓ Prepared demonstration case
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                a, b = st.columns(2)

                with a:

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

                with b:

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

                st.write("")

                st.caption(
                    f"File: {uploaded_file.name}"
                )

                st.caption(
                    f"Size: {len(file_bytes) / 1024:.1f} KB"
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
                            This public demo currently supports
                            prepared cases with precomputed results.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as exc:

        st.error(
            f"Unable to process image: {exc}"
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

    safe_analysis = html.escape(
        str(
            st.session_state.analysis
        )
    )

    st.markdown(
        f"""
        <div class="analysis-box">

            <div class="analysis-label">
                MedGemma observation
            </div>

            <div class="analysis-text">
                {safe_analysis}
            </div>

            <div style="margin-top:0.9rem;">

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
                Upload a demonstration image to display
                the precomputed MedGemma observation.
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
                Actual reference images retrieved from the visual
                embedding space.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


results = (
    st.session_state.retrieval_results
)


if results:

    st.markdown(
        """
        <div class="soft-card"
             style="margin-bottom:1rem;">

            <span style="
                color:#8996a3;
                font-size:0.73rem;
                line-height:1.5;
            ">
                Ranked by cosine similarity between the selected
                image and the reference-image embeddings.
                The selected case itself is excluded.
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
                and Path(
                    image_path
                ).exists()
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
                            Image unavailable
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            similarity = float(
                result["similarity"]
            )

            prototype = int(
                result["prototype_id"]
            )

            st.markdown(
                f"""
                <div class="case-meta">

                    <div class="case-rank">
                        Rank {int(result["rank"]):02d}
                    </div>

                    <div class="case-title">
                        Reference Case {int(result["dataset_index"])}
                    </div>

                    <div class="case-subtitle">
                        Assigned prototype P{prototype:02d}
                    </div>

                    <div class="similarity-pill">
                        Cosine similarity&nbsp;&nbsp;
                        {similarity:.5f}
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
                Upload a prepared demonstration image to
                retrieve visually similar reference cases.
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
                Representative images from the assigned learned
                visual prototype.
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


prototype_id = (
    st.session_state.prototype_id
)


if prototype_id is not None:

    description = (
        get_prototype_description(
            prototype_id
        )
    )

    score = (
        st.session_state.prototype_similarity
    )

    score_text = (
        f"{float(score):.5f}"
        if score is not None
        else "—"
    )

    safe_description = (
        html.escape(
            str(description)
        )
    )

    st.markdown(
        f"""
        <div class="prototype-card">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:1rem;
                flex-wrap:wrap;
            ">

                <div>

                    <div class="prototype-id">
                        Visual Prototype
                        P{int(prototype_id):02d}
                    </div>

                    <div class="prototype-description">
                        {safe_description}
                    </div>

                </div>

                <div style="
                    text-align:right;
                    min-width:100px;
                ">

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

    prototype_images = (
        get_prototype_images(
            prototype_id
        )
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
                        Representative {index}
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
                    Representative images unavailable
                </div>

                <div class="empty-text">
                    The prototype was identified, but its
                    representative images could not be resolved.
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
                How the visual evidence layer contextualizes
                the model output.
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns(
    [1.05, 1],
    gap="large",
)


with left:

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
                this embedding space, while the assigned visual
                prototype provides a compact representation of
                the image's learned visual neighborhood.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with right:

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
                    The image is represented as a
                    1152-dimensional visual embedding.
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
                    Visually nearby reference images are
                    ranked using cosine similarity.
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

        Prototype affinity represents visual association
        in the learned representation space. It does not
        establish clinical validity, causality, or
        diagnostic certainty.

        <br><br>

        Prototype retrieval is also separate from MedGemma
        generation. Therefore, retrieval does not establish
        that MedGemma used the retrieved prototype when
        producing its observation.

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
                Explore questions covered by the precomputed
                VQA results for this case.
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


if st.session_state.demo_case:

    qa = parse_qa(
        st.session_state.demo_case.get(
            "qa"
        )
    )

    if qa:

        question_options = list(
            qa.keys()
        )

        selected_question = (
            st.selectbox(
                "Precomputed questions",
                options=question_options,
                format_func=lambda x: (
                    str(x).strip().capitalize()
                ),
            )
        )

        if st.button(
            "Show answer",
            type="primary",
        ):

            answer = qa.get(
                selected_question
            )

            if answer:

                st.session_state.chat_history.append(
                    {
                        "question": str(
                            selected_question
                        ),
                        "answer": str(
                            answer
                        ),
                    }
                )

                st.session_state.chat_history = (
                    st.session_state.chat_history[
                        -MAX_CHAT_TURNS:
                    ]
                )

        if st.session_state.chat_history:

            st.write("")

            for turn in reversed(
                st.session_state.chat_history
            ):

                safe_question = (
                    html.escape(
                        turn["question"]
                    )
                )

                safe_answer = (
                    html.escape(
                        turn["answer"]
                    )
                )

                st.markdown(
                    f"""
                    <div class="soft-card"
                         style="margin-bottom:0.65rem;">

                        <div class="status-label">
                            Question
                        </div>

                        <div style="
                            margin-top:0.25rem;
                            color:#dfe6ec;
                            font-size:0.84rem;
                            font-weight:650;
                        ">
                            {safe_question}
                        </div>

                        <div class="status-label"
                             style="margin-top:0.75rem;">
                            Precomputed answer
                        </div>

                        <div style="
                            margin-top:0.25rem;
                            color:#aab5c0;
                            font-size:0.81rem;
                            line-height:1.6;
                        ">
                            {safe_answer}
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
                    No precomputed questions
                </div>

                <div class="empty-text">
                    This case does not contain saved VQA
                    question-answer pairs.
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
                Precomputed VQA questions will appear
                after a demonstration case is loaded.
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

        <b style="color:#8995a1;">
            Explainable Medical VLM
        </b>

        <br>

        MedGemma · Visual Retrieval ·
        Prototype-Grounded Evidence

        <br><br>

        Research prototype for interpretable medical
        image analysis. Model outputs are not clinical
        diagnoses. Prototype affinity represents learned
        visual association rather than clinical certainty.

    </div>
    """,
    unsafe_allow_html=True,
)
