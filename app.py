import hashlib
import html
import json
import re
from pathlib import Path
from textwrap import dedent

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

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

MAX_SIMILAR_CASES = 3
MAX_PROTOTYPE_IMAGES = 3
MAX_CHAT_TURNS = 8
MAX_VQA_IN_EXPLAINABILITY = 3


# ============================================================
# HTML RENDERING HELPER
# ============================================================

def render_html(content: str) -> None:
    body = dedent(content).strip()
    if not body:
        return
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

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
        box-shadow: 0 18px 50px rgba(0,0,0,0.24);
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
        margin-right: 0.3rem;
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

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

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

    .case-card {
        height: 100%;
        overflow: hidden;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(19,25,33,0.92);
    }

    .case-meta {
        padding: 0.75rem 0.85rem 0.9rem 0.85rem;
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

    .debug-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.55rem;
    }

    .debug-item {
        padding: 0.65rem 0.75rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
    }

    .debug-key {
        color: #778492;
        font-size: 0.62rem;
        font-weight: 750;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .debug-val {
        margin-top: 0.2rem;
        color: #dfe6ec;
        font-size: 0.8rem;
        font-weight: 650;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 15px !important;
        background: rgba(255,255,255,0.022) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

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
    "embedding_row": None,
    "retrieval_results": [],
    "prototype_id": None,
    "prototype_similarity": None,
    "demo_case": None,
    "chat_history": [],
    "error": None,
    "debug": {},
    "similar_image_hashes": [],
    "prototype_images": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value


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

    df = pd.read_csv(DEMO_CASES_PATH)

    for column in ("dataset_index", "embedding_row", "prototype_id"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


try:
    artifacts = get_artifacts()
except Exception as exc:
    st.error(f"Retrieval artifacts unavailable: {exc}")
    st.stop()

demo_df = get_demo_cases()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_data(show_spinner=False)
def file_sha256(path_str: str):
    try:
        return hashlib.sha256(Path(path_str).read_bytes()).hexdigest()
    except Exception:
        return None


def parse_qa(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        return 0.0

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


# ============================================================
# DEMO IMAGE RESOLUTION  (authoritative: dataset_index)
# ============================================================

def _row_get(row, key):
    if row is None:
        return None
    try:
        if isinstance(row, pd.Series):
            if key not in row.index:
                return None
            return row[key]
        return row.get(key)
    except Exception:
        return None


def resolve_demo_image(dataset_index, row=None):
    if dataset_index is not None:
        try:
            idx = int(dataset_index)
            for ext in IMAGE_EXTENSIONS:
                candidate = (
                    DEMO_IMAGES_DIR / f"case_{idx:05d}_train{ext}"
                )
                if candidate.exists():
                    return candidate.resolve()
        except Exception:
            pass

    filename = _row_get(row, "image_filename")
    if filename is not None and not pd.isna(filename):
        candidate = DEMO_IMAGES_DIR / str(filename)
        if candidate.exists():
            return candidate.resolve()

    image_path = _row_get(row, "image_path")
    if image_path is not None and not pd.isna(image_path):
        candidate = ROOT / str(image_path)
        if candidate.exists():
            try:
                candidate.resolve().relative_to(
                    PROTOTYPE_IMAGES_DIR.resolve()
                )
                return None
            except ValueError:
                return candidate.resolve()

    return None


# ============================================================
# DEMO CASE LOOKUP
# ============================================================

def find_demo_case_by_hash(file_hash):
    if demo_df.empty or "image_hash" not in demo_df.columns:
        return None

    matches = demo_df[
        demo_df["image_hash"].astype(str) == str(file_hash)
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def find_case_metadata(dataset_index):
    if demo_df.empty or "dataset_index" not in demo_df.columns:
        return None

    try:
        idx = int(dataset_index)
    except Exception:
        return None

    matches = demo_df[demo_df["dataset_index"] == idx]

    if matches.empty:
        return None

    return matches.iloc[0]


# ============================================================
# PROTOTYPE AFFINITY
# ============================================================

def calculate_prototype_affinity(embedding, prototype_id):
    centroids = artifacts.get("cluster_centroids")
    if centroids is None:
        return None

    centroids = np.asarray(centroids)

    try:
        prototype_id = int(prototype_id)
    except Exception:
        return None

    if prototype_id < 0 or prototype_id >= len(centroids):
        return None

    centroid = np.asarray(
        centroids[prototype_id], dtype=np.float32
    ).reshape(-1)

    embedding = np.asarray(
        embedding, dtype=np.float32
    ).reshape(-1)

    if centroid.shape[0] == embedding.shape[0]:
        return cosine_similarity(embedding, centroid)

    pca = artifacts.get("pca_model")
    scaler = artifacts.get("pca_scaler")

    if pca is None or scaler is None:
        return None

    try:
        scaled = scaler.transform(embedding.reshape(1, -1))
        transformed = pca.transform(scaled)[0]
        return cosine_similarity(transformed, centroid)
    except Exception:
        return None


# ============================================================
# PROTOTYPE DESCRIPTION
# ============================================================

def get_prototype_description(prototype_id):
    summary = artifacts.get("prototype_summary")

    if summary is None or summary.empty:
        return "Learned visual prototype."

    if "prototype_id" not in summary.columns:
        return "Learned visual prototype."

    matches = summary[
        summary["prototype_id"].astype(str) == str(prototype_id)
    ]

    if matches.empty:
        return "Learned visual prototype."

    row = matches.iloc[0]

    for column in (
        "description",
        "prototype_description",
        "summary",
        "interpretation",
    ):
        if column not in row.index:
            continue
        value = row[column]
        if pd.notna(value) and str(value).strip():
            return str(value)

    return "Learned visual prototype."


# ============================================================
# PROTOTYPE REPRESENTATIVE IMAGES
# ============================================================

def get_unique_prototype_images(prototype_id, exclude_hashes=None):
    candidates = []

    prototype_df = artifacts.get("prototype_images")

    if (
        prototype_df is not None
        and not prototype_df.empty
        and "prototype_id" in prototype_df.columns
    ):
        matches = prototype_df[
            prototype_df["prototype_id"].astype(str)
            == str(prototype_id)
        ]

        for _, row in matches.iterrows():
            found = False

            for column in (
                "image_path",
                "representative_image",
                "image",
                "filename",
            ):
                if column not in row.index:
                    continue

                value = row[column]
                if pd.isna(value):
                    continue

                value = str(value).strip()
                if not value:
                    continue

                for candidate in (
                    ROOT / value,
                    PROTOTYPE_IMAGES_DIR / value,
                ):
                    if candidate.exists():
                        candidates.append(candidate)
                        found = True
                        break

                if found:
                    break

    if not candidates:
        try:
            folder = (
                PROTOTYPE_IMAGES_DIR / f"P{int(prototype_id):02d}"
            )
        except Exception:
            folder = None

        if folder is not None and folder.exists():
            candidates = sorted(
                p
                for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )

    unique = []
    seen_paths = set()
    seen_hashes = set()

    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            continue

        key = str(resolved)
        if key in seen_paths:
            continue

        content_hash = file_sha256(key)
        if content_hash is None:
            continue

        if content_hash in seen_hashes:
            continue

        seen_paths.add(key)
        seen_hashes.add(content_hash)
        unique.append((resolved, content_hash))

    if exclude_hashes:
        preferred = [t for t in unique if t[1] not in exclude_hashes]
        overlapping = [t for t in unique if t[1] in exclude_hashes]
        unique = preferred + overlapping

    return [path for path, _ in unique[:MAX_PROTOTYPE_IMAGES]]


# ============================================================
# SIMILAR CASE RETRIEVAL
# ============================================================

def retrieve_similar_cases(
    query_embedding,
    exclude_dataset_index=None,
    exclude_hashes=None,
    top_k=MAX_SIMILAR_CASES,
):
    debug = {
        "db_shape": None,
        "candidates_scanned": 0,
        "exact_embedding_duplicates": 0,
    }

    embeddings = np.asarray(
        artifacts["visual_embeddings"], dtype=np.float32
    )
    valid_indices = np.asarray(artifacts["valid_indices"])
    cluster_labels = np.asarray(artifacts["cluster_labels"])

    query_embedding = np.asarray(
        query_embedding, dtype=np.float32
    ).reshape(-1)

    if embeddings.ndim != 2:
        return [], debug

    debug["db_shape"] = tuple(embeddings.shape)

    if embeddings.shape[1] != query_embedding.shape[0]:
        return [], debug

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return [], debug

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    denominator = embedding_norms * query_norm
    denominator[denominator == 0] = 1e-12

    scores = np.dot(embeddings, query_embedding) / denominator
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    seen_dataset_indices = set()
    seen_image_paths = set()
    seen_image_hashes = set(exclude_hashes or set())

    for row_index in ranked_indices:
        debug["candidates_scanned"] += 1
        dataset_index = int(valid_indices[row_index])

        if (
            exclude_dataset_index is not None
            and dataset_index == int(exclude_dataset_index)
        ):
            continue

        if dataset_index in seen_dataset_indices:
            continue

        metadata = find_case_metadata(dataset_index)
        actual_image = resolve_demo_image(dataset_index, metadata)

        if actual_image is None:
            continue

        image_key = str(actual_image)
        if image_key in seen_image_paths:
            continue

        content_hash = file_sha256(image_key)
        if content_hash is None:
            continue

        if content_hash in seen_image_hashes:
            continue

        if np.array_equal(embeddings[row_index], query_embedding):
            debug["exact_embedding_duplicates"] += 1

        seen_dataset_indices.add(dataset_index)
        seen_image_paths.add(image_key)
        seen_image_hashes.add(content_hash)

        try:
            prototype_id = int(cluster_labels[row_index])
        except Exception:
            prototype_id = -1

        results.append(
            {
                "rank": len(results) + 1,
                "embedding_row": int(row_index),
                "dataset_index": dataset_index,
                "prototype_id": prototype_id,
                "similarity": float(scores[row_index]),
                "image_path": actual_image,
                "image_filename": actual_image.name,
                "image_hash": content_hash,
            }
        )

        if len(results) >= top_k:
            break

    debug["unique_retrieved"] = len(results)
    debug["requested"] = top_k

    return results, debug


# ============================================================
# RESET
# ============================================================

def reset_analysis():
    st.session_state.analysis = None
    st.session_state.embedding = None
    st.session_state.embedding_row = None
    st.session_state.retrieval_results = []
    st.session_state.prototype_id = None
    st.session_state.prototype_similarity = None
    st.session_state.demo_case = None
    st.session_state.chat_history = []
    st.session_state.error = None
    st.session_state.debug = {}
    st.session_state.similar_image_hashes = []
    st.session_state.prototype_images = []


# ============================================================
# PRECOMPUTED PIPELINE
# ============================================================

def run_precomputed_case(demo_case, query_image_hash=None):
    if demo_case is None:
        return False

    try:
        dataset_index = int(demo_case["dataset_index"])
        embedding_row = int(demo_case["embedding_row"])

        embeddings = np.asarray(
            artifacts["visual_embeddings"], dtype=np.float32
        )

        if embedding_row < 0 or embedding_row >= embeddings.shape[0]:
            st.session_state.error = (
                f"Embedding row {embedding_row} is out of range."
            )
            return False

        embedding = np.asarray(
            embeddings[embedding_row], dtype=np.float32
        )

        answer = (
            demo_case.get("answer")
            or demo_case.get("analysis")
            or demo_case.get("initial_analysis")
            or "No precomputed observation is available for this case."
        )

        try:
            prototype_id = int(demo_case["prototype_id"])
        except Exception:
            prototype_id = None

        prototype_similarity = (
            calculate_prototype_affinity(embedding, prototype_id)
            if prototype_id is not None
            else None
        )

        exclude_hashes = set()
        if query_image_hash:
            exclude_hashes.add(str(query_image_hash))

        similar_cases, debug = retrieve_similar_cases(
            query_embedding=embedding,
            exclude_dataset_index=dataset_index,
            exclude_hashes=exclude_hashes,
            top_k=MAX_SIMILAR_CASES,
        )

        similar_hashes = {
            r["image_hash"]
            for r in similar_cases
            if r.get("image_hash")
        }

        prototype_images = (
            get_unique_prototype_images(
                prototype_id, exclude_hashes=similar_hashes
            )
            if prototype_id is not None
            else []
        )

        st.session_state.analysis = str(answer)
        st.session_state.embedding = embedding
        st.session_state.embedding_row = embedding_row
        st.session_state.prototype_id = prototype_id
        st.session_state.prototype_similarity = prototype_similarity
        st.session_state.retrieval_results = similar_cases
        st.session_state.demo_case = demo_case
        st.session_state.similar_image_hashes = list(similar_hashes)
        st.session_state.prototype_images = prototype_images

        st.session_state.debug = {
            "db_shape": debug.get("db_shape"),
            "embedding_row": embedding_row,
            "prototype_id": prototype_id,
            "top_similarity": (
                float(similar_cases[0]["similarity"])
                if similar_cases
                else None
            ),
            "unique_retrieved": debug.get("unique_retrieved", 0),
            "requested": debug.get("requested", MAX_SIMILAR_CASES),
            "exact_embedding_duplicates": debug.get(
                "exact_embedding_duplicates", 0
            ),
            "prototype_images_found": len(prototype_images),
            "qa_count": len(parse_qa(demo_case.get("qa"))),
        }

        st.session_state.error = None
        return True

    except Exception as exc:
        st.session_state.error = str(exc)
        return False


# ============================================================
# CHAT — CONTEXT + ANSWER LAYER
# ============================================================

def build_case_context():
    """Snapshot of the currently loaded case, used for chat."""
    if not st.session_state.demo_case:
        return None

    return {
        "dataset_index": int(st.session_state.demo_case.get("dataset_index", -1)),
        "prototype_id": st.session_state.prototype_id,
        "prototype_affinity": st.session_state.prototype_similarity,
        "analysis": st.session_state.analysis,
        "qa": parse_qa(st.session_state.demo_case.get("qa")),
        "retrieval": list(st.session_state.retrieval_results or []),
        "prototype_description": (
            get_prototype_description(st.session_state.prototype_id)
            if st.session_state.prototype_id is not None
            else None
        ),
    }


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _match_qa(question, qa_dict, threshold=0.4):
    """Return the best-matching QA key for the user question, or None."""
    if not qa_dict:
        return None

    q_tokens = _tokenize(question)
    if not q_tokens:
        return None

    best_key = None
    best_score = 0.0

    for qa_q in qa_dict.keys():
        qa_tokens = _tokenize(qa_q)
        if not qa_tokens:
            continue
        overlap = len(q_tokens & qa_tokens) / max(len(qa_tokens), 1)
        if overlap > best_score:
            best_score = overlap
            best_key = qa_q

    if best_key is not None and best_score >= threshold:
        return best_key

    return None


def _affinity_text(affinity):
    if affinity is None:
        return "not available"
    return f"{float(affinity):.5f}"


def _format_reference_lines(retrieval):
    lines = []
    for r in retrieval:
        lines.append(
            f"- Reference Case {int(r['dataset_index'])} — "
            f"cosine similarity {float(r['similarity']):.5f}, "
            f"assigned prototype P{int(r['prototype_id']):02d}"
        )
    return "\n".join(lines) if lines else "No reference cases were retrieved."


def answer_user_question(question, context):
    """Answer strictly from precomputed case evidence. Never hallucinate."""

    if context is None:
        return (
            "Please upload a prepared demonstration image first. "
            "Once a case is loaded I can answer questions about its "
            "model observation, visual prototype, retrieved reference "
            "cases, and similarity scores."
        )

    q_lower = str(question).lower().strip()
    if not q_lower:
        return (
            "Please type a question about the selected image, its "
            "visual evidence, retrieved reference cases, or learned "
            "prototype."
        )

    # --------------------------------------------------------
    # 1. Precomputed QA (fuzzy token match)
    # --------------------------------------------------------

    matched_key = _match_qa(question, context.get("qa") or {})

    if matched_key is not None:
        return str(context["qa"][matched_key])

    # --------------------------------------------------------
    # 2. Model observation
    # --------------------------------------------------------

    observation_keywords = (
        "what does the model see",
        "what does the model say",
        "what does medgemma say",
        "what does medgemma see",
        "what findings",
        "what abnormalities",
        "what do you see",
        "describe the image",
        "describe this image",
        "model observation",
        "medgemma observation",
    )
    if any(k in q_lower for k in observation_keywords):
        return (
            "Precomputed MedGemma observation for this case:\n\n"
            f"\"{context.get('analysis')}\"\n\n"
            "This is a model-generated observation drawn from the "
            "prepared demonstration artifacts. It is not a clinical "
            "diagnosis."
        )

    # --------------------------------------------------------
    # 3. Prototype association
    # --------------------------------------------------------

    prototype_keywords = (
        "what prototype",
        "which prototype",
        "why p",
        "why is this image associated with p",
        "assigned to p",
        "which cluster",
        "prototype context",
    )
    if any(k in q_lower for k in prototype_keywords):
        pid = context.get("prototype_id")
        if pid is None:
            return "No visual prototype is associated with this case."

        aff_text = _affinity_text(context.get("prototype_affinity"))
        desc = context.get("prototype_description") or (
            "Learned visual prototype."
        )

        return (
            f"The selected image is associated with learned visual "
            f"prototype P{int(pid):02d}. Its prototype affinity is "
            f"{aff_text}.\n\n"
            f"Prototype description: {desc}\n\n"
            "Prototype affinity reflects proximity in the learned "
            "visual embedding space. It is not a clinical confidence "
            "score and it is not a diagnostic probability."
        )

    # --------------------------------------------------------
    # 4. Prototype affinity specifically
    # --------------------------------------------------------

    affinity_keywords = (
        "affinity",
        "prototype score",
        "how close to prototype",
    )
    if any(k in q_lower for k in affinity_keywords):
        aff_text = _affinity_text(context.get("prototype_affinity"))
        return (
            f"Prototype affinity for this case is {aff_text}. "
            "It is the cosine similarity between the selected image's "
            "visual embedding and its assigned prototype centroid. "
            "It represents visual association in the learned "
            "representation space, not clinical certainty."
        )

    # --------------------------------------------------------
    # 5. Similar cases
    # --------------------------------------------------------

    similar_keywords = (
        "similar case",
        "reference case",
        "nearest case",
        "nearest neighbor",
        "which case",
        "why are these cases similar",
        "why is this image similar",
        "what are the retrieved cases",
        "retrieved cases",
    )
    if any(k in q_lower for k in similar_keywords):
        retrieval = context.get("retrieval") or []
        return (
            "These reference cases were retrieved because their "
            "learned visual embeddings are close to the selected "
            "image in the embedding space:\n\n"
            + _format_reference_lines(retrieval)
            + "\n\nSimilarity measures proximity in the learned visual "
            "representation. It does not imply a shared clinical "
            "diagnosis."
        )

    # --------------------------------------------------------
    # 6. Similarity meaning / how retrieval works
    # --------------------------------------------------------

    similarity_keywords = (
        "what does similarity mean",
        "how does similarity work",
        "what is cosine similarity",
        "how is similarity computed",
        "how does retrieval work",
    )
    if any(k in q_lower for k in similarity_keywords):
        return (
            "Cosine similarity measures the angle between two "
            "embedding vectors. Values near 1.0 indicate that the "
            "images occupy nearly the same direction in the learned "
            "visual embedding space. This reflects visual proximity in "
            "the representation, not a clinical equivalence."
        )

    # --------------------------------------------------------
    # 7. Explainability / pipeline
    # --------------------------------------------------------

    explain_keywords = (
        "how does the explanation work",
        "how were these cases selected",
        "what is the pipeline",
        "explainability",
        "how does the explanation pipeline work",
    )
    if any(k in q_lower for k in explain_keywords):
        return (
            "The explanation pipeline works as follows:\n\n"
            "1. The selected image is mapped to a precomputed "
            "1152-dimensional visual embedding.\n"
            "2. Cosine similarity ranks every reference embedding.\n"
            "3. The nearest unique reference cases are retrieved.\n"
            "4. The image is associated with a learned visual "
            "prototype based on its assigned cluster.\n"
            "5. Prototype affinity is computed between the image "
            "embedding and the prototype centroid.\n\n"
            "The MedGemma observation is a separate precomputed "
            "output. Retrieval does not establish that MedGemma relied "
            "on the retrieved prototype when producing its observation."
        )

    # --------------------------------------------------------
    # 8. Evidence inventory
    # --------------------------------------------------------

    evidence_keywords = (
        "what evidence",
        "what is available",
        "what can you tell me",
        "available evidence",
    )
    if any(k in q_lower for k in evidence_keywords):
        retrieval = context.get("retrieval") or []
        qa = context.get("qa") or {}
        pid = context.get("prototype_id")
        aff_text = _affinity_text(context.get("prototype_affinity"))
        proto_text = f"P{int(pid):02d}" if pid is not None else "not available"

        return (
            "Available precomputed evidence for this case:\n\n"
            f"- MedGemma observation: available\n"
            f"- Visual prototype: {proto_text}\n"
            f"- Prototype affinity: {aff_text}\n"
            f"- Retrieved reference cases: {len(retrieval)}\n"
            f"- Precomputed VQA entries: {len(qa)}\n\n"
            "All content is drawn from the prepared demonstration "
            "artifacts. Live MedGemma inference is not enabled in this "
            "public demo."
        )

    # --------------------------------------------------------
    # 9. Fallback — never hallucinate
    # --------------------------------------------------------

    return (
        "I can answer questions supported by the prepared evidence for "
        "this demonstration case, such as:\n\n"
        "- the precomputed MedGemma observation,\n"
        "- the assigned visual prototype and its affinity,\n"
        "- the retrieved reference cases and their similarity scores,\n"
        "- the precomputed VQA question–answer pairs,\n"
        "- how the retrieval and explainability pipeline works.\n\n"
        "This public demo does not run new MedGemma inference for "
        "arbitrary questions."
    )


# ============================================================
# HERO
# ============================================================

render_html(
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
    """
)


# ============================================================
# 01 — MEDICAL IMAGE
# ============================================================

render_html(
    """
    <div class="section-header">

        <div class="section-number">01</div>

        <div>
            <div class="section-title">Medical Image</div>
            <div class="section-description">
                Upload one of the prepared demonstration images.
            </div>
        </div>

    </div>
    """
)


uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)


if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        current_hash = calculate_hash(file_bytes)

        if st.session_state.image_hash != current_hash:
            reset_analysis()
            st.session_state.image_hash = current_hash

            demo_case = find_demo_case_by_hash(current_hash)

            if demo_case is not None:
                success = run_precomputed_case(
                    demo_case, query_image_hash=current_hash
                )
                if success:
                    st.success(
                        "Prepared demonstration case recognized."
                    )
                else:
                    st.error(
                        "Prepared case recognized, but precomputed "
                        "artifacts could not be loaded."
                    )
            else:
                st.session_state.error = (
                    "This image is not currently included in the "
                    "prepared demonstration artifacts. "
                    "Live MedGemma inference is not enabled in "
                    "the public demo."
                )

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.image(image, use_container_width=True)

        with right:
            if st.session_state.demo_case is not None:
                case = st.session_state.demo_case

                render_html(
                    """
                    <div class="soft-card">
                        <div class="status-label">Case status</div>
                        <div class="status-value">
                            ✓ Prepared demonstration case
                        </div>
                    </div>
                    """
                )

                st.write("")

                a, b = st.columns(2)

                with a:
                    render_html(
                        f"""
                        <div class="status-card">
                            <div class="status-label">
                                Dataset index
                            </div>
                            <div class="status-value">
                                {int(case["dataset_index"])}
                            </div>
                        </div>
                        """
                    )

                with b:
                    proto_display = (
                        f"P{int(case['prototype_id']):02d}"
                        if pd.notna(case.get("prototype_id"))
                        else "—"
                    )
                    render_html(
                        f"""
                        <div class="status-card">
                            <div class="status-label">
                                Prototype
                            </div>
                            <div class="status-value">
                                {proto_display}
                            </div>
                        </div>
                        """
                    )

                st.write("")
                st.caption(f"📄 {uploaded_file.name}")
                st.caption(f"📦 {len(file_bytes) / 1024:.1f} KB")

            else:
                render_html(
                    """
                    <div class="empty-state">
                        <div class="empty-icon">🔎</div>
                        <div class="empty-title">
                            Image not in demonstration library
                        </div>
                        <div class="empty-text">
                            This public demo currently supports
                            prepared cases with precomputed
                            results. Live MedGemma inference is
                            not enabled in the public demo.
                        </div>
                    </div>
                    """
                )

    except Exception as exc:
        st.error(f"Unable to process image: {exc}")


# ============================================================
# SYSTEM INFORMATION  (static + optional debug)
# ============================================================

with st.expander("⚙️ System information", expanded=False):

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_html(
            """
            <div class="status-card">
                <div class="status-label">Model</div>
                <div class="status-value">MedGemma 1.5 4B</div>
            </div>
            """
        )

    with c2:
        render_html(
            """
            <div class="status-card">
                <div class="status-label">Embedding</div>
                <div class="status-value">1152-D</div>
            </div>
            """
        )

    with c3:
        render_html(
            """
            <div class="status-card">
                <div class="status-label">Prototypes</div>
                <div class="status-value">
                    30 learned clusters
                </div>
            </div>
            """
        )

    with c4:
        render_html(
            """
            <div class="status-card">
                <div class="status-label">Reference cases</div>
                <div class="status-value">1,793 images</div>
            </div>
            """
        )

    debug = st.session_state.get("debug") or {}

    db_shape = debug.get("db_shape")
    db_shape_text = (
        f"{db_shape[0]} × {db_shape[1]}" if db_shape else "—"
    )

    top_sim = debug.get("top_similarity")
    top_sim_text = (
        f"{float(top_sim):.5f}" if top_sim is not None else "—"
    )

    unique_retrieved = debug.get("unique_retrieved")
    requested = debug.get("requested")
    unique_text = (
        f"{unique_retrieved} / {requested}"
        if unique_retrieved is not None and requested is not None
        else "—"
    )

    proto_id_dbg = debug.get("prototype_id")
    proto_id_dbg_text = (
        f"P{int(proto_id_dbg):02d}"
        if proto_id_dbg is not None
        else "—"
    )

    exact_dupes = debug.get("exact_embedding_duplicates")
    exact_dupes_text = (
        "Yes" if (exact_dupes or 0) > 0
        else ("No" if exact_dupes is not None else "—")
    )

    emb_row_dbg = debug.get("embedding_row")
    emb_row_text = (
        str(int(emb_row_dbg)) if emb_row_dbg is not None else "—"
    )

    proto_imgs_found = debug.get("prototype_images_found")
    proto_imgs_text = (
        str(proto_imgs_found)
        if proto_imgs_found is not None
        else "—"
    )

    qa_count = debug.get("qa_count")
    qa_count_text = (
        str(qa_count) if qa_count is not None else "—"
    )

    dataset_index_dbg = (
        int(st.session_state.demo_case["dataset_index"])
        if st.session_state.demo_case is not None
        else None
    )
    dataset_index_text = (
        str(dataset_index_dbg)
        if dataset_index_dbg is not None
        else "—"
    )

    affinity_dbg = st.session_state.prototype_similarity
    affinity_dbg_text = (
        f"{float(affinity_dbg):.5f}"
        if affinity_dbg is not None
        else "—"
    )

    st.markdown(
        "<div style='height:0.8rem;'></div>",
        unsafe_allow_html=True,
    )

    render_html(
        f"""
        <div class="debug-grid">

            <div class="debug-item">
                <div class="debug-key">Embedding database</div>
                <div class="debug-val">{html.escape(db_shape_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Selected dataset index</div>
                <div class="debug-val">{html.escape(dataset_index_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Selected embedding row</div>
                <div class="debug-val">{html.escape(emb_row_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Selected prototype</div>
                <div class="debug-val">{html.escape(proto_id_dbg_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Prototype affinity</div>
                <div class="debug-val">{html.escape(affinity_dbg_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Top similarity</div>
                <div class="debug-val">{html.escape(top_sim_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Unique images retrieved</div>
                <div class="debug-val">{html.escape(unique_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Exact embedding duplicate</div>
                <div class="debug-val">{html.escape(exact_dupes_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Prototype representatives</div>
                <div class="debug-val">{html.escape(proto_imgs_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Precomputed VQA entries</div>
                <div class="debug-val">{html.escape(qa_count_text)}</div>
            </div>

        </div>
        """
    )

    st.caption(
        "Debug values reflect the currently loaded demonstration case. "
        "Similarity values are computed exactly and are never adjusted "
        "for display."
    )


# ============================================================
# 02 — MODEL ANALYSIS
# ============================================================

render_html(
    """
    <div class="section-header">
        <div class="section-number">02</div>
        <div>
            <div class="section-title">Model Analysis</div>
            <div class="section-description">
                Precomputed MedGemma observation for the selected case.
            </div>
        </div>
    </div>
    """
)


if st.session_state.analysis:
    safe_analysis = html.escape(str(st.session_state.analysis))

    render_html(
        f"""
        <div class="analysis-box">

            <div class="analysis-label">
                MedGemma observation
            </div>

            <div class="analysis-text">
                {safe_analysis}
            </div>

            <div style="margin-top:0.9rem;">
                <span class="badge">Precomputed</span>
                <span class="badge">Image-grounded</span>
                <span class="badge">VQA result</span>
            </div>

        </div>
        """
    )
else:
    render_html(
        """
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <div class="empty-title">
                Waiting for a prepared image
            </div>
            <div class="empty-text">
                Upload a demonstration image to display
                the precomputed MedGemma observation.
            </div>
        </div>
        """
    )


# ============================================================
# 03 — SIMILAR CASES
# ============================================================

render_html(
    """
    <div class="section-header">
        <div class="section-number">03</div>
        <div>
            <div class="section-title">Similar Cases</div>
            <div class="section-description">
                Nearest actual reference cases retrieved from the
                visual embedding database.
            </div>
        </div>
    </div>
    """
)


results = st.session_state.retrieval_results

if results:
    render_html(
        """
        <div class="soft-card" style="margin-bottom:1rem;">
            <span style="color:#8996a3; font-size:0.73rem; line-height:1.5;">
                Ranked by cosine similarity between the selected
                image and the reference-image embeddings.
                The selected case is excluded, and each reference
                image is shown only once (deduplicated by dataset
                index, path, and file content).
            </span>
        </div>
        """
    )

    columns = st.columns(len(results), gap="medium")

    for column, result in zip(columns, results):
        with column:
            image_path = result.get("image_path")

            if image_path is not None and Path(image_path).exists():
                st.image(str(image_path), use_container_width=True)
            else:
                render_html(
                    f"""
                    <div class="empty-state">
                        <div class="empty-icon">🖼️</div>
                        <div class="empty-title">
                            Reference image unavailable
                        </div>
                        <div class="empty-text">
                            Dataset index:
                            {int(result.get("dataset_index", -1))}
                        </div>
                    </div>
                    """
                )

            similarity = float(result["similarity"])
            prototype = int(result["prototype_id"])

            render_html(
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
                        Cosine similarity&nbsp;&nbsp;{similarity:.5f}
                    </div>
                </div>
                """
            )

else:
    render_html(
        """
        <div class="empty-state">
            <div class="empty-icon">🔗</div>
            <div class="empty-title">No retrieved cases yet</div>
            <div class="empty-text">
                Upload a prepared demonstration image to
                retrieve visually similar reference cases.
            </div>
        </div>
        """
    )


# ============================================================
# 04 — PROTOTYPE EVIDENCE
# ============================================================

render_html(
    """
    <div class="section-header">
        <div class="section-number">04</div>
        <div>
            <div class="section-title">Prototype Evidence</div>
            <div class="section-description">
                Representative images belonging to the assigned
                learned visual prototype.
            </div>
        </div>
    </div>
    """
)


prototype_id = st.session_state.prototype_id

if prototype_id is not None:
    description = get_prototype_description(prototype_id)
    score = st.session_state.prototype_similarity

    score_text = (
        f"{float(score):.5f}" if score is not None else "—"
    )
    safe_description = html.escape(str(description))

    render_html(
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
                        Visual Prototype P{int(prototype_id):02d}
                    </div>
                    <div class="prototype-description">
                        {safe_description}
                    </div>
                </div>

                <div style="text-align:right; min-width:110px;">
                    <div class="prototype-score-label">
                        Prototype affinity
                    </div>
                    <div class="prototype-score">
                        {score_text}
                    </div>
                </div>

            </div>

        </div>
        """
    )

    prototype_images = st.session_state.prototype_images or []

    if prototype_images:
        columns = st.columns(len(prototype_images), gap="medium")

        for index, (column, image_path) in enumerate(
            zip(columns, prototype_images), start=1
        ):
            with column:
                st.image(str(image_path), use_container_width=True)
                render_html(
                    f"""
                    <div class="evidence-caption">
                        Representative {index}
                    </div>
                    """
                )
    else:
        render_html(
            """
            <div class="empty-state">
                <div class="empty-icon">🧩</div>
                <div class="empty-title">
                    Representative images unavailable
                </div>
                <div class="empty-text">
                    The prototype was identified, but no unique
                    representative images could be resolved.
                </div>
            </div>
            """
        )

else:
    render_html(
        """
        <div class="empty-state">
            <div class="empty-icon">🧩</div>
            <div class="empty-title">
                Prototype evidence not available yet
            </div>
            <div class="empty-text">
                Upload a prepared demonstration image to
                identify its learned visual prototype.
            </div>
        </div>
        """
    )


# ============================================================
# 05 — EXPLAINABILITY  (CASE-SPECIFIC)
# ============================================================

render_html(
    """
    <div class="section-header">
        <div class="section-number">05</div>
        <div>
            <div class="section-title">Explainability</div>
            <div class="section-description">
                Case-specific summary of the retrieved visual
                evidence and learned prototype for the selected image.
            </div>
        </div>
    </div>
    """
)


case = st.session_state.demo_case
retrieval = st.session_state.retrieval_results or []
proto_id_current = st.session_state.prototype_id
affinity_current = st.session_state.prototype_similarity


if case is None:

    render_html(
        """
        <div class="empty-state">
            <div class="empty-icon">🧪</div>
            <div class="empty-title">
                Case-specific explanation not available yet
            </div>
            <div class="empty-text">
                Upload a prepared demonstration image to see a
                case-specific evidence summary.
            </div>
        </div>
        """
    )

else:

    try:
        dataset_index_val = int(case.get("dataset_index"))
    except Exception:
        dataset_index_val = -1

    proto_text = (
        f"P{int(proto_id_current):02d}"
        if proto_id_current is not None
        else "—"
    )

    affinity_text = (
        f"{float(affinity_current):.5f}"
        if affinity_current is not None
        else "—"
    )

    # --------------------------------------------------------
    # Case summary grid
    # --------------------------------------------------------

    render_html(
        f"""
        <div class="debug-grid">

            <div class="debug-item">
                <div class="debug-key">Selected case</div>
                <div class="debug-val">Dataset {dataset_index_val}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Visual prototype</div>
                <div class="debug-val">{html.escape(proto_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Prototype affinity</div>
                <div class="debug-val">{html.escape(affinity_text)}</div>
            </div>

            <div class="debug-item">
                <div class="debug-key">Retrieved cases</div>
                <div class="debug-val">{len(retrieval)}</div>
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # What the retrieval evidence shows
    # --------------------------------------------------------

    if retrieval:

        sims = [float(r["similarity"]) for r in retrieval]
        sims_text = ", ".join(f"{s:.5f}" for s in sims)

        if proto_id_current is not None and affinity_current is not None:
            evidence_summary = (
                f"The selected image is assigned to visual prototype "
                f"{proto_text} with a prototype affinity of "
                f"{float(affinity_current):.5f}. Its nearest retrieved "
                f"reference cases have cosine similarities of "
                f"{sims_text}, indicating that the selected image lies "
                f"close to these cases in the learned visual embedding "
                f"space."
            )
        else:
            evidence_summary = (
                f"The selected image has {len(retrieval)} retrieved "
                f"reference cases with cosine similarities of "
                f"{sims_text}, indicating proximity in the learned "
                f"visual embedding space."
            )

    else:

        evidence_summary = (
            "No reference cases were retrieved for this image. "
            "The explanation is therefore limited to the precomputed "
            "model observation and any available prototype association."
        )

    render_html(
        f"""
        <div class="card" style="margin-top:1rem;">
            <div class="evidence-title">
                What the retrieval evidence shows
            </div>
            <div class="evidence-text">
                {html.escape(evidence_summary)}
            </div>
        </div>
        """
    )

    # --------------------------------------------------------
    # Why these reference cases?
    # --------------------------------------------------------

    if retrieval:

        ref_lines = []
        for r in retrieval:
            ref_lines.append(
                f"<div style='margin-top:0.4rem;'>"
                f"<span style='color:#c7d0d9; font-weight:650;'>"
                f"Reference Case {int(r['dataset_index'])}"
                f"</span>"
                f"<br>"
                f"<span style='color:#8996a3; font-size:0.75rem;'>"
                f"Cosine similarity: "
                f"{float(r['similarity']):.5f} &nbsp;·&nbsp; "
                f"Prototype: P{int(r['prototype_id']):02d}"
                f"</span>"
                f"</div>"
            )
        ref_html = "".join(ref_lines)
    else:
        ref_html = (
            "<div style='color:#8996a3; font-size:0.78rem;'>"
            "No reference cases were retrieved for this image."
            "</div>"
        )

    render_html(
        f"""
        <div class="card" style="margin-top:0.8rem;">
            <div class="evidence-title">
                Why these reference cases?
            </div>
            <div class="evidence-text">
                The reference cases are not selected because they
                share a clinical diagnosis. They are retrieved because
                their learned visual embeddings are close to the
                selected image in the embedding space.
            </div>
            <div style="margin-top:0.7rem;">
                {ref_html}
            </div>
        </div>
        """
    )

    # --------------------------------------------------------
    # Prototype context
    # --------------------------------------------------------

    if proto_id_current is not None and affinity_current is not None:
        proto_context_text = (
            f"The selected image is associated with learned visual "
            f"prototype {proto_text}. Its prototype affinity is "
            f"{float(affinity_current):.5f}.\n\n"
            "This reflects proximity to the learned visual cluster "
            "represented by this prototype. It should not be "
            "interpreted as clinical confidence or diagnostic "
            "certainty."
        )
    else:
        proto_context_text = (
            "No prototype association is available for this case. "
            "Prototype affinity cannot be reported."
        )

    render_html(
        f"""
        <div class="card" style="margin-top:0.8rem;">
            <div class="evidence-title">
                Prototype context
            </div>
            <div class="evidence-text">
                {html.escape(proto_context_text).replace(chr(10)+chr(10), "<br><br>")}
            </div>
        </div>
        """
    )

    # --------------------------------------------------------
    # Model observation vs retrieval evidence
    # --------------------------------------------------------

    observation_text = (
        st.session_state.analysis
        or "No precomputed observation is available for this case."
    )

    if retrieval:
        ref_indices_text = ", ".join(
            str(int(r["dataset_index"])) for r in retrieval
        )
        proto_ref_text = (
            f"and associates it with prototype {proto_text}"
            if proto_id_current is not None
            else ""
        )
        retrieval_distinction = (
            f"The visual retrieval system independently places this "
            f"image near reference cases {ref_indices_text} "
            f"{proto_ref_text}. Retrieval is a separate information "
            f"source and does not establish that MedGemma relied on "
            f"the retrieved evidence when producing its observation."
        )
    else:
        retrieval_distinction = (
            "No reference cases were retrieved for this image. "
            "Retrieval evidence is therefore not available."
        )

    obs_col, ret_col = st.columns(2, gap="large")

    with obs_col:
        render_html(
            f"""
            <div class="card">
                <div class="evidence-title">Model observation</div>
                <div class="evidence-text"
                     style="font-style:italic; color:#dfe6ec;">
                    "{html.escape(str(observation_text))}"
                </div>
                <div style="margin-top:0.7rem;">
                    <span class="badge">Precomputed</span>
                    <span class="badge">Image-grounded</span>
                </div>
            </div>
            """
        )

    with ret_col:
        render_html(
            f"""
            <div class="card">
                <div class="evidence-title">Retrieval evidence</div>
                <div class="evidence-text">
                    {html.escape(retrieval_distinction)}
                </div>
                <div style="margin-top:0.7rem;">
                    <span class="badge">Embedding similarity</span>
                    <span class="badge">Learned prototype</span>
                </div>
            </div>
            """
        )

    # --------------------------------------------------------
    # Additional precomputed observations
    # --------------------------------------------------------

    qa = parse_qa(case.get("qa"))

    if qa:

        qa_items = list(qa.items())[:MAX_VQA_IN_EXPLAINABILITY]
        qa_html_parts = []

        for q_text, a_text in qa_items:
            qa_html_parts.append(
                f"<div style='margin-top:0.55rem;'>"
                f"<div class='status-label'>Question</div>"
                f"<div style='margin-top:0.2rem; color:#dfe6ec; "
                f"font-size:0.82rem; font-weight:650;'>"
                f"{html.escape(str(q_text))}"
                f"</div>"
                f"<div class='status-label' style='margin-top:0.5rem;'>"
                f"Precomputed answer</div>"
                f"<div style='margin-top:0.2rem; color:#aab5c0; "
                f"font-size:0.8rem; line-height:1.55;'>"
                f"{html.escape(str(a_text))}"
                f"</div>"
                f"</div>"
            )

        qa_html = "".join(qa_html_parts)

        render_html(
            f"""
            <div class="card" style="margin-top:0.8rem;">
                <div class="evidence-title">
                    Additional precomputed observations
                </div>
                <div class="evidence-text">
                    Question–answer pairs available for this case.
                </div>
                {qa_html}
            </div>
            """
        )

    # --------------------------------------------------------
    # Interpretation boundary
    # --------------------------------------------------------

    render_html(
        """
        <div class="limitation">

            <div class="limitation-title">
                Interpretation boundary
            </div>

            <b style="color:#cfc4a3;">What the system can show:</b>

            <ul style="margin-top:0.35rem; margin-bottom:0.6rem; padding-left:1.1rem;">
                <li>embedding similarity between images</li>
                <li>prototype association and prototype affinity</li>
                <li>retrieved visual neighbours from the reference database</li>
                <li>precomputed MedGemma observations and VQA outputs</li>
            </ul>

            <b style="color:#cfc4a3;">What the system cannot currently prove:</b>

            <ul style="margin-top:0.35rem; padding-left:1.1rem;">
                <li>clinical diagnosis or clinical validity</li>
                <li>causality between retrieved evidence and model output</li>
                <li>model faithfulness or attention to specific evidence</li>
                <li>that MedGemma relied on the retrieved prototype when generating its observation</li>
            </ul>

            <br>

            Prototype affinity represents visual association in the
            learned representation space. Retrieval is a separate
            information source and does not establish that MedGemma
            used the retrieved prototype.

        </div>
        """
    )


# ============================================================
# 06 — ASK ABOUT THIS IMAGE  (CHAT)
# ============================================================

render_html(
    """
    <div class="section-header">
        <div class="section-number">06</div>
        <div>
            <div class="section-title">Ask About This Image</div>
            <div class="section-description">
                Ask questions about the selected image, its visual
                evidence, retrieved cases, or learned prototype.
            </div>
        </div>
    </div>
    """
)


if case is None:

    render_html(
        """
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-title">
                Select a prepared image first
            </div>
            <div class="empty-text">
                The conversational assistant answers questions
                supported by the prepared evidence for the selected
                demonstration case.
            </div>
        </div>
        """
    )

else:

    chat_context = build_case_context()

    # --------------------------------------------------------
    # Quick questions
    # --------------------------------------------------------

    render_html(
        """
        <div class="soft-card" style="margin-bottom:0.75rem;">
            <div class="status-label">Quick questions</div>
            <div style="margin-top:0.2rem; color:#8996a3; font-size:0.72rem;">
                These shortcuts are routed through the same
                evidence-based answer layer as free-form questions.
            </div>
        </div>
        """
    )

    quick_questions = [
        "What does the model see?",
        "Why is this image similar to the retrieved cases?",
        "What is the prototype?",
        "What does prototype affinity mean?",
        "What evidence is available for this case?",
    ]

    qcols = st.columns(len(quick_questions))

    for qcol, qtext in zip(qcols, quick_questions):
        with qcol:
            if st.button(
                qtext,
                key=f"quick_{qtext}_{st.session_state.image_hash or 'none'}",
                use_container_width=True,
            ):
                st.session_state.chat_history.append(
                    {"role": "user", "content": qtext}
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer_user_question(
                            qtext, chat_context
                        ),
                    }
                )
                st.session_state.chat_history = (
                    st.session_state.chat_history[-MAX_CHAT_TURNS:]
                )
                st.rerun()

    # --------------------------------------------------------
    # Render conversation history
    # --------------------------------------------------------

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        render_html(
            """
            <div class="empty-state" style="margin-top:0.4rem;">
                <div class="empty-icon">💬</div>
                <div class="empty-title">
                    No messages yet
                </div>
                <div class="empty-text">
                    Ask a question below, or use one of the quick
                    questions above.
                </div>
            </div>
            """
        )

    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    user_prompt = st.chat_input(
        "Ask about the selected image, its prototype, or its "
        "retrieved reference cases..."
    )

    if user_prompt:

        st.session_state.chat_history.append(
            {"role": "user", "content": user_prompt}
        )

        assistant_reply = answer_user_question(
            user_prompt, chat_context
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_reply}
        )

        st.session_state.chat_history = (
            st.session_state.chat_history[-MAX_CHAT_TURNS:]
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">

        <b style="color:#8995a1;">
            Explainable Medical VLM
        </b>

        <br>

        MedGemma · Visual Retrieval ·
        Prototype-Grounded Evidence

        <br><br>

        Research demonstration only.
        Model outputs are not clinical diagnoses.
        Prototype affinity represents learned visual association
        rather than clinical certainty.

    </div>
    """
)
