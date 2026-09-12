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


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content: str) -> None:
    body = dedent(content).strip()
    if not body:
        return
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def section_header(number: str, title: str, subtitle: str = "") -> None:
    sub = (
        f'<div class="section-desc">{html.escape(subtitle)}</div>'
        if subtitle
        else ""
    )
    render_html(
        f"""
        <div class="section-header">
            <div class="section-number">{html.escape(number)}</div>
            <div>
                <div class="section-title">{html.escape(title)}</div>
                {sub}
            </div>
        </div>
        """
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 8% 0%,
                rgba(70,90,115,0.12), transparent 34%),
            #0b0f14;
        color: #e8edf3;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.6rem;
        padding-bottom: 2.4rem;
    }

    h1, h2, h3, h4, h5 { color: #f1f4f7 !important; }
    p { color: #aeb8c3; }

    /* ---------- HERO ---------- */

    .hero {
        padding: 1.35rem 1.6rem 1.25rem 1.6rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.07);
        background: linear-gradient(135deg,
            rgba(24,31,40,0.96), rgba(13,18,24,0.96));
        box-shadow: 0 12px 40px rgba(0,0,0,0.22);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 760;
        letter-spacing: -0.035em;
        color: #f4f7fa;
        line-height: 1.15;
    }
    .hero-subtitle {
        margin-top: 0.3rem;
        color: #a4aeb9;
        font-size: 0.92rem;
    }
    .hero-row {
        display: flex; flex-wrap: wrap;
        gap: 0.4rem; margin-top: 0.85rem;
    }
    .badge {
        display: inline-flex; align-items: center;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.04);
        color: #c7d0d9;
        font-size: 0.7rem;
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
    .hero-note {
        margin-top: 0.85rem;
        color: #7a8692;
        font-size: 0.72rem;
    }

    /* ---------- SECTIONS ---------- */

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 1.6rem 0 0.7rem 0;
    }
    .section-number {
        width: 30px; height: 30px; flex: 0 0 30px;
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #d7e0e8;
        font-size: 0.72rem;
        font-weight: 750;
    }
    .section-title {
        color: #edf1f5;
        font-size: 1.05rem;
        font-weight: 720;
        letter-spacing: -0.01em;
    }
    .section-desc {
        margin-top: 0.05rem;
        color: #7f8b98;
        font-size: 0.76rem;
    }

    /* ---------- METRIC STRIP ---------- */

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.5rem;
        margin-bottom: 0.9rem;
    }
    .metric {
        padding: 0.6rem 0.75rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.022);
    }
    .metric-label {
        color: #778492;
        font-size: 0.6rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .metric-value {
        margin-top: 0.18rem;
        color: #e4eaf0;
        font-size: 0.98rem;
        font-weight: 650;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .metric-value.ok { color: #a6d7b6; }

    /* ---------- OBSERVATION CARD ---------- */

    .obs-card {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        border: 1px solid rgba(105,145,190,0.18);
        background: linear-gradient(135deg,
            rgba(50,72,98,0.16), rgba(20,27,35,0.86));
    }
    .obs-label {
        color: #8094a9;
        font-size: 0.62rem;
        font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .obs-text {
        color: #e9eef3;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    /* ---------- CASE CARD (minimal) ---------- */

    .case-mini {
        margin-top: 0.5rem;
        padding: 0.5rem 0.65rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
    }
    .case-mini-title {
        color: #dfe6ec;
        font-size: 0.82rem;
        font-weight: 650;
    }
    .case-mini-meta {
        margin-top: 0.12rem;
        color: #8996a3;
        font-size: 0.72rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* ---------- PROTOTYPE HEADER ---------- */

    .proto-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.8rem;
        padding: 0.85rem 1.05rem;
        border-radius: 13px;
        border: 1px solid rgba(255,255,255,0.07);
        background: linear-gradient(135deg,
            rgba(28,35,44,0.95), rgba(17,22,29,0.95));
        margin-bottom: 0.75rem;
    }
    .proto-id {
        color: #e9eef3;
        font-size: 1.05rem;
        font-weight: 750;
        letter-spacing: -0.01em;
    }
    .proto-desc {
        margin-top: 0.15rem;
        color: #929eaa;
        font-size: 0.75rem;
    }
    .proto-aff {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
    }
    .proto-aff-label {
        color: #778492;
        font-size: 0.6rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .proto-aff-val {
        color: #e6edf4;
        font-size: 1.15rem;
        font-weight: 750;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* ---------- NEIGHBOURS TABLE ---------- */

    .neighbors {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.3rem 0.75rem;
        padding: 0.35rem 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.85rem;
    }
    .neighbors .hdr {
        color: #778492;
        font-size: 0.6rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: system-ui, sans-serif;
    }
    .neighbors .cell { color: #dfe6ec; }
    .neighbors .cell.muted { color: #a3aeb9; }

    /* ---------- CAPTION / MUTED ---------- */

    .muted-line {
        color: #7a8692;
        font-size: 0.72rem;
        line-height: 1.45;
        margin-top: 0.4rem;
    }
    .img-caption {
        margin-top: 0.35rem;
        text-align: center;
        color: #6f7c88;
        font-size: 0.68rem;
    }
    .img-placeholder {
        padding: 2.2rem 0.5rem;
        border-radius: 12px;
        border: 1px dashed rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.015);
        text-align: center;
        color: #7a8692;
        font-size: 0.75rem;
    }

    /* ---------- EMPTY STATES ---------- */

    .empty-state {
        padding: 1.2rem 1rem;
        border-radius: 12px;
        border: 1px dashed rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.017);
        text-align: center;
    }
    .empty-title {
        color: #d6dee5;
        font-size: 0.84rem;
        font-weight: 700;
    }
    .empty-text {
        margin-top: 0.2rem;
        color: #77838f;
        font-size: 0.72rem;
        line-height: 1.45;
    }

    /* ---------- INPUTS ---------- */

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 13px !important;
        background: rgba(255,255,255,0.022) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
    }
    div[data-baseweb="select"] > div { border-radius: 10px; }
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    [data-testid="stImage"] img { border-radius: 11px; }

    /* ---------- FOOTER ---------- */

    .footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        color: #65717d;
        font-size: 0.68rem;
        line-height: 1.55;
        text-align: center;
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
# UTILITIES
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
# IMAGE RESOLUTION
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
                candidate = DEMO_IMAGES_DIR / f"case_{idx:05d}_train{ext}"
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
# PROTOTYPE
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
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

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
# RETRIEVAL
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

        results.append({
            "rank": len(results) + 1,
            "embedding_row": int(row_index),
            "dataset_index": dataset_index,
            "prototype_id": prototype_id,
            "similarity": float(scores[row_index]),
            "image_path": actual_image,
            "image_filename": actual_image.name,
            "image_hash": content_hash,
        })

        if len(results) >= top_k:
            break

    debug["unique_retrieved"] = len(results)
    debug["requested"] = top_k
    return results, debug


# ============================================================
# RESET + PIPELINE
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
# CHAT ANSWER LAYER
# ============================================================

def build_case_context():
    if not st.session_state.demo_case:
        return None
    return {
        "dataset_index": int(
            st.session_state.demo_case.get("dataset_index", -1)
        ),
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


def answer_user_question(question, context):
    if context is None:
        return (
            "Please upload a prepared demonstration image first. "
            "I answer questions supported by the selected case's "
            "precomputed evidence."
        )

    q = str(question).lower().strip()
    if not q:
        return "Please type a question about the selected image."

    # 1) Precomputed QA
    matched = _match_qa(question, context.get("qa") or {})
    if matched is not None:
        return str(context["qa"][matched])

    pid = context.get("prototype_id")
    aff = context.get("prototype_affinity")
    aff_txt = f"{float(aff):.5f}" if aff is not None else "not available"
    pid_txt = f"P{int(pid):02d}" if pid is not None else "—"

    # 2) Model observation
    if any(k in q for k in (
        "what does the model see", "what does medgemma",
        "what does the model say", "model observation",
        "medgemma observation", "what findings", "what is visible",
        "what do you see", "describe the image", "describe this image",
    )):
        obs = context.get("analysis") or "No observation available."
        return f'MedGemma observation: "{obs}"'

    # 3) Prototype
    if any(k in q for k in (
        "what is the prototype", "which prototype", "why p",
        "why is this image assigned", "why was this image assigned",
        "prototype context", "assigned to p",
    )):
        if pid is None:
            return "No visual prototype is associated with this case."
        return (
            f"This image is associated with visual prototype {pid_txt}. "
            f"Prototype affinity: {aff_txt}. "
            "This reflects embedding-space proximity, not clinical "
            "confidence."
        )

    if "affinity" in q:
        if aff is None:
            return "Prototype affinity is not available for this case."
        return (
            f"Prototype affinity is {float(aff):.5f}. It is the cosine "
            "similarity between the image embedding and its prototype "
            "centroid — a measure of visual association."
        )

    # 4) Similar cases
    if any(k in q for k in (
        "similar", "reference case", "nearest", "why is this case",
        "why are these", "retrieved",
    )):
        ret = context.get("retrieval") or []
        if not ret:
            return "No reference cases were retrieved for this image."
        sims = ", ".join(f"{float(r['similarity']):.5f}" for r in ret)
        cases = ", ".join(str(int(r["dataset_index"])) for r in ret)
        return (
            f"Cases {cases} were retrieved as the nearest visual "
            f"neighbours in the learned embedding space. "
            f"Cosine similarities: {sims}. Similarity reflects visual "
            "proximity, not clinical equivalence."
        )

    # 5) Pipeline / methodology
    if any(k in q for k in (
        "how does", "how were", "pipeline", "explainability",
        "how is similarity", "how does similarity",
    )):
        return (
            "Each image is mapped to a 1152-D visual embedding. "
            "Cosine similarity ranks the reference database. "
            "The image is assigned to a learned visual prototype. "
            "MedGemma's observation is a separate precomputed output "
            "and is not derived from the retrieval step."
        )

    # 6) Evidence inventory
    if any(k in q for k in (
        "what evidence", "what is available", "what can you",
    )):
        ret = context.get("retrieval") or []
        qa = context.get("qa") or {}
        return (
            f"Available for this case: prototype {pid_txt}, "
            f"{len(ret)} retrieved reference cases, "
            f"{len(qa)} precomputed VQA entries, and one MedGemma "
            "observation."
        )

    # 7) Fallback
    return (
        "I answer questions supported by the prepared evidence for "
        "this case. This public demo does not run new MedGemma "
        "inference for arbitrary questions."
    )


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">
        <div class="hero-title">🩺 Explainable Medical VLM</div>
        <div class="hero-subtitle">
            Prototype-grounded visual evidence for medical image analysis
        </div>
        <div class="hero-row">
            <span class="badge badge-blue">MedGemma 1.5 4B</span>
            <span class="badge">Visual Retrieval</span>
            <span class="badge">Learned Prototypes</span>
            <span class="badge badge-green">Offline Demo</span>
        </div>
        <div class="hero-note">
            Research demo · Precomputed outputs · Not for clinical diagnosis
        </div>
    </div>
    """
)


# ============================================================
# 01 — MEDICAL IMAGE
# ============================================================

section_header("01", "Medical Image", "Prepared demonstration case")

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
                run_precomputed_case(
                    demo_case, query_image_hash=current_hash
                )
            else:
                st.session_state.error = (
                    "This image is not in the prepared demonstration "
                    "library. Live MedGemma inference is not enabled "
                    "in this public demo."
                )

        image = Image.open(uploaded_file).convert("RGB")
        case = st.session_state.demo_case

        left, right = st.columns([1.5, 1], gap="large")

        with left:
            st.image(image, use_container_width=True)
            st.caption(f"📄 {uploaded_file.name}")

        with right:
            if case is not None:
                dataset_index_val = int(case["dataset_index"])
                proto_val = (
                    f"P{int(case['prototype_id']):02d}"
                    if pd.notna(case.get("prototype_id"))
                    else "—"
                )
                render_html(
                    f"""
                    <div class="metric-strip"
                         style="grid-template-columns: 1fr 1fr;">
                        <div class="metric">
                            <div class="metric-label">Dataset</div>
                            <div class="metric-value">
                                {dataset_index_val}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Prototype</div>
                            <div class="metric-value">
                                {html.escape(proto_val)}
                            </div>
                        </div>
                        <div class="metric" style="grid-column: span 2;">
                            <div class="metric-label">Status</div>
                            <div class="metric-value ok">✓ Prepared</div>
                        </div>
                    </div>
                    """
                )
            else:
                render_html(
                    """
                    <div class="empty-state">
                        <div class="empty-title">
                            Not in demonstration library
                        </div>
                        <div class="empty-text">
                            This public demo supports prepared cases
                            with precomputed results.
                        </div>
                    </div>
                    """
                )

        if st.session_state.error:
            st.warning(st.session_state.error)

    except Exception as exc:
        st.error(f"Unable to process image: {exc}")


# ============================================================
# SYSTEM INFO EXPANDER
# ============================================================

with st.expander("⚙️ System information", expanded=False):
    debug = st.session_state.get("debug") or {}
    db_shape = debug.get("db_shape")
    db_shape_text = (
        f"{db_shape[0]} × {db_shape[1]}" if db_shape else "—"
    )
    row_text = (
        str(int(debug["embedding_row"]))
        if debug.get("embedding_row") is not None
        else "—"
    )
    proto_dbg = debug.get("prototype_id")
    proto_text = (
        f"P{int(proto_dbg):02d}" if proto_dbg is not None else "—"
    )
    aff_dbg = st.session_state.prototype_similarity
    aff_text = (
        f"{float(aff_dbg):.5f}" if aff_dbg is not None else "—"
    )
    unique_txt = (
        f"{debug.get('unique_retrieved', 0)} / "
        f"{debug.get('requested', MAX_SIMILAR_CASES)}"
    )
    qa_txt = str(debug.get("qa_count", "—"))
    dataset_dbg = (
        int(st.session_state.demo_case["dataset_index"])
        if st.session_state.demo_case is not None
        else None
    )
    dataset_text = str(dataset_dbg) if dataset_dbg is not None else "—"

    render_html(
        f"""
        <div class="metric-strip">
            <div class="metric">
                <div class="metric-label">Embedding</div>
                <div class="metric-value">{html.escape(db_shape_text)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Dataset index</div>
                <div class="metric-value">{html.escape(dataset_text)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Embedding row</div>
                <div class="metric-value">{html.escape(row_text)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Prototype</div>
                <div class="metric-value">{html.escape(proto_text)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Affinity</div>
                <div class="metric-value">{html.escape(aff_text)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Retrieved</div>
                <div class="metric-value">{html.escape(unique_txt)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">VQA entries</div>
                <div class="metric-value">{html.escape(qa_txt)}</div>
            </div>
        </div>
        """
    )
    st.caption(
        "Similarity values are computed exactly and never adjusted "
        "for display."
    )


# ============================================================
# 02 — MODEL ANALYSIS
# ============================================================

section_header("02", "Model Analysis", "Precomputed MedGemma observation")

if st.session_state.analysis:
    safe_analysis = html.escape(str(st.session_state.analysis))
    render_html(
        f"""
        <div class="obs-card">
            <div class="obs-label">MedGemma observation</div>
            <div class="obs-text">{safe_analysis}</div>
            <div style="margin-top:0.6rem;">
                <span class="badge">Precomputed</span>
                <span class="badge">Image-grounded</span>
            </div>
        </div>
        """
    )
else:
    render_html(
        """
        <div class="empty-state">
            <div class="empty-title">Waiting for a prepared image</div>
            <div class="empty-text">
                Upload a demonstration image to display the
                precomputed MedGemma observation.
            </div>
        </div>
        """
    )


# ============================================================
# 03 — SIMILAR CASES
# ============================================================

section_header("03", "Similar Cases", "Nearest visual neighbours")

results = st.session_state.retrieval_results

if results:
    columns = st.columns(len(results), gap="medium")
    for column, result in zip(columns, results):
        with column:
            image_path = result.get("image_path")
            if image_path is not None and Path(image_path).exists():
                st.image(str(image_path), use_container_width=True)
            else:
                render_html(
                    """
                    <div class="img-placeholder">
                        Image unavailable
                    </div>
                    """
                )

            similarity = float(result["similarity"])
            prototype = int(result["prototype_id"])

            render_html(
                f"""
                <div class="case-mini">
                    <div class="case-mini-title">
                        Case {int(result["dataset_index"])}
                    </div>
                    <div class="case-mini-meta">
                        P{prototype:02d} · {similarity:.5f}
                    </div>
                </div>
                """
            )

    render_html(
        """
        <div class="muted-line">
            Ranked by cosine similarity in the learned visual
            embedding space.
        </div>
        """
    )
else:
    render_html(
        """
        <div class="empty-state">
            <div class="empty-title">No retrieved cases yet</div>
            <div class="empty-text">
                Upload a prepared demonstration image to retrieve
                visually similar reference cases.
            </div>
        </div>
        """
    )


# ============================================================
# 04 — PROTOTYPE EVIDENCE
# ============================================================

section_header(
    "04", "Prototype Evidence",
    "Representative images of the assigned learned prototype",
)

prototype_id = st.session_state.prototype_id

if prototype_id is not None:
    description = get_prototype_description(prototype_id)
    score = st.session_state.prototype_similarity
    score_text = (
        f"{float(score):.5f}" if score is not None else "—"
    )

    render_html(
        f"""
        <div class="proto-header">
            <div>
                <div class="proto-id">P{int(prototype_id):02d}</div>
                <div class="proto-desc">
                    {html.escape(str(description))}
                </div>
            </div>
            <div class="proto-aff">
                <span class="proto-aff-label">Affinity</span>
                <span class="proto-aff-val">{score_text}</span>
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
                    <div class="img-caption">
                        Representative {index}
                    </div>
                    """
                )
        render_html(
            """
            <div class="muted-line">
                Learned visual prototype · latent representation
            </div>
            """
        )
    else:
        render_html(
            """
            <div class="empty-state">
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
            <div class="empty-title">Prototype not available yet</div>
            <div class="empty-text">
                Upload a prepared demonstration image to identify
                its learned visual prototype.
            </div>
        </div>
        """
    )


# ============================================================
# 05 — EXPLAINABILITY
# ============================================================

section_header(
    "05", "Explainability",
    "Case-specific summary of retrieved visual evidence",
)

case = st.session_state.demo_case

if case is None:
    render_html(
        """
        <div class="empty-state">
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

    proto_txt = (
        f"P{int(prototype_id):02d}"
        if prototype_id is not None
        else "—"
    )
    aff_txt = (
        f"{float(st.session_state.prototype_similarity):.5f}"
        if st.session_state.prototype_similarity is not None
        else "—"
    )

    render_html(
        f"""
        <div class="metric-strip">
            <div class="metric">
                <div class="metric-label">Dataset</div>
                <div class="metric-value">{dataset_index_val}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Prototype</div>
                <div class="metric-value">{html.escape(proto_txt)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Affinity</div>
                <div class="metric-value">{html.escape(aff_txt)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Neighbours</div>
                <div class="metric-value">{len(results)}</div>
            </div>
        </div>
        """
    )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        if results:
            rows_html = (
                '<div class="neighbors">'
                '<div class="hdr">Case</div>'
                '<div class="hdr">Prototype</div>'
                '<div class="hdr">Similarity</div>'
            )
            for r in results:
                rows_html += (
                    f'<div class="cell">'
                    f'{int(r["dataset_index"])}'
                    f'</div>'
                    f'<div class="cell muted">'
                    f'P{int(r["prototype_id"]):02d}'
                    f'</div>'
                    f'<div class="cell">'
                    f'{float(r["similarity"]):.5f}'
                    f'</div>'
                )
            rows_html += "</div>"

            render_html(
                f"""
                <div style="margin-bottom:0.35rem;
                            color:#8996a3;
                            font-size:0.68rem;
                            font-weight:750;
                            letter-spacing:0.08em;
                            text-transform:uppercase;">
                    Nearest visual neighbours
                </div>
                {rows_html}
                """
            )
        else:
            render_html(
                """
                <div class="empty-state">
                    <div class="empty-text">
                        No reference cases retrieved.
                    </div>
                </div>
                """
            )

    with right:
        observation_text = (
            st.session_state.analysis
            or "No precomputed observation is available."
        )
        render_html(
            f"""
            <div class="obs-card">
                <div class="obs-label">Model observation</div>
                <div class="obs-text"
                     style="font-style:italic;">
                    "{html.escape(str(observation_text))}"
                </div>
            </div>
            """
        )

    render_html(
        """
        <div class="muted-line">
            Similarity reflects visual embedding proximity, not
            clinical similarity.
        </div>
        """
    )

    with st.expander("▸ How the evidence layer works", expanded=False):
        render_html(
            """
            <div style="color:#a3aeb9; font-size:0.8rem; line-height:1.6;">
                Image
                → 1152-D visual embedding
                → cosine retrieval against the reference database
                → learned visual prototype assignment
                → prototype affinity
                → retrieved visual evidence.
                <br><br>
                The MedGemma observation is a separate precomputed
                output. Retrieval does not cause the observation.
            </div>
            """
        )

    with st.expander("▸ Interpretation boundary", expanded=False):
        render_html(
            """
            <div style="color:#a3aeb9; font-size:0.8rem; line-height:1.6;">
                <b style="color:#cfc4a3;">Can show:</b>
                embedding similarity · prototype association ·
                retrieved visual neighbours · precomputed model
                observations.
                <br><br>
                <b style="color:#cfc4a3;">Cannot prove:</b>
                clinical diagnosis · causality · model faithfulness ·
                that MedGemma relied on the retrieved prototype.
                <br><br>
                Prototype affinity represents visual association in
                the learned representation space. It is not clinical
                confidence.
            </div>
            """
        )


# ============================================================
# 06 — ASK ABOUT THIS IMAGE
# ============================================================

section_header(
    "06", "Ask About This Image",
    "Ask about the image, retrieved evidence, or prototype",
)

if case is None:
    render_html(
        """
        <div class="empty-state">
            <div class="empty-title">Select a prepared image first</div>
            <div class="empty-text">
                The assistant answers questions supported by the
                prepared evidence for the selected case.
            </div>
        </div>
        """
    )
else:
    chat_context = build_case_context()

    quick_questions = [
        "What does the model see?",
        "Why is this case similar?",
        "What is the prototype?",
        "What evidence is available?",
    ]

    qcols = st.columns(len(quick_questions))

    for qcol, qtext in zip(qcols, quick_questions):
        with qcol:
            if st.button(
                qtext,
                key=f"quick_{qtext}_{st.session_state.image_hash or 'x'}",
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

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        render_html(
            """
            <div class="empty-state" style="margin-top:0.4rem;">
                <div class="empty-title">No messages yet</div>
                <div class="empty-text">
                    Ask a question below, or use one of the quick
                    questions above.
                </div>
            </div>
            """
        )

    user_prompt = st.chat_input(
        "Ask about the image, its prototype, or its retrieved cases..."
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
        Explainable Medical VLM · MedGemma · Visual Retrieval ·
        Prototype-Grounded Evidence
        <br>
        Research demonstration only · Not for clinical diagnosis
    </div>
    """
)
