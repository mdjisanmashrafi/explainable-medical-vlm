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
# HTML HELPERS
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
        f'<div class="s-desc">{html.escape(subtitle)}</div>'
        if subtitle
        else ""
    )
    render_html(
        f"""
        <div class="s-head">
            <div class="s-num">{html.escape(number)}</div>
            <div>
                <div class="s-title">{html.escape(title)}</div>
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
        max-width: 1220px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5 { color: #f1f4f7 !important; }
    p { color: #aeb8c3; }

    /* ---------- HERO ---------- */
    .hero {
        padding: 1.2rem 1.5rem 1.1rem 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.07);
        background: linear-gradient(135deg,
            rgba(24,31,40,0.96), rgba(13,18,24,0.96));
        box-shadow: 0 12px 36px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 760;
        letter-spacing: -0.035em;
        color: #f4f7fa;
        line-height: 1.12;
    }
    .hero-sub {
        margin-top: 0.22rem;
        color: #a4aeb9;
        font-size: 0.88rem;
    }
    .hero-row {
        display: flex; flex-wrap: wrap;
        gap: 0.35rem; margin-top: 0.7rem;
    }
    .badge {
        display: inline-flex; align-items: center;
        padding: 0.24rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.04);
        color: #c7d0d9;
        font-size: 0.68rem;
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
        margin-top: 0.7rem;
        color: #7a8692;
        font-size: 0.7rem;
    }

    /* ---------- SECTION HEADER ---------- */
    .s-head {
        display: flex; align-items: center;
        gap: 0.6rem; margin: 1.5rem 0 0.6rem 0;
    }
    .s-num {
        width: 28px; height: 28px; flex: 0 0 28px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #d7e0e8;
        font-size: 0.7rem; font-weight: 750;
    }
    .s-title {
        color: #edf1f5;
        font-size: 1rem;
        font-weight: 720;
        letter-spacing: -0.01em;
    }
    .s-desc {
        margin-top: 0.04rem;
        color: #7f8b98;
        font-size: 0.73rem;
    }

    /* ---------- METRIC STRIP ---------- */
    .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(115px, 1fr));
        gap: 0.45rem;
        margin-bottom: 0.75rem;
    }
    .metric {
        padding: 0.55rem 0.7rem;
        border-radius: 9px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.022);
    }
    .m-label {
        color: #778492;
        font-size: 0.58rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .m-value {
        margin-top: 0.15rem;
        color: #e4eaf0;
        font-size: 0.95rem;
        font-weight: 650;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .m-value.ok { color: #a6d7b6; }

    /* ---------- OBSERVATION CARD ---------- */
    .obs-card {
        padding: 0.95rem 1.05rem;
        border-radius: 12px;
        border: 1px solid rgba(105,145,190,0.18);
        background: linear-gradient(135deg,
            rgba(50,72,98,0.16), rgba(20,27,35,0.86));
    }
    .obs-label {
        color: #8094a9;
        font-size: 0.6rem; font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .obs-text {
        color: #e9eef3;
        font-size: 0.94rem;
        line-height: 1.5;
    }

    /* ---------- MINI CASE CARD ---------- */
    .case-mini {
        margin-top: 0.4rem;
        padding: 0.45rem 0.6rem;
        border-radius: 9px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.5rem;
    }
    .case-mini-title {
        color: #dfe6ec;
        font-size: 0.82rem;
        font-weight: 650;
    }
    .case-mini-meta {
        color: #8996a3;
        font-size: 0.72rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* ---------- PROTOTYPE HEADER ---------- */
    .proto-header {
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 0.7rem;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.07);
        background: linear-gradient(135deg,
            rgba(28,35,44,0.95), rgba(17,22,29,0.95));
        margin-bottom: 0.65rem;
    }
    .proto-id {
        color: #e9eef3;
        font-size: 1rem;
        font-weight: 750;
    }
    .proto-desc {
        margin-top: 0.12rem;
        color: #929eaa;
        font-size: 0.72rem;
    }
    .proto-aff {
        display: flex; align-items: baseline; gap: 0.45rem;
    }
    .proto-aff-label {
        color: #778492;
        font-size: 0.58rem; font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .proto-aff-val {
        color: #e6edf4;
        font-size: 1.1rem; font-weight: 750;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* ---------- NEIGHBOURS TABLE ---------- */
    .neighbors {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.25rem 0.6rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.82rem;
    }
    .neighbors .hdr {
        color: #778492;
        font-size: 0.58rem; font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: system-ui, sans-serif;
        padding-bottom: 0.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .neighbors .cell { color: #dfe6ec; padding-top: 0.15rem; }
    .neighbors .cell.muted { color: #a3aeb9; }

    /* ---------- MUTED / CAPTION ---------- */
    .muted {
        color: #7a8692;
        font-size: 0.7rem;
        line-height: 1.45;
        margin-top: 0.35rem;
    }
    .img-caption {
        margin-top: 0.3rem;
        text-align: center;
        color: #6f7c88;
        font-size: 0.66rem;
    }
    .img-placeholder {
        padding: 2rem 0.5rem;
        border-radius: 11px;
        border: 1px dashed rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.015);
        text-align: center;
        color: #7a8692;
        font-size: 0.72rem;
    }

    /* ---------- EMPTY STATES ---------- */
    .empty {
        padding: 1.1rem 0.9rem;
        border-radius: 11px;
        border: 1px dashed rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.017);
        text-align: center;
    }
    .empty-t {
        color: #d6dee5;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .empty-x {
        margin-top: 0.18rem;
        color: #77838f;
        font-size: 0.7rem;
        line-height: 1.4;
    }

    /* ---------- INPUTS ---------- */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.022) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
    }
    div[data-baseweb="select"] > div { border-radius: 9px; }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    [data-testid="stImage"] img { border-radius: 10px; }

    /* ---------- FOOTER ---------- */
    .footer {
        margin-top: 2.2rem;
        padding-top: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        color: #65717d;
        font-size: 0.66rem;
        line-height: 1.5;
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
        "description", "prototype_description",
        "summary", "interpretation",
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
                "image_path", "representative_image",
                "image", "filename",
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
                p for p in folder.iterdir()
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
# PIPELINE
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
            r["image_hash"] for r in similar_cases
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
                if similar_cases else None
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


def _match_qa(question, qa_dict, threshold=0.45):
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
    """Concise, evidence-grounded answers. Never hallucinate."""

    if context is None:
        return "Upload a prepared demonstration image first."

    q = str(question).lower().strip()
    if not q:
        return "Please type a question about the selected image."

    # 1) Precomputed QA — best semantic match
    matched = _match_qa(question, context.get("qa") or {})
    if matched is not None:
        return str(context["qa"][matched])

    pid = context.get("prototype_id")
    aff = context.get("prototype_affinity")
    pid_txt = f"P{int(pid):02d}" if pid is not None else "—"
    aff_txt = f"{float(aff):.5f}" if aff is not None else "not available"
    ret = context.get("retrieval") or []

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
        "what is the prototype", "which prototype",
        "why p", "why is this image assigned",
        "why was this image assigned", "assigned to p",
    )):
        if pid is None:
            return "No visual prototype is associated with this case."
        return (
            f"Assigned to visual prototype {pid_txt} with affinity "
            f"{aff_txt}. This is a learned visual grouping, not a "
            "clinical diagnosis."
        )

    if "affinity" in q:
        if aff is None:
            return "Prototype affinity is not available for this case."
        return (
            f"Prototype affinity is {float(aff):.5f} — cosine "
            "similarity to the prototype centroid."
        )

    # 4) Similar cases
    if any(k in q for k in (
        "similar", "reference case", "nearest",
        "why is this case", "why are these", "retrieved",
    )):
        if not ret:
            return "No reference cases were retrieved for this image."
        cases = ", ".join(str(int(r["dataset_index"])) for r in ret)
        sims = ", ".join(f"{float(r['similarity']):.5f}" for r in ret)
        return (
            f"Nearest visual neighbours are Cases {cases} "
            f"(cosine similarity: {sims})."
        )

    # 5) Pipeline
    if any(k in q for k in (
        "how does", "how were", "pipeline",
        "explainability", "how is similarity",
        "how does similarity",
    )):
        return (
            "Each image is mapped to a 1152-D visual embedding; "
            "cosine similarity ranks the reference database; the "
            "image is assigned a learned visual prototype. Retrieval "
            "is a separate layer from MedGemma's precomputed output."
        )

    # 6) Evidence inventory
    if any(k in q for k in (
        "what evidence", "what is available", "what can you",
    )):
        return (
            f"Available: prototype {pid_txt}, "
            f"{len(ret)} retrieved neighbours, "
            f"{len(context.get('qa') or {})} precomputed VQA entries, "
            "and one MedGemma observation."
        )

    # 7) Fallback
    return (
        "That information is not available in the current "
        "precomputed evidence."
    )


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">
        <div class="hero-title">🩺 Explainable Medical VLM</div>
        <div class="hero-sub">
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

        left, right = st.columns([1.55, 1], gap="large")

        with left:
            st.image(image, use_container_width=True)
            st.caption(f"📄 {uploaded_file.name}")

        with right:
            if case is not None:
                dataset_val = int(case["dataset_index"])
                proto_val = (
                    f"P{int(case['prototype_id']):02d}"
                    if pd.notna(case.get("prototype_id"))
                    else "—"
                )
                render_html(
                    f"""
                    <div class="metrics"
                         style="grid-template-columns: 1fr 1fr;">
                        <div class="metric">
                            <div class="m-label">Dataset</div>
                            <div class="m-value">{dataset_val}</div>
                        </div>
                        <div class="metric">
                            <div class="m-label">Prototype</div>
                            <div class="m-value">{html.escape(proto_val)}</div>
                        </div>
                        <div class="metric" style="grid-column: span 2;">
                            <div class="m-label">Status</div>
                            <div class="m-value ok">✓ Prepared</div>
                        </div>
                    </div>
                    """
                )
            else:
                render_html(
                    """
                    <div class="empty">
                        <div class="empty-t">
                            Not in demonstration library
                        </div>
                        <div class="empty-x">
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
# SYSTEM INFO
# ============================================================

with st.expander("⚙️ System information", expanded=False):
    debug = st.session_state.get("debug") or {}
    db_shape = debug.get("db_shape")
    db_text = f"{db_shape[0]} × {db_shape[1]}" if db_shape else "—"
    row_text = (
        str(int(debug["embedding_row"]))
        if debug.get("embedding_row") is not None else "—"
    )
    proto_dbg = debug.get("prototype_id")
    proto_text = (
        f"P{int(proto_dbg):02d}" if proto_dbg is not None else "—"
    )
    aff_dbg = st.session_state.prototype_similarity
    aff_text = f"{float(aff_dbg):.5f}" if aff_dbg is not None else "—"
    unique_txt = (
        f"{debug.get('unique_retrieved', 0)} / "
        f"{debug.get('requested', MAX_SIMILAR_CASES)}"
    )
    qa_txt = str(debug.get("qa_count", "—"))
    dataset_dbg = (
        int(st.session_state.demo_case["dataset_index"])
        if st.session_state.demo_case is not None else None
    )
    dataset_text = str(dataset_dbg) if dataset_dbg is not None else "—"

    render_html(
        f"""
        <div class="metrics">
            <div class="metric">
                <div class="m-label">Embedding</div>
                <div class="m-value">{html.escape(db_text)}</div>
            </div>
            <div class="metric">
                <div class="m-label">Dataset index</div>
                <div class="m-value">{html.escape(dataset_text)}</div>
            </div>
            <div class="metric">
                <div class="m-label">Embedding row</div>
                <div class="m-value">{html.escape(row_text)}</div>
            </div>
            <div class="metric">
                <div class="m-label">Prototype</div>
                <div class="m-value">{html.escape(proto_text)}</div>
            </div>
            <div class="metric">
                <div class="m-label">Affinity</div>
                <div class="m-value">{html.escape(aff_text)}</div>
            </div>
            <div class="metric">
                <div class="m-label">Retrieved</div>
                <div class="m-value">{html.escape(unique_txt)}</div>
            </div>
            <div class="metric">
                <div class="m-label">VQA entries</div>
                <div class="m-value">{html.escape(qa_txt)}</div>
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

section_header(
    "02", "Model Analysis", "Precomputed MedGemma observation"
)

if st.session_state.analysis:
    safe_analysis = html.escape(str(st.session_state.analysis))
    render_html(
        f"""
        <div class="obs-card">
            <div class="obs-label">MedGemma observation</div>
            <div class="obs-text">{safe_analysis}</div>
            <div style="margin-top:0.55rem;">
                <span class="badge">Precomputed</span>
                <span class="badge">Image-grounded</span>
            </div>
        </div>
        """
    )
else:
    render_html(
        """
        <div class="empty">
            <div class="empty-t">Waiting for a prepared image</div>
            <div class="empty-x">
                Upload a demonstration image to display the
                precomputed observation.
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

            sim = float(result["similarity"])
            proto = int(result["prototype_id"])

            render_html(
                f"""
                <div class="case-mini">
                    <div class="case-mini-title">
                        Case {int(result["dataset_index"])}
                    </div>
                    <div class="case-mini-meta">
                        P{proto:02d} · {sim:.5f}
                    </div>
                </div>
                """
            )

    render_html(
        """
        <div class="muted">
            Ranked by cosine similarity in the learned visual
            embedding space.
        </div>
        """
    )
else:
    render_html(
        """
        <div class="empty">
            <div class="empty-t">No retrieved cases yet</div>
            <div class="empty-x">
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
    "04", "Prototype Evidence", "Learned visual prototype"
)

prototype_id = st.session_state.prototype_id

if prototype_id is not None:
    description = get_prototype_description(prototype_id)
    score = st.session_state.prototype_similarity
    score_text = f"{float(score):.5f}" if score is not None else "—"

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
            <div class="muted">
                Learned visual prototype · latent representation
            </div>
            """
        )
    else:
        render_html(
            """
            <div class="empty">
                <div class="empty-t">
                    Representative images unavailable
                </div>
                <div class="empty-x">
                    The prototype was identified, but no unique
                    representative images could be resolved.
                </div>
            </div>
            """
        )
else:
    render_html(
        """
        <div class="empty">
            <div class="empty-t">Prototype not available yet</div>
            <div class="empty-x">
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
    "Case-specific visual evidence and retrieved evidence summary",
)

case = st.session_state.demo_case


# ------------------------------------------------------------
# LOCAL VISUAL SALIENCY HEATMAP
# ------------------------------------------------------------
def generate_visual_saliency_heatmap(image):
    """
    Generate a lightweight image-based visual saliency map.

    IMPORTANT:
    This is NOT Grad-CAM and does not use MedGemma gradients.
    It highlights visually salient regions using local contrast
    and image intensity variation.
    """

    import numpy as np
    from PIL import Image, ImageFilter

    # Convert to grayscale
    gray = image.convert("L")

    # Normalize image
    arr = np.asarray(gray, dtype=np.float32)

    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr)

    # --------------------------------------------------------
    # 1. Local contrast
    # --------------------------------------------------------
    smooth = (
        Image.fromarray((arr * 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(radius=18))
    )

    smooth_arr = np.asarray(smooth, dtype=np.float32) / 255.0

    local_contrast = np.abs(arr - smooth_arr)

    # --------------------------------------------------------
    # 2. Edge / intensity variation
    # --------------------------------------------------------
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)

    gx[:, 1:-1] = np.abs(arr[:, 2:] - arr[:, :-2])
    gy[1:-1, :] = np.abs(arr[2:, :] - arr[:-2, :])

    edge_strength = np.sqrt(gx ** 2 + gy ** 2)

    # --------------------------------------------------------
    # 3. Combine signals
    # --------------------------------------------------------
    saliency = (
        0.70 * local_contrast +
        0.30 * edge_strength
    )

    # Remove tiny numerical noise
    saliency[saliency < np.percentile(saliency, 35)] = 0

    # Smooth the final map
    saliency_img = Image.fromarray(
        np.clip(saliency * 255, 0, 255).astype(np.uint8)
    )

    saliency_img = saliency_img.filter(
        ImageFilter.GaussianBlur(radius=5)
    )

    saliency = np.asarray(
        saliency_img,
        dtype=np.float32
    ) / 255.0

    # Robust normalization
    low = np.percentile(saliency, 5)
    high = np.percentile(saliency, 98)

    if high > low:
        saliency = np.clip(
            (saliency - low) / (high - low),
            0,
            1
        )
    else:
        saliency = np.zeros_like(saliency)

    # --------------------------------------------------------
    # 4. Create heatmap
    # --------------------------------------------------------
    heat_uint8 = np.clip(
        saliency * 255,
        0,
        255
    ).astype(np.uint8)

    heat_gray = Image.fromarray(
        heat_uint8,
        mode="L"
    )

    # Use PIL's built-in colorization
    heatmap = Image.new(
        "RGB",
        heat_gray.size
    )

    # Create a simple blue → cyan → yellow → red map
    h = np.asarray(heat_gray, dtype=np.float32) / 255.0

    r = np.clip(2.0 * h - 0.5, 0, 1)
    g = np.clip(2.0 * h, 0, 1)
    b = np.clip(1.5 - 2.0 * h, 0, 1)

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    heatmap = Image.fromarray(
        (rgb * 255).astype(np.uint8),
        mode="RGB"
    )

    # --------------------------------------------------------
    # 5. Overlay heatmap on original image
    # --------------------------------------------------------
    base = image.convert("RGB").resize(heatmap.size)

    overlay = Image.blend(
        base,
        heatmap,
        alpha=0.48
    )

    return heatmap, overlay


# ------------------------------------------------------------
# EMPTY STATE
# ------------------------------------------------------------
if case is None:

    render_html(
        """
        <div class="empty">
            <div class="empty-t">
                Case-specific explanation not available yet
            </div>
            <div class="empty-x">
                Upload a prepared demonstration image to see a
                case-specific evidence summary.
            </div>
        </div>
        """
    )

else:

    # --------------------------------------------------------
    # CASE METADATA
    # --------------------------------------------------------
    try:
        dataset_val = int(case.get("dataset_index"))
    except Exception:
        dataset_val = -1

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

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------
    render_html(
        f"""
        <div class="metrics">

            <div class="metric">
                <div class="m-label">Dataset</div>
                <div class="m-value">{dataset_val}</div>
            </div>

            <div class="metric">
                <div class="m-label">Prototype</div>
                <div class="m-value">
                    {html.escape(proto_txt)}
                </div>
            </div>

            <div class="metric">
                <div class="m-label">Affinity</div>
                <div class="m-value">
                    {html.escape(aff_txt)}
                </div>
            </div>

            <div class="metric">
                <div class="m-label">Neighbours</div>
                <div class="m-value">
                    {len(results)}
                </div>
            </div>

        </div>
        """
    )

    # ========================================================
    # VISUAL EVIDENCE
    # ========================================================

    render_html(
        """
        <div style="
            margin-top:1.2rem;
            margin-bottom:0.7rem;
            color:#8996a3;
            font-size:0.66rem;
            font-weight:750;
            letter-spacing:0.08em;
            text-transform:uppercase;
        ">
            Visual evidence
        </div>
        """
    )

    # --------------------------------------------------------
    # Get currently selected / prepared image
    # --------------------------------------------------------
    current_image = None

    try:

        image_path = None

        # Prefer deterministic dataset-index mapping
        if dataset_val >= 0:

            possible_paths = [
                ROOT / "demo_images" /
                f"case_{dataset_val:05d}_train.png",

                ROOT / "demo_images" /
                f"case_{dataset_val:05d}_train.jpg",

                ROOT / "demo_images" /
                f"case_{dataset_val:05d}_train.jpeg",
            ]

            for p in possible_paths:
                if p.exists():
                    image_path = p
                    break

        # Fallback to case image path
        if image_path is None:

            case_path = case.get("image_path")

            if case_path:
                candidate = ROOT / str(case_path)

                if candidate.exists():
                    image_path = candidate

        if image_path is not None:
            current_image = Image.open(image_path).convert("RGB")

    except Exception:
        current_image = None


    # --------------------------------------------------------
    # Generate local saliency map
    # --------------------------------------------------------
    if current_image is not None:

        try:

            heatmap_image, overlay_image = (
                generate_visual_saliency_heatmap(
                    current_image
                )
            )

            img_col, heat_col = st.columns(
                [1, 1],
                gap="large"
            )

            with img_col:

                render_html(
                    """
                    <div style="
                        color:#8996a3;
                        font-size:0.66rem;
                        font-weight:750;
                        letter-spacing:0.08em;
                        text-transform:uppercase;
                        margin-bottom:0.45rem;
                    ">
                        Original image
                    </div>
                    """
                )

                st.image(
                    current_image,
                    use_container_width=True
                )

            with heat_col:

                render_html(
                    """
                    <div style="
                        color:#8996a3;
                        font-size:0.66rem;
                        font-weight:750;
                        letter-spacing:0.08em;
                        text-transform:uppercase;
                        margin-bottom:0.45rem;
                    ">
                        Visual saliency
                    </div>
                    """
                )

                st.image(
                    overlay_image,
                    use_container_width=True
                )

            render_html(
                """
                <div style="
                    margin-top:0.55rem;
                    padding:0.75rem 0.9rem;
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:10px;
                    background:rgba(255,255,255,0.025);
                    color:#8996a3;
                    font-size:0.72rem;
                    line-height:1.5;
                ">
                    <b style="color:#cfd6dd;">
                        Visual saliency map
                    </b>
                    highlights image regions with stronger local
                    visual contrast and intensity variation.
                    It is an image-based visualization and is
                    <b style="color:#cfd6dd;">
                        not a MedGemma Grad-CAM map,
                    </b>
                    lesion segmentation, or clinical diagnosis.
                </div>
                """
            )

        except Exception as e:

            render_html(
                f"""
                <div class="empty">
                    <div class="empty-x">
                        Visual evidence map could not be generated.
                    </div>
                </div>
                """
            )

    else:

        render_html(
            """
            <div class="empty">
                <div class="empty-x">
                    Original image unavailable for visual evidence.
                </div>
            </div>
            """
        )


    # ========================================================
    # RETRIEVED EVIDENCE + MODEL OBSERVATION
    # ========================================================

    render_html(
        """
        <div style="
            margin-top:1.4rem;
            margin-bottom:0.7rem;
            color:#8996a3;
            font-size:0.66rem;
            font-weight:750;
            letter-spacing:0.08em;
            text-transform:uppercase;
        ">
            Retrieved evidence
        </div>
        """
    )

    left, right = st.columns(
        [1.15, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # NEAREST NEIGHBOURS
    # --------------------------------------------------------
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
                <div style="
                    margin-bottom:0.3rem;
                    color:#8996a3;
                    font-size:0.66rem;
                    font-weight:750;
                    letter-spacing:0.08em;
                    text-transform:uppercase;
                ">
                    Nearest visual neighbours
                </div>

                {rows_html}
                """
            )

        else:

            render_html(
                """
                <div class="empty">
                    <div class="empty-x">
                        No reference cases retrieved.
                    </div>
                </div>
                """
            )


    # --------------------------------------------------------
    # MODEL OBSERVATION
    # --------------------------------------------------------
    with right:

        observation_text = (
            st.session_state.analysis
            or "No precomputed observation is available."
        )

        render_html(
            f"""
            <div class="obs-card">

                <div class="obs-label">
                    Model observation
                </div>

                <div class="obs-text"
                     style="font-style:italic;">
                    "{html.escape(str(observation_text))}"
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # RETRIEVAL DISCLAIMER
    # --------------------------------------------------------
    render_html(
        """
        <div class="muted">
            Similarity reflects visual embedding proximity,
            not clinical equivalence.
        </div>
        """
    )


    # ========================================================
    # HOW THE EVIDENCE LAYER WORKS
    # ========================================================

    with st.expander(
        "▸ How the evidence layer works",
        expanded=False
    ):

        render_html(
            """
            <div style="
                color:#a3aeb9;
                font-size:0.78rem;
                line-height:1.55;
            ">

                <b style="color:#cfd6dd;">1.</b>
                The selected image has a precomputed 1152-D
                visual embedding.

                <br>

                <b style="color:#cfd6dd;">2.</b>
                The embedding is compared against the reference
                image database.

                <br>

                <b style="color:#cfd6dd;">3.</b>
                Nearest visual neighbours are retrieved using
                cosine similarity.

                <br>

                <b style="color:#cfd6dd;">4.</b>
                The image is assigned to a learned visual
                prototype.

                <br>

                <b style="color:#cfd6dd;">5.</b>
                Prototype representatives provide additional
                visual context.

                <br>

                <b style="color:#cfd6dd;">6.</b>
                A local visual-saliency map provides an additional
                image-level visualization of visually prominent
                regions.

                <br>

                <b style="color:#cfd6dd;">7.</b>
                These signals form an evidence layer around the
                precomputed MedGemma observation.

                <br><br>

                <span style="color:#7a8692;">
                    Retrieval does not cause the model output.
                    The observation is a separate precomputed
                    artifact.
                </span>

            </div>
            """
        )


    # ========================================================
    # INTERPRETATION BOUNDARY
    # ========================================================

    with st.expander(
        "▸ Interpretation boundary",
        expanded=False
    ):

        render_html(
            """
            <div style="
                color:#a3aeb9;
                font-size:0.78rem;
                line-height:1.55;
            ">

                <b style="color:#cfc4a3;">
                    Boundaries:
                </b>

                <ul style="
                    margin-top:0.3rem;
                    padding-left:1.1rem;
                ">

                    <li>
                        Prototypes are latent visual groupings,
                        not clinical concepts.
                    </li>

                    <li>
                        Similarity indicates embedding-space
                        proximity, not clinical equivalence.
                    </li>

                    <li>
                        Retrieval does not prove causal influence
                        on the model output.
                    </li>

                    <li>
                        The visual saliency map is not a lesion
                        segmentation or clinical localization.
                    </li>

                    <li>
                        MedGemma observations are model-generated
                        and are not clinical diagnoses.
                    </li>

                </ul>

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
        <div class="empty">
            <div class="empty-t">Select a prepared image first</div>
            <div class="empty-x">
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
                key=f"q_{qtext}_{st.session_state.image_hash or 'x'}",
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
            <div class="empty" style="margin-top:0.35rem;">
                <div class="empty-t">No messages yet</div>
                <div class="empty-x">
                    Ask a question below, or use one of the quick
                    questions above.
                </div>
            </div>
            """
        )

    user_prompt = st.chat_input(
        "Ask a question about this image..."
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
