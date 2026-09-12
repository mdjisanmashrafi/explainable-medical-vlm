import hashlib
import html
import json
import re
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter

from src.retrieval import load_retrieval_artifacts


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "MedXplain"
APP_SUBTITLE = (
    "Interpretable medical image understanding through visual "
    "evidence and prototypes"
)

st.set_page_config(
    page_title=APP_NAME,
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
    sub = f'<div class="s-desc">{html.escape(subtitle)}</div>' if subtitle else ""
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


def empty_state(title: str, detail: str = "") -> None:
    detail_html = f'<div class="empty-x">{html.escape(detail)}</div>' if detail else ""
    render_html(
        f"""
        <div class="empty">
            <div class="empty-t">{html.escape(title)}</div>
            {detail_html}
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
            radial-gradient(circle at 8% 0%, rgba(70,90,115,0.12), transparent 34%),
            #0b0f14;
        color: #e8edf3;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5 { color: #f1f4f7 !important; }
    p { color: #aeb8c3; }

    /* ---------- HERO ---------- */
    .hero {
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
        background: linear-gradient(135deg, rgba(24,31,40,0.96), rgba(13,18,24,0.96));
        margin-bottom: 1.1rem;
    }
    .hero-title {
        font-size: 1.55rem;
        font-weight: 780;
        letter-spacing: -0.02em;
        color: #f4f7fa;
    }
    .hero-sub {
        margin-top: 0.25rem;
        color: #a4aeb9;
        font-size: 0.85rem;
    }
    .hero-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.65rem; }
    .badge {
        display: inline-flex; align-items: center;
        padding: 0.22rem 0.55rem;
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

    /* ---------- SECTION HEADER ---------- */
    .s-head { display: flex; align-items: center; gap: 0.6rem; margin: 1.6rem 0 0.6rem 0; }
    .s-num {
        width: 26px; height: 26px; flex: 0 0 26px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #d7e0e8;
        font-size: 0.68rem; font-weight: 750;
    }
    .s-title { color: #edf1f5; font-size: 0.98rem; font-weight: 720; letter-spacing: -0.01em; }
    .s-desc { margin-top: 0.03rem; color: #7f8b98; font-size: 0.72rem; }

    /* ---------- OBSERVATION CARD ---------- */
    .obs-card {
        padding: 0.9rem 1.05rem;
        border-radius: 12px;
        border: 1px solid rgba(105,145,190,0.18);
        background: linear-gradient(135deg, rgba(50,72,98,0.16), rgba(20,27,35,0.86));
    }
    .obs-label {
        color: #8094a9; font-size: 0.6rem; font-weight: 750;
        letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.4rem;
    }
    .obs-text { color: #e9eef3; font-size: 0.93rem; line-height: 1.5; }

    /* ---------- CASE CARD (similar cases) ---------- */
    .case-meta {
        margin-top: 0.4rem;
        text-align: center;
    }
    .case-title { color: #dfe6ec; font-size: 0.83rem; font-weight: 650; }
    .case-sub {
        color: #8996a3; font-size: 0.74rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        margin-top: 0.1rem;
    }

    /* ---------- PROTOTYPE HEADER ---------- */
    .proto-header {
        display: flex; align-items: baseline; justify-content: space-between;
        flex-wrap: wrap; gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .proto-id { color: #e9eef3; font-size: 1.05rem; font-weight: 750; }
    .proto-aff {
        color: #a4aeb9; font-size: 0.82rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .proto-aff b { color: #e6edf4; font-weight: 700; }

    /* ---------- MUTED / CAPTION ---------- */
    .muted { color: #7a8692; font-size: 0.7rem; line-height: 1.45; margin-top: 0.5rem; }
    .compact-line {
        color: #96a1ac; font-size: 0.76rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        margin-top: 0.55rem;
    }
    .img-label {
        color: #8996a3; font-size: 0.64rem; font-weight: 750;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem;
    }

    /* ---------- EMPTY STATES ---------- */
    .empty {
        padding: 0.9rem 0.9rem;
        border-radius: 11px;
        border: 1px dashed rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.017);
        text-align: center;
    }
    .empty-t { color: #d6dee5; font-size: 0.8rem; font-weight: 700; }
    .empty-x { margin-top: 0.16rem; color: #77838f; font-size: 0.7rem; line-height: 1.4; }

    /* ---------- IMAGE SIZING (single source of truth) ---------- */
    [data-testid="stImage"] img {
        border-radius: 10px;
        object-fit: contain;
        max-height: 360px;
        width: auto;
        max-width: 100%;
        margin: 0 auto;
        display: block;
    }

    /* ---------- INPUTS ---------- */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.022) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
    }
    .stButton > button { border-radius: 8px; font-weight: 600; font-size: 0.8rem; }

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
                candidate.resolve().relative_to(PROTOTYPE_IMAGES_DIR.resolve())
                return None
            except ValueError:
                return candidate.resolve()

    return None


def find_demo_case_by_hash(file_hash):
    if demo_df.empty or "image_hash" not in demo_df.columns:
        return None
    matches = demo_df[demo_df["image_hash"].astype(str) == str(file_hash)]
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

    centroid = np.asarray(centroids[prototype_id], dtype=np.float32).reshape(-1)
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


def get_unique_prototype_images(prototype_id, exclude_hashes=None):
    candidates = []
    prototype_df = artifacts.get("prototype_images")

    if (
        prototype_df is not None
        and not prototype_df.empty
        and "prototype_id" in prototype_df.columns
    ):
        matches = prototype_df[prototype_df["prototype_id"].astype(str) == str(prototype_id)]
        for _, row in matches.iterrows():
            found = False
            for column in ("image_path", "representative_image", "image", "filename"):
                if column not in row.index:
                    continue
                value = row[column]
                if pd.isna(value):
                    continue
                value = str(value).strip()
                if not value:
                    continue
                for candidate in (ROOT / value, PROTOTYPE_IMAGES_DIR / value):
                    if candidate.exists():
                        candidates.append(candidate)
                        found = True
                        break
                if found:
                    break

    if not candidates:
        try:
            folder = PROTOTYPE_IMAGES_DIR / f"P{int(prototype_id):02d}"
        except Exception:
            folder = None
        if folder is not None and folder.exists():
            candidates = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
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
    debug = {"db_shape": None, "candidates_scanned": 0, "exact_embedding_duplicates": 0}

    embeddings = np.asarray(artifacts["visual_embeddings"], dtype=np.float32)
    valid_indices = np.asarray(artifacts["valid_indices"])
    cluster_labels = np.asarray(artifacts["cluster_labels"])

    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

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

        if exclude_dataset_index is not None and dataset_index == int(exclude_dataset_index):
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

        embeddings = np.asarray(artifacts["visual_embeddings"], dtype=np.float32)
        if embedding_row < 0 or embedding_row >= embeddings.shape[0]:
            st.session_state.error = f"Embedding row {embedding_row} is out of range."
            return False

        embedding = np.asarray(embeddings[embedding_row], dtype=np.float32)

        answer = (
            demo_case.get("answer")
            or demo_case.get("analysis")
            or demo_case.get("initial_analysis")
            or "No observation is available for this case."
        )

        try:
            prototype_id = int(demo_case["prototype_id"])
        except Exception:
            prototype_id = None

        prototype_similarity = (
            calculate_prototype_affinity(embedding, prototype_id)
            if prototype_id is not None else None
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

        similar_hashes = {r["image_hash"] for r in similar_cases if r.get("image_hash")}

        prototype_images = (
            get_unique_prototype_images(prototype_id, exclude_hashes=similar_hashes)
            if prototype_id is not None else []
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
                float(similar_cases[0]["similarity"]) if similar_cases else None
            ),
            "unique_retrieved": debug.get("unique_retrieved", 0),
            "requested": debug.get("requested", MAX_SIMILAR_CASES),
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
        "prototype_similarity": st.session_state.prototype_similarity,
        "analysis": st.session_state.analysis,
        "answer": st.session_state.analysis,
        "observation": st.session_state.analysis,
        "qa": parse_qa(
            st.session_state.demo_case.get("qa")
        ),
        "retrieval_results": list(
            st.session_state.retrieval_results or []
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
    """
    Answer questions using only the prepared evidence available for the selected case.
    """

    q = question.lower().strip()

    # --------------------------------------------------------
    # 1. WHAT DOES THE MODEL SEE?
    # --------------------------------------------------------
    if (
        "what does the model see" in q
        or "model see" in q
        or "observation" in q
        or "finding" in q and "where" not in q
    ):
        observation = (
            context.get("analysis")
            or context.get("answer")
            or context.get("observation")
        )

        if observation:
            return f'**MedGemma observation:** "{observation}"'

        return "No model observation is available for this case."

    # --------------------------------------------------------
    # 2. WHERE IS THE FINDING?
    # --------------------------------------------------------
    if (
        "where is the finding" in q
        or "where" in q and "finding" in q
        or "location" in q
        or "located" in q
    ):
        observation = (
            context.get("analysis")
            or context.get("answer")
            or context.get("observation")
        )

        if observation:
            return (
                f'**Finding location:** The prepared MedGemma observation states: '
                f'"{observation}"'
            )

        return "The finding location is not available in the prepared evidence."

    # --------------------------------------------------------
    # 3. WHY THIS PROTOTYPE?
    # --------------------------------------------------------
    if (
        "why this prototype" in q
        or "why this prototype?" in q
        or "prototype" in q and "why" in q
    ):
        prototype_id = context.get("prototype_id")
        affinity = context.get("prototype_similarity")

        if prototype_id is not None and affinity is not None:
            return (
                f"**Prototype P{int(prototype_id):02d}** was selected because "
                f"the image has a visual prototype affinity of **{float(affinity):.3f}**. "
                f"This indicates that its visual embedding is closely associated "
                f"with this learned prototype."
            )

        return "Prototype information is not available for this case."

    # --------------------------------------------------------
    # 4. WHAT ARE THE SIMILAR CASES?
    # --------------------------------------------------------
    if (
        "similar cases" in q
        or "similar case" in q
        or "nearest cases" in q
        or "similar images" in q
    ):
        results = context.get("retrieval_results", [])

        if results:
            lines = ["**Most similar reference cases:**"]

            for r in results[:3]:
                case_id = r.get("dataset_index", "Unknown")
                prototype = r.get("prototype_id")

                similarity = r.get("similarity")

                if similarity is not None:
                    lines.append(
                        f"- Case {case_id} · P{int(prototype):02d} · "
                        f"similarity **{float(similarity):.3f}**"
                    )
                else:
                    lines.append(
                        f"- Case {case_id} · P{int(prototype):02d}"
                    )

            return "\n".join(lines)

        return "No similar reference cases are available."

    # --------------------------------------------------------
    # 5. HOW STRONG IS THE VISUAL MATCH?
    # --------------------------------------------------------
    if (
        "how strong" in q
        or "visual match" in q
        or "similarity" in q
        or "match strength" in q
    ):
        affinity = context.get("prototype_similarity")
        results = context.get("retrieval_results", [])

        response = []

        if affinity is not None:
            response.append(
                f"The image has a prototype affinity of **{float(affinity):.3f}**."
            )

        if results:
            similarities = [
                r.get("similarity")
                for r in results[:3]
                if r.get("similarity") is not None
            ]

            if similarities:
                formatted = ", ".join(
                    f"{float(s):.3f}" for s in similarities
                )
                response.append(
                    f"The three nearest reference cases have similarities of "
                    f"**{formatted}**."
                )

        if response:
            return " ".join(response)

        return "Visual-match strength is not available for this case."

    # --------------------------------------------------------
    # 6. DIAGNOSIS
    # --------------------------------------------------------
    if (
        "diagnosis" in q
        or "diagnose" in q
        or "clinical" in q
    ):
        return (
            "No. This demo provides a model-generated observation and visual "
            "evidence, not a clinical diagnosis."
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    return (
        "I can answer questions about the model observation, finding location, "
        "prototype assignment, visual similarity, and retrieved reference cases."
    )


# ============================================================
# FINDING-GUIDED VISUALIZATION
# ============================================================

def generate_finding_guided_heatmap(image, observation=""):
    """
    Generate a finding-guided visualization from the image and the
    precomputed model observation.

    This is NOT Grad-CAM, does not use MedGemma gradients, and does not
    perform clinical lesion segmentation. It combines anatomical-region
    heuristics, image intensity/contrast, and spatial priors into a
    research-demo visual aid only.
    """

    original = image.convert("RGB")
    max_size = 768
    if max(original.size) > max_size:
        scale = max_size / max(original.size)
        new_size = (int(original.width * scale), int(original.height * scale))
        original = original.resize(new_size, Image.Resampling.LANCZOS)

    gray = original.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    h, w = arr.shape

    norm = (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else np.zeros_like(arr)

    blurred = gray.filter(ImageFilter.GaussianBlur(radius=max(8, int(w * 0.025))))
    smooth = np.asarray(blurred, dtype=np.float32) / 255.0
    local_contrast = np.abs(norm - smooth)

    gx = np.zeros_like(norm)
    gy = np.zeros_like(norm)
    gx[:, 1:-1] = np.abs(norm[:, 2:] - norm[:, :-2])
    gy[1:-1, :] = np.abs(norm[2:, :] - norm[:-2, :])
    variation = np.sqrt(gx ** 2 + gy ** 2)

    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / max(w - 1, 1)
    y = yy / max(h - 1, 1)

    text = str(observation or "").lower()
    center_x, center_y = 0.50, 0.50
    spread_x, spread_y = 0.30, 0.28

    if any(t in text for t in (
        "periventricular", "white matter", "hyperintensit", "infarct",
        "infarcted", "ischemi", "lesion", "ischemic",
    )):
        center_x, center_y = 0.50, 0.50
        spread_x, spread_y = 0.25, 0.22

    if "frontal" in text:
        center_x, center_y = 0.50, 0.30
        spread_x, spread_y = 0.27, 0.18
    elif "parietal" in text:
        center_x, center_y = 0.50, 0.43
        spread_x, spread_y = 0.28, 0.18
    elif "temporal" in text:
        center_x, center_y = 0.50, 0.62
        spread_x, spread_y = 0.30, 0.20
    elif "occipital" in text:
        center_x, center_y = 0.50, 0.76
        spread_x, spread_y = 0.27, 0.17

    if re.search(r"\bleft\b|\bleft-sided\b|\bleft side\b", text):
        center_x = 0.35
    elif re.search(r"\bright\b|\bright-sided\b|\bright side\b", text):
        center_x = 0.65

    spatial_prior = np.exp(
        -(((x - center_x) ** 2 / (2 * spread_x ** 2)) + ((y - center_y) ** 2 / (2 * spread_y ** 2)))
    )

    cx, cy, rx, ry = 0.50, 0.50, 0.43, 0.45
    brain_mask = (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0).astype(np.float32)
    mask_img = Image.fromarray((brain_mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=10)
    )
    brain_mask = np.asarray(mask_img, dtype=np.float32) / 255.0

    finding_signal = (0.65 * spatial_prior + 0.25 * local_contrast + 0.10 * variation)
    finding_signal *= brain_mask

    valid_mask = brain_mask > 0.15
    if valid_mask.any():
        threshold = np.percentile(finding_signal[valid_mask], 35)
        finding_signal[finding_signal < threshold] = 0

    signal_img = Image.fromarray(np.clip(finding_signal * 255, 0, 255).astype(np.uint8))
    signal_img = signal_img.filter(ImageFilter.GaussianBlur(radius=12))
    finding_signal = np.asarray(signal_img, dtype=np.float32) / 255.0

    valid = finding_signal[valid_mask] if valid_mask.any() else finding_signal
    if valid.size > 0:
        low = np.percentile(valid, 20)
        high = np.percentile(valid, 98)
        if high > low:
            finding_signal = np.clip((finding_signal - low) / (high - low), 0, 1)
    finding_signal *= brain_mask

    hmap = np.clip(finding_signal, 0, 1)
    r = np.clip(2.0 * hmap - 0.5, 0, 1)
    g = np.clip(2.0 * hmap, 0, 1)
    b = np.clip(1.5 - 2.0 * hmap, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    heatmap = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")

    overlay = Image.blend(original.convert("RGB"), heatmap, alpha=0.40)
    return heatmap, overlay


# ============================================================
# HERO
# ============================================================

render_html(
    f"""
    <div class="hero">
        <div class="hero-title">{html.escape(APP_NAME)}</div>
        <div class="hero-sub">{html.escape(APP_SUBTITLE)}</div>
        <div class="hero-row">
            <span class="badge badge-blue">MedGemma</span>
            <span class="badge">Visual Retrieval</span>
            <span class="badge">Prototype Evidence</span>
        </div>
    </div>
    """
)


# ============================================================
# 01 — MEDICAL IMAGE
# ============================================================

section_header("01", "Medical Image")

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

case = None

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        current_hash = calculate_hash(file_bytes)

        if st.session_state.image_hash != current_hash:
            reset_analysis()
            st.session_state.image_hash = current_hash
            demo_case = find_demo_case_by_hash(current_hash)
            if demo_case is not None:
                run_precomputed_case(demo_case, query_image_hash=current_hash)
            else:
                st.session_state.error = (
                    "This image is not part of the prepared demonstration set."
                )

        image = Image.open(uploaded_file).convert("RGB")
        case = st.session_state.demo_case

        center_l, center_m, center_r = st.columns([1, 2, 1])
        with center_m:
            st.image(image, use_container_width=True)

        if st.session_state.error:
            st.warning(st.session_state.error)

    except Exception as exc:
        st.error(f"Unable to process image: {exc}")
else:
    empty_state("Upload a medical image to begin")


# ============================================================
# 02 — MODEL FINDING
# ============================================================

section_header("02", "Model Finding")

if st.session_state.analysis:
    render_html(
        f"""
        <div class="obs-card">
            <div class="obs-label">MedGemma observation</div>
            <div class="obs-text">{html.escape(str(st.session_state.analysis))}</div>
        </div>
        """
    )
else:
    empty_state("No finding yet", "Upload a demonstration image to see the model finding.")


# ============================================================
# 03 — SIMILAR CASES
# ============================================================

section_header("03", "Similar Cases", "Visually similar reference cases")

results = st.session_state.retrieval_results

if results:
    columns = st.columns(len(results), gap="medium")
    for column, result in zip(columns, results):
        with column:
            image_path = result.get("image_path")
            if image_path is not None and Path(image_path).exists():
                st.image(str(image_path), use_container_width=True)
            else:
                empty_state("Image unavailable")

            render_html(
                f"""
                <div class="case-meta">
                    <div class="case-title">Case {int(result["dataset_index"])}</div>
                    <div class="case-sub">
                        P{int(result["prototype_id"]):02d} · {float(result["similarity"]):.5f}
                    </div>
                </div>
                """
            )
else:
    empty_state(
        "No retrieved cases yet",
        "Upload a prepared demonstration image to retrieve similar reference cases.",
    )


# ============================================================
# 04 — PROTOTYPE EVIDENCE
# ============================================================

section_header("04", "Prototype Evidence")

prototype_id = st.session_state.prototype_id

if prototype_id is not None:
    score = st.session_state.prototype_similarity
    score_text = f"{float(score):.5f}" if score is not None else "—"

    render_html(
        f"""
        <div class="proto-header">
            <div class="proto-id">P{int(prototype_id):02d}</div>
            <div class="proto-aff">Affinity <b>{score_text}</b></div>
        </div>
        """
    )

    prototype_images = st.session_state.prototype_images or []

    if prototype_images:
        columns = st.columns(len(prototype_images), gap="medium")
        for column, image_path in zip(columns, prototype_images):
            with column:
                st.image(str(image_path), use_container_width=True)
    else:
        empty_state(
            "Representative images unavailable",
            "The prototype was identified, but no unique representative image could be resolved.",
        )
else:
    empty_state(
        "Prototype not available yet",
        "Upload a prepared demonstration image to identify its learned visual prototype.",
    )


# ============================================================
# 05 — EXPLAINABILITY
# ============================================================

section_header("05", "Explainability")

if case is None:
    empty_state(
        "Case-specific explanation not available yet",
        "Upload a prepared demonstration image to see visual evidence.",
    )
else:
    observation_text = st.session_state.analysis or "No observation available."

    try:
        dataset_val = int(case.get("dataset_index"))
    except Exception:
        dataset_val = -1

    current_image = None
    try:
        image_path = None
        if dataset_val >= 0:
            for ext in IMAGE_EXTENSIONS:
                candidate = DEMO_IMAGES_DIR / f"case_{dataset_val:05d}_train{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
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

    if current_image is not None:
        try:
            _, overlay_image = generate_finding_guided_heatmap(current_image, observation_text)

            left, right = st.columns([1, 1], gap="large")
            with left:
                render_html('<div class="img-label">Original</div>')
                st.image(current_image, use_container_width=True)
            with right:
                render_html('<div class="img-label">Finding-guided view</div>')
                st.image(overlay_image, use_container_width=True)

            aff_text = (
                f"{float(st.session_state.prototype_similarity):.5f}"
                if st.session_state.prototype_similarity is not None else "—"
            )
            proto_text = f"P{int(prototype_id):02d}" if prototype_id is not None else "—"
            render_html(
                f"""
                <div class="compact-line">
                    Prototype {html.escape(proto_text)} · Affinity {html.escape(aff_text)} ·
                    {len(results)} retrieved cases
                </div>
                <div class="muted">Research visualization; not a clinical localization.</div>
                """
            )
        except Exception:
            empty_state("Finding-guided visualization could not be generated")
    else:
        empty_state("Original image unavailable for visual evidence")


# ============================================================
# 06 — ASK ABOUT THIS IMAGE
# ============================================================

section_header("06", "Ask About This Image", "Explore the finding and visual evidence")

if case is None:
    empty_state(
        "Select a prepared image first",
        "The assistant answers questions supported by the prepared evidence for the selected case.",
    )
else:
    chat_context = build_case_context()

    quick_questions = [
        "What does the model see?",
        "Where is the finding?",
        "Why this prototype?",
        "What are the similar cases?",
    ]

    qcols = st.columns(len(quick_questions))
    for qcol, qtext in zip(qcols, quick_questions):
        with qcol:
            if st.button(
                qtext,
                key=f"q_{qtext}_{st.session_state.image_hash or 'x'}",
                use_container_width=True,
            ):
                st.session_state.chat_history.append({"role": "user", "content": qtext})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer_user_question(qtext, chat_context)}
                )
                st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_TURNS:]
                st.rerun()

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.caption("Ask a question about this case.")
    
    user_prompt = st.chat_input("Ask a question about this image...")



    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        assistant_reply = answer_user_question(user_prompt, chat_context)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_TURNS:]
        st.rerun()


