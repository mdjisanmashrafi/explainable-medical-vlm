import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.retrieval import (
    load_retrieval_artifacts,
    cosine_similarity,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent

MAX_SIMILAR_CASES = 3
MAX_CHAT_CONTEXT_TURNS = 3


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Explainable Medical VLM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .project-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .project-subtitle {
        color: #9aa4b2;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .prototype-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .small-text {
        color: #8b949e;
        font-size: 0.9rem;
    }

    .warning-text {
        color: #d29922;
    }

    .success-text {
        color: #3fb950;
    }

    .metric-label {
        color: #8b949e;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

DEFAULT_STATE = {
    "image_hash": None,
    "initial_analysis": None,
    "chat_history": [],
    "query_embedding": None,
    "retrieval_results": [],
    "recommended_prototype": None,
    "prototype_similarity": None,
    "last_error": None,
    "demo_case": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

@st.cache_resource
def load_artifacts():
    return load_retrieval_artifacts()


try:
    artifacts = load_artifacts()

    visual_embeddings = artifacts["visual_embeddings"]
    valid_indices = artifacts["valid_indices"]
    cluster_labels = artifacts["cluster_labels"]
    cluster_centroids = artifacts["cluster_centroids"]
    prototype_summary = artifacts["prototype_summary"]
    prototype_images = artifacts["prototype_images"]

except Exception as exc:
    st.error("Failed to load retrieval artifacts.")
    st.exception(exc)
    st.stop()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_image_hash(image: Image.Image) -> str:
    """Return a stable SHA256 hash for an image."""
    image = image.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        image.save(tmp.name, format="PNG")
        tmp.seek(0)
        data = tmp.read()

    return hashlib.sha256(data).hexdigest()


def cosine_similarity_single(query, matrix):
    """Safe cosine similarity."""
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query={query.shape[0]}, "
            f"matrix={matrix.shape[1]}"
        )

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    matrix_norm = np.linalg.norm(matrix, axis=1)

    denominator = query_norm * matrix_norm
    denominator[denominator == 0] = 1e-12

    return np.dot(matrix, query) / denominator


def get_prototype_similarity(query_embedding, prototype_id):
    """
    Calculate query-to-prototype similarity.

    Handles two possible cases:
    1. Centroids are already in 1152-D embedding space.
    2. Centroids are in PCA space.
    """

    if prototype_id is None:
        return None

    centroid = np.asarray(
        cluster_centroids[int(prototype_id)],
        dtype=np.float32,
    ).reshape(-1)

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    # -------------------------------------------------------------------------
    # Case 1: centroid is already in original embedding space
    # -------------------------------------------------------------------------

    if centroid.shape[0] == query.shape[0]:
        return float(cosine_similarity_single(query, centroid.reshape(1, -1))[0])

    # -------------------------------------------------------------------------
    # Case 2: centroid is in PCA space
    # -------------------------------------------------------------------------

    try:
        pca_scaler = artifacts["pca_scaler"]
        pca_model = artifacts["pca_model"]

        query_scaled = pca_scaler.transform(
            query.reshape(1, -1)
        )

        query_pca = pca_model.transform(query_scaled)

        if query_pca.shape[1] == centroid.shape[0]:
            return float(
                cosine_similarity_single(
                    query_pca[0],
                    centroid.reshape(1, -1),
                )[0]
            )

    except Exception:
        pass

    return None


def find_top_unique_similar_cases(
    query_embedding,
    top_k=3,
):
    """
    Retrieve the most similar unique reference images.

    Duplicate dataset indices are removed.
    """

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    embeddings = np.asarray(
        visual_embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D visual embeddings, got shape {embeddings.shape}"
        )

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"query={query_embedding.shape[0]}, "
            f"database={embeddings.shape[1]}"
        )

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    sorted_rows = np.argsort(scores)[::-1]

    results = []
    seen_dataset_indices = set()

    for row_index in sorted_rows:

        dataset_index = int(valid_indices[row_index])

        if dataset_index in seen_dataset_indices:
            continue

        seen_dataset_indices.add(dataset_index)

        prototype_id = int(cluster_labels[row_index])

        results.append(
            {
                "rank": len(results) + 1,
                "embedding_row": int(row_index),
                "dataset_index": dataset_index,
                "prototype_id": prototype_id,
                "similarity": float(scores[row_index]),
            }
        )

        if len(results) >= top_k:
            break

    return results


def get_prototype_image_paths(prototype_id):
    """
    Return available representative images for a prototype.
    """

    if prototype_id is None:
        return []

    prototype_name = f"P{int(prototype_id):02d}"

    prototype_dir = ROOT / "representative_images" / prototype_name

    if not prototype_dir.exists():
        return []

    paths = sorted(
        prototype_dir.glob("*.png")
    )

    return paths


def get_representative_image_for_result(result):
    """
    Resolve an image for a retrieval result.

    First tries the prototype_representative_images.csv mapping.
    Then falls back to the prototype representative folder.
    """

    prototype_id = result.get("prototype_id")

    # -------------------------------------------------------------------------
    # Try CSV mapping
    # -------------------------------------------------------------------------

    if prototype_images is not None and len(prototype_images) > 0:

        try:

            dataset_index = int(result["dataset_index"])

            possible_columns = [
                "dataset_index",
                "valid_index",
                "image_index",
                "index",
            ]

            matching_column = None

            for column in possible_columns:
                if column in prototype_images.columns:
                    matching_column = column
                    break

            if matching_column is not None:

                rows = prototype_images[
                    prototype_images[matching_column] == dataset_index
                ]

                if len(rows) > 0:

                    row = rows.iloc[0]

                    path_columns = [
                        "image_path",
                        "representative_image",
                        "path",
                        "file_path",
                        "filename",
                    ]

                    for column in path_columns:

                        if column in rows.columns:

                            value = row[column]

                            if pd.notna(value):

                                candidate = ROOT / str(value)

                                if candidate.exists():
                                    return candidate

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Fallback: representative image from prototype folder
    # -------------------------------------------------------------------------

    paths = get_prototype_image_paths(prototype_id)

    if paths:
        return paths[0]

    return None


def prototype_description(prototype_id):
    """Get prototype description from prototype_summary.csv."""

    if prototype_id is None:
        return "No prototype selected."

    if prototype_summary is None:
        return "Learned visual prototype."

    try:

        rows = prototype_summary[
            prototype_summary["prototype_id"] == int(prototype_id)
        ]

        if len(rows) == 0:
            return "Learned visual prototype."

        row = rows.iloc[0]

        possible_columns = [
            "description",
            "prototype_description",
            "summary",
            "dominant_pattern",
            "pattern",
        ]

        for column in possible_columns:

            if column in rows.columns:

                value = row[column]

                if pd.notna(value):
                    return str(value)

    except Exception:
        pass

    return "Recurring visual pattern in the learned image representation."


def reset_image_state():
    """Reset all image-dependent state."""

    st.session_state["initial_analysis"] = None
    st.session_state["chat_history"] = []
    st.session_state["query_embedding"] = None
    st.session_state["retrieval_results"] = []
    st.session_state["recommended_prototype"] = None
    st.session_state["prototype_similarity"] = None
    st.session_state["last_error"] = None
    st.session_state["demo_case"] = None


# =============================================================================
# DEMO DATA SUPPORT
# =============================================================================

def find_demo_metadata():
    """
    Look for optional precomputed demo metadata.

    Supported filenames:
        demo_cases.csv
        demo_cases.json
        precomputed_cases.csv
        precomputed_cases.json
    """

    candidates = [
        ROOT / "demo_cases.csv",
        ROOT / "demo_cases.json",
        ROOT / "precomputed_cases.csv",
        ROOT / "precomputed_cases.json",
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def load_demo_metadata():
    path = find_demo_metadata()

    if path is None:
        return None

    try:

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        if path.suffix.lower() == ".json":

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return pd.DataFrame(data)

            if isinstance(data, dict):
                return pd.DataFrame([data])

    except Exception:
        return None

    return None


@st.cache_data
def get_demo_metadata():
    return load_demo_metadata()


def match_demo_image(image):
    """
    Match uploaded image against optional precomputed demo cases.

    Matching is based on SHA256 hash.
    """

    demo_df = get_demo_metadata()

    if demo_df is None or len(demo_df) == 0:
        return None

    current_hash = get_image_hash(image)

    possible_hash_columns = [
        "image_hash",
        "sha256",
        "hash",
    ]

    hash_column = None

    for column in possible_hash_columns:

        if column in demo_df.columns:
            hash_column = column
            break

    if hash_column is None:
        return None

    rows = demo_df[
        demo_df[hash_column].astype(str) == current_hash
    ]

    if len(rows) == 0:
        return None

    return rows.iloc[0].to_dict()


# =============================================================================
# PRECOMPUTED DEMO PIPELINE
# =============================================================================

def run_precomputed_demo(image):
    """
    Use precomputed demo metadata when available.

    This function does NOT call Hugging Face or ZeroGPU.
    """

    demo_case = match_demo_image(image)

    if demo_case is None:
        raise RuntimeError(
            "This image is not registered as a precomputed demo case. "
            "For the free public version, new images must first be processed "
            "in Kaggle and added to the demo artifacts."
        )

    # -------------------------------------------------------------------------
    # Load precomputed answer
    # -------------------------------------------------------------------------

    answer = demo_case.get(
        "analysis",
        demo_case.get(
            "answer",
            demo_case.get(
                "initial_analysis",
                "No precomputed MedGemma analysis is available.",
            ),
        ),
    )

    # -------------------------------------------------------------------------
    # Load precomputed embedding
    # -------------------------------------------------------------------------

    embedding = None

    embedding_value = demo_case.get(
        "embedding",
        demo_case.get("query_embedding"),
    )

    if embedding_value is not None:

        try:

            if isinstance(embedding_value, str):
                embedding_value = json.loads(embedding_value)

            embedding = np.asarray(
                embedding_value,
                dtype=np.float32,
            ).reshape(-1)

        except Exception:
            embedding = None

    # -------------------------------------------------------------------------
    # Load precomputed retrieval
    # -------------------------------------------------------------------------

    retrieval_results = []

    retrieval_value = demo_case.get(
        "retrieval_results"
    )

    if retrieval_value is not None:

        try:

            if isinstance(retrieval_value, str):
                retrieval_value = json.loads(retrieval_value)

            retrieval_results = retrieval_value

        except Exception:
            retrieval_results = []

    # -------------------------------------------------------------------------
    # Alternative: reconstruct retrieval from embedding
    # -------------------------------------------------------------------------

    if embedding is not None and not retrieval_results:

        retrieval_results = find_top_unique_similar_cases(
            embedding,
            top_k=MAX_SIMILAR_CASES,
        )

    # -------------------------------------------------------------------------
    # Prototype
    # -------------------------------------------------------------------------

    prototype_id = demo_case.get(
        "prototype_id"
    )

    if prototype_id is None and retrieval_results:

        prototype_id = retrieval_results[0].get(
            "prototype_id"
        )

    if prototype_id is not None:

        try:
            prototype_id = int(prototype_id)
        except Exception:
            prototype_id = None

    prototype_similarity = None

    if embedding is not None and prototype_id is not None:

        prototype_similarity = get_prototype_similarity(
            embedding,
            prototype_id,
        )

    return {
        "answer": str(answer),
        "embedding": embedding,
        "retrieval_results": retrieval_results,
        "prototype_id": prototype_id,
        "prototype_similarity": prototype_similarity,
        "demo_case": demo_case,
    }


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="project-title">🩺 Explainable Medical VLM</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="project-subtitle">'
    "MedGemma · Visual Retrieval · Prototype Reasoning"
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# ARTIFACT STATUS
# =============================================================================

with st.expander("System information", expanded=False):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Reference embeddings",
            f"{len(visual_embeddings):,}",
        )

    with col2:
        st.metric(
            "Embedding dimension",
            f"{visual_embeddings.shape[1]}",
        )

    with col3:
        st.metric(
            "Visual prototypes",
            f"{len(cluster_centroids)}",
        )

    with col4:
        st.metric(
            "Representative images",
            "90",
        )

    st.caption(
        "Retrieval artifacts are loaded locally from the repository. "
        "No live HF ZeroGPU embedding call is required."
    )


# =============================================================================
# 01 MEDICAL IMAGE
# =============================================================================

st.markdown(
    '<div class="section-title">01 Medical Image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    current_hash = get_image_hash(image)

    if (
        st.session_state["image_hash"] is not None
        and st.session_state["image_hash"] != current_hash
    ):
        reset_image_state()

    st.session_state["image_hash"] = current_hash

    col_image, col_info = st.columns(
        [1.15, 0.85]
    )

    with col_image:

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with col_info:

        st.markdown(
            """
            <div class="info-card">
            <b>Image loaded successfully.</b><br><br>
            The public demo uses precomputed MedGemma results and
            visual-retrieval artifacts so that the application does not
            depend on a paid inference endpoint or HF ZeroGPU quota.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"SHA256: {current_hash[:16]}..."
        )


# =============================================================================
# 02 MODEL ANALYSIS
# =============================================================================

st.markdown(
    '<div class="section-title">02 Model Analysis</div>',
    unsafe_allow_html=True,
)

if uploaded_file is None:

    st.info(
        "Upload a prepared demo image to run the analysis."
    )

else:

    if st.session_state["initial_analysis"] is None:

        if st.button(
            "Run Full Analysis",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Loading precomputed analysis and retrieval..."
                ):

                    result = run_precomputed_demo(
                        image
                    )

                st.session_state["initial_analysis"] = (
                    result["answer"]
                )

                st.session_state["query_embedding"] = (
                    result["embedding"]
                )

                st.session_state["retrieval_results"] = (
                    result["retrieval_results"]
                )

                st.session_state["recommended_prototype"] = (
                    result["prototype_id"]
                )

                st.session_state["prototype_similarity"] = (
                    result["prototype_similarity"]
                )

                st.session_state["demo_case"] = (
                    result["demo_case"]
                )

                st.session_state["chat_history"] = [
                    {
                        "role": "user",
                        "content": (
                            "What findings are visible in this image?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": result["answer"],
                    },
                ]

                st.rerun()

            except Exception as exc:

                st.session_state["last_error"] = str(
                    exc
                )

    if st.session_state["last_error"]:

        st.warning(
            st.session_state["last_error"]
        )

        st.caption(
            "This is expected for an image that has not yet been "
            "processed as a precomputed demo case."
        )

    if st.session_state["initial_analysis"]:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**MedGemma image-grounded analysis**"
        )

        st.write(
            st.session_state["initial_analysis"]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# 03 SIMILAR CASES
# =============================================================================

st.markdown(
    '<div class="section-title">03 Similar Cases</div>',
    unsafe_allow_html=True,
)

retrieval_results = st.session_state.get(
    "retrieval_results",
    [],
)

if retrieval_results:

    displayed_hashes = set()

    display_results = []

    for result in retrieval_results:

        image_path = get_representative_image_for_result(
            result
        )

        if image_path is None:
            continue

        try:

            image_bytes = image_path.read_bytes()

            image_hash = hashlib.sha256(
                image_bytes
            ).hexdigest()

            if image_hash in displayed_hashes:
                continue

            displayed_hashes.add(image_hash)

            display_results.append(
                (
                    result,
                    image_path,
                )
            )

        except Exception:
            continue

        if len(display_results) >= MAX_SIMILAR_CASES:
            break

    if display_results:

        columns = st.columns(
            len(display_results)
        )

        for column, item in zip(
            columns,
            display_results,
        ):

            result, image_path = item

            with column:

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

                st.markdown(
                    f"**Rank {result['rank']} · "
                    f"Prototype P{result['prototype_id']:02d}**"
                )

                st.metric(
                    "Visual similarity",
                    f"{result['similarity'] * 100:.1f}%",
                )

                st.caption(
                    f"Reference index: "
                    f"{result['dataset_index']}"
                )

    else:

        st.info(
            "Retrieval results were found, but representative "
            "images could not be resolved."
        )

else:

    st.info(
        "Run the full analysis to retrieve visually similar cases."
    )


# =============================================================================
# 04 PROTOTYPE EXPLANATION
# =============================================================================

st.markdown(
    '<div class="section-title">04 Prototype Explanation</div>',
    unsafe_allow_html=True,
)

recommended_prototype = st.session_state.get(
    "recommended_prototype"
)

prototype_similarity = st.session_state.get(
    "prototype_similarity"
)

if recommended_prototype is not None:

    prototype_name = (
        f"P{int(recommended_prototype):02d}"
    )

    description = prototype_description(
        recommended_prototype
    )

    col_a, col_b = st.columns(
        [1.2, 0.8]
    )

    with col_a:

        st.markdown(
            '<div class="prototype-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"### Visual Prototype {prototype_name}"
        )

        st.write(
            description
        )

        st.caption(
            "This prototype represents a recurring pattern in the "
            "learned visual embedding space. It is not a clinically "
            "validated diagnostic concept."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with col_b:

        if prototype_similarity is not None:

            st.metric(
                "Prototype Similarity",
                f"{prototype_similarity * 100:.1f}%",
            )

        else:

            st.metric(
                "Prototype Similarity",
                "N/A",
            )

        st.caption(
            "Query-to-prototype similarity"
        )

        representative_paths = (
            get_prototype_image_paths(
                recommended_prototype
            )
        )

        if representative_paths:

            st.image(
                str(representative_paths[0]),
                caption=(
                    f"Representative image · "
                    f"{prototype_name}"
                ),
                use_container_width=True,
            )

else:

    st.info(
        "The recommended prototype will appear after retrieval."
    )


# =============================================================================
# 05 EXPLAINABILITY
# =============================================================================

st.markdown(
    '<div class="section-title">05 Explainability</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">

    <b>Prototype-Grounded Visual Evidence</b>

    <br><br>

    The system extracts a visual representation of the medical image
    and compares it with a reference medical image database.

    <br><br>

    Visually similar reference cases are grouped into learned visual
    prototypes. The highest-ranked prototype provides an interpretable
    evidence layer that contextualizes the model-generated observation.

    <br><br>

    <b>Important limitation:</b>

    Prototype similarity represents visual association in the learned
    representation space. It does not establish a causal medical
    relationship, clinical validity, or diagnostic certainty.

    <br><br>

    The retrieved prototype also does <b>not</b> prove that MedGemma
    used that prototype when generating its response.

    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 06 ASK ABOUT THIS IMAGE
# =============================================================================

st.markdown(
    '<div class="section-title">06 Ask About This Image</div>',
    unsafe_allow_html=True,
)

if uploaded_file is not None:

    question = st.text_input(
        "Question",
        value="What findings are visible in this image?",
        key="image_question",
    )

    if st.button(
        "Ask",
        use_container_width=True,
    ):

        demo_case = st.session_state.get(
            "demo_case"
        )

        answer = None

        if demo_case is not None:

            # -------------------------------------------------------------
            # Try question-specific precomputed answers
            # -------------------------------------------------------------

            qa_json = demo_case.get(
                "qa",
                demo_case.get(
                    "questions",
                    demo_case.get(
                        "precomputed_qa"
                    ),
                ),
            )

            if qa_json is not None:

                try:

                    if isinstance(qa_json, str):
                        qa_json = json.loads(
                            qa_json
                        )

                    if isinstance(qa_json, dict):

                        normalized_question = (
                            question.strip().lower()
                        )

                        for q, a in qa_json.items():

                            if (
                                str(q).strip().lower()
                                == normalized_question
                            ):

                                answer = str(a)
                                break

                except Exception:
                    pass

            # -------------------------------------------------------------
            # Generic precomputed answer fallback
            # -------------------------------------------------------------

            if answer is None:

                answer = demo_case.get(
                    "answer",
                    demo_case.get(
                        "analysis",
                        demo_case.get(
                            "initial_analysis"
                        ),
                    ),
                )

        if answer is None:

            st.warning(
                "Question-specific precomputed answers are not available "
                "for this demo image."
            )

        else:

            st.session_state["chat_history"].append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": str(answer),
                }
            )

    # -------------------------------------------------------------------------
    # Display chat history
    # -------------------------------------------------------------------------

    history = st.session_state.get(
        "chat_history",
        [],
    )

    if history:

        for message in history:

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )

else:

    st.info(
        "Upload a medical image first."
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <br>
    <hr>

    <div style="text-align:center; color:#6e7681; font-size:0.85rem;">

    Explainable Medical VLM ·
    MedGemma + Visual Prototype Retrieval

    <br><br>

    Research prototype for interpretable medical image analysis.
    Model outputs are not clinical diagnoses.

    </div>
    """,
    unsafe_allow_html=True,
)
