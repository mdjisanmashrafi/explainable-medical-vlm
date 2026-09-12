
import hashlib
import json
from pathlib import Path

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
MAX_PROTOTYPE_IMAGES = 3


# =============================================================================
# PAGE CONFIGURATION
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

    /* ---------- Global ---------- */

    .stApp {
        background: #0b0f14;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* ---------- Header ---------- */

    .hero {
        padding: 0.5rem 0 1.8rem 0;
    }

    .hero-title {
        font-size: 2.55rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #9da7b3;
        font-size: 1.02rem;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 0.9rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #151b23;
        border: 1px solid #29313d;
        color: #aeb8c5;
        font-size: 0.78rem;
    }

    /* ---------- Sections ---------- */

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-top: 2.2rem;
        margin-bottom: 1rem;
    }

    .section-number {
        color: #8ea0b5;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .section-name {
        font-size: 1.32rem;
        font-weight: 700;
    }

    .section-description {
        color: #7f8a98;
        font-size: 0.9rem;
        margin-top: -0.55rem;
        margin-bottom: 1rem;
    }

    /* ---------- Cards ---------- */

    .card {
        background: #111720;
        border: 1px solid #252e3a;
        border-radius: 14px;
        padding: 1.2rem;
    }

    .card-soft {
        background: #0f151d;
        border: 1px solid #202936;
        border-radius: 14px;
        padding: 1.1rem;
    }

    .card-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .card-muted {
        color: #8c97a5;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    /* ---------- Status ---------- */

    .status-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.7rem;
    }

    .status {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.74rem;
        border: 1px solid #293442;
        color: #aeb9c7;
        background: #121923;
    }

    /* ---------- Similar cases ---------- */

    .case-label {
        margin-top: 0.7rem;
        font-size: 0.92rem;
        font-weight: 650;
    }

    .case-meta {
        color: #7f8a98;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    /* ---------- Prototype ---------- */

    .prototype-id {
        font-size: 1.55rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .prototype-description {
        color: #a5afbb;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ---------- Explainability ---------- */

    .evidence-box {
        background: #101720;
        border: 1px solid #293442;
        border-radius: 14px;
        padding: 1.25rem;
        line-height: 1.65;
    }

    .evidence-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .limitation {
        margin-top: 1rem;
        padding: 0.8rem 0.9rem;
        border-radius: 10px;
        background: #17150f;
        border: 1px solid #40351d;
        color: #c5b27a;
        font-size: 0.83rem;
        line-height: 1.55;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #697482;
        font-size: 0.78rem;
        line-height: 1.6;
        padding-top: 2.5rem;
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
# LOAD RETRIEVAL ARTIFACTS
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
# LOAD DEMO CASES
# =============================================================================

@st.cache_data
def load_demo_cases():
    path = ROOT / "demo_cases.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


demo_df = load_demo_cases()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def section_header(number, title, description=None):
    st.markdown(
        f"""
        <div class="section-header">
            <span class="section-number">{number}</span>
            <span class="section-name">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<div class="section-description">{description}</div>',
            unsafe_allow_html=True,
        )


def get_image_hash(image: Image.Image) -> str:
    image = image.convert("RGB")

    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return hashlib.sha256(
        buffer.getvalue()
    ).hexdigest()


def cosine_similarity_single(query, matrix):
    query = np.asarray(
        query,
        dtype=np.float32,
    ).reshape(-1)

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query={query.shape[0]}, "
            f"matrix={matrix.shape[1]}"
        )

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        return np.zeros(
            matrix.shape[0],
            dtype=np.float32,
        )

    matrix_norm = np.linalg.norm(
        matrix,
        axis=1,
    )

    denominator = (
        query_norm * matrix_norm
    )

    denominator[
        denominator == 0
    ] = 1e-12

    return np.dot(
        matrix,
        query,
    ) / denominator


def get_prototype_similarity(
    query_embedding,
    prototype_id,
):
    if prototype_id is None:
        return None

    centroid = np.asarray(
        cluster_centroids[int(prototype_id)],
        dtype=np.float32,
    ).reshape(-1)

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    # Original embedding space
    if centroid.shape[0] == query.shape[0]:

        return float(
            cosine_similarity_single(
                query,
                centroid.reshape(1, -1),
            )[0]
        )

    # PCA space
    try:

        pca_scaler = artifacts["pca_scaler"]
        pca_model = artifacts["pca_model"]

        query_scaled = pca_scaler.transform(
            query.reshape(1, -1)
        )

        query_pca = pca_model.transform(
            query_scaled
        )

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
    exclude_dataset_index=None,
    top_k=3,
):
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    embeddings = np.asarray(
        visual_embeddings,
        dtype=np.float32,
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    sorted_rows = np.argsort(scores)[::-1]

    results = []
    seen_indices = set()

    for row_index in sorted_rows:

        dataset_index = int(
            valid_indices[row_index]
        )

        # Never retrieve the uploaded image itself
        if (
            exclude_dataset_index is not None
            and dataset_index == exclude_dataset_index
        ):
            continue

        if dataset_index in seen_indices:
            continue

        seen_indices.add(dataset_index)

        prototype_id = int(
            cluster_labels[row_index]
        )

        results.append(
            {
                "rank": len(results) + 1,
                "embedding_row": int(row_index),
                "dataset_index": dataset_index,
                "prototype_id": prototype_id,
                "similarity": float(
                    scores[row_index]
                ),
            }
        )

        if len(results) >= top_k:
            break

    return results


def get_demo_case_by_dataset_index(
    dataset_index,
):
    if demo_df.empty:
        return None

    rows = demo_df[
        demo_df["dataset_index"].astype(int)
        == int(dataset_index)
    ]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


def get_demo_image_path(dataset_index):
    """
    Resolve the exact image belonging to a retrieved dataset index.
    """

    case = get_demo_case_by_dataset_index(
        dataset_index
    )

    if case is None:
        return None

    filename = case.get(
        "image_filename"
    )

    if not filename or pd.isna(filename):
        return None

    candidate = (
        ROOT
        / "demo_images"
        / Path(str(filename)).name
    )

    if candidate.exists():
        return candidate

    # Optional fallback using stored relative path
    stored_path = case.get(
        "image_path"
    )

    if stored_path and not pd.isna(stored_path):

        candidate = (
            ROOT
            / Path(str(stored_path))
        )

        if candidate.exists():
            return candidate

    return None


def get_prototype_image_paths(
    prototype_id,
):
    if prototype_id is None:
        return []

    prototype_dir = (
        ROOT
        / "representative_images"
        / f"P{int(prototype_id):02d}"
    )

    if not prototype_dir.exists():
        return []

    return sorted(
        prototype_dir.glob("*.png")
    )[:MAX_PROTOTYPE_IMAGES]


def prototype_description(
    prototype_id,
):
    if prototype_id is None:
        return (
            "No visual prototype was selected."
        )

    if (
        prototype_summary is None
        or prototype_summary.empty
    ):
        return (
            "Recurring visual pattern in the "
            "learned image representation."
        )

    try:

        rows = prototype_summary[
            prototype_summary["prototype_id"].astype(int)
            == int(prototype_id)
        ]

        if rows.empty:
            return (
                "Recurring visual pattern in the "
                "learned image representation."
            )

        row = rows.iloc[0]

        for column in [
            "description",
            "prototype_description",
            "summary",
            "dominant_pattern",
            "pattern",
        ]:

            if column in rows.columns:

                value = row[column]

                if pd.notna(value):
                    return str(value)

    except Exception:
        pass

    return (
        "Recurring visual pattern in the "
        "learned image representation."
    )


def reset_image_state():
    st.session_state["initial_analysis"] = None
    st.session_state["chat_history"] = []
    st.session_state["query_embedding"] = None
    st.session_state["retrieval_results"] = []
    st.session_state["recommended_prototype"] = None
    st.session_state["prototype_similarity"] = None
    st.session_state["last_error"] = None
    st.session_state["demo_case"] = None


def match_demo_image(image):
    if demo_df.empty:
        return None

    current_hash = get_image_hash(
        image
    )

    if "image_hash" not in demo_df.columns:
        return None

    rows = demo_df[
        demo_df["image_hash"].astype(str)
        == current_hash
    ]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


# =============================================================================
# PRECOMPUTED ANALYSIS
# =============================================================================

def run_precomputed_demo(image):

    demo_case = match_demo_image(
        image
    )

    if demo_case is None:

        raise RuntimeError(
            "This image is not included in the "
            "precomputed public demo."
        )

    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------

    answer = demo_case.get(
        "answer",
        demo_case.get(
            "analysis",
            "No precomputed analysis is available.",
        ),
    )

    # ---------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------

    embedding_value = demo_case.get(
        "embedding"
    )

    embedding = None

    if embedding_value is not None:

        try:

            if isinstance(
                embedding_value,
                str,
            ):
                embedding_value = json.loads(
                    embedding_value
                )

            embedding = np.asarray(
                embedding_value,
                dtype=np.float32,
            ).reshape(-1)

        except Exception:
            embedding = None

    # ---------------------------------------------------------
    # Dataset index
    # ---------------------------------------------------------

    dataset_index = demo_case.get(
        "dataset_index"
    )

    if dataset_index is not None:

        try:
            dataset_index = int(
                dataset_index
            )
        except Exception:
            dataset_index = None

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    retrieval_results = []

    if embedding is not None:

        retrieval_results = (
            find_top_unique_similar_cases(
                embedding,
                exclude_dataset_index=dataset_index,
                top_k=MAX_SIMILAR_CASES,
            )
        )

    # ---------------------------------------------------------
    # Prototype
    # ---------------------------------------------------------

    prototype_id = demo_case.get(
        "prototype_id"
    )

    if prototype_id is None and retrieval_results:

        prototype_id = retrieval_results[0][
            "prototype_id"
        ]

    if prototype_id is not None:

        try:
            prototype_id = int(
                prototype_id
            )
        except Exception:
            prototype_id = None

    prototype_similarity = None

    if (
        embedding is not None
        and prototype_id is not None
    ):

        prototype_similarity = (
            get_prototype_similarity(
                embedding,
                prototype_id,
            )
        )

    return {
        "answer": str(answer),
        "embedding": embedding,
        "dataset_index": dataset_index,
        "retrieval_results": retrieval_results,
        "prototype_id": prototype_id,
        "prototype_similarity": prototype_similarity,
        "demo_case": demo_case,
    }


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🩺 Explainable Medical VLM
        </div>

        <div class="hero-subtitle">
            Prototype-grounded visual evidence for interpretable
            medical image analysis
        </div>

        <div class="hero-badge">
            MedGemma · Visual Retrieval · Learned Prototypes
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

with st.expander(
    "System information",
    expanded=False,
):

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Reference images",
            f"{len(visual_embeddings):,}",
        )

    with c2:
        st.metric(
            "Embedding dimension",
            f"{visual_embeddings.shape[1]}",
        )

    with c3:
        st.metric(
            "Visual prototypes",
            f"{len(cluster_centroids)}",
        )

    with c4:
        st.metric(
            "Prototype examples",
            "90",
        )

    st.markdown(
        """
        <div class="status-row">
            <span class="status">Local retrieval artifacts</span>
            <span class="status">Precomputed VLM analysis</span>
            <span class="status">No live inference endpoint</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 01 — MEDICAL IMAGE
# =============================================================================

section_header(
    "01",
    "Medical Image",
    "Upload one of the prepared demonstration images.",
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

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    current_hash = get_image_hash(
        image
    )

    previous_hash = (
        st.session_state.get(
            "image_hash"
        )
    )

    if (
        previous_hash is not None
        and previous_hash != current_hash
    ):

        reset_image_state()

    st.session_state["image_hash"] = (
        current_hash
    )

    left, right = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with left:

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with right:

        demo_match = match_demo_image(
            image
        )

        if demo_match is not None:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Prepared demonstration case
                    </div>

                    <div class="card-muted">
                        This image has a precomputed MedGemma
                        response and visual embedding. Retrieval
                        and prototype analysis can therefore run
                        without a live inference service.
                    </div>

                    <div class="status-row">
                        <span class="status">Ready</span>
                        <span class="status">Precomputed</span>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            st.caption(
                f"Dataset index: "
                f"{int(demo_match['dataset_index'])}"
            )

        else:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Image not registered
                    </div>

                    <div class="card-muted">
                        This public demo uses precomputed cases.
                        A new image must first be processed
                        offline and added to the demo artifacts.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# 02 — MODEL ANALYSIS
# =============================================================================

section_header(
    "02",
    "Model Analysis",
    "Precomputed MedGemma observation for the selected case.",
)

if uploaded_file is None:

    st.info(
        "Upload a prepared demonstration image to continue."
    )

else:

    if st.session_state[
        "initial_analysis"
    ] is None:

        if st.button(
            "Run Full Analysis",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Loading precomputed analysis..."
                ):

                    result = run_precomputed_demo(
                        image
                    )

                st.session_state[
                    "initial_analysis"
                ] = result["answer"]

                st.session_state[
                    "query_embedding"
                ] = result["embedding"]

                st.session_state[
                    "retrieval_results"
                ] = result[
                    "retrieval_results"
                ]

                st.session_state[
                    "recommended_prototype"
                ] = result[
                    "prototype_id"
                ]

                st.session_state[
                    "prototype_similarity"
                ] = result[
                    "prototype_similarity"
                ]

                st.session_state[
                    "demo_case"
                ] = result[
                    "demo_case"
                ]

                st.session_state[
                    "chat_history"
                ] = [
                    {
                        "role": "assistant",
                        "content": result[
                            "answer"
                        ],
                    }
                ]

                st.session_state[
                    "last_error"
                ] = None

                st.rerun()

            except Exception as exc:

                st.session_state[
                    "last_error"
                ] = str(exc)

    if st.session_state[
        "last_error"
    ]:

        st.warning(
            st.session_state[
                "last_error"
            ]
        )

    if st.session_state[
        "initial_analysis"
    ]:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    MedGemma observation
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            st.session_state[
                "initial_analysis"
            ]
        )

        st.markdown(
            """
                <div class="card-muted">
                    Model-generated observation based on the
                    selected medical image. This should not be
                    interpreted as a clinical diagnosis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 03 — SIMILAR CASES
# =============================================================================

section_header(
    "03",
    "Similar Cases",
    "Unique reference images retrieved from the visual embedding space.",
)

retrieval_results = st.session_state.get(
    "retrieval_results",
    [],
)

if retrieval_results:

    display_results = []

    for result in retrieval_results:

        image_path = get_demo_image_path(
            result["dataset_index"]
        )

        if image_path is None:
            continue

        display_results.append(
            (
                result,
                image_path,
            )
        )

    if display_results:

        columns = st.columns(
            len(display_results),
            gap="medium",
        )

        for column, (
            result,
            image_path,
        ) in zip(
            columns,
            display_results,
        ):

            with column:

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

                st.markdown(
                    f"""
                    <div class="case-label">
                        Rank {result['rank']}
                    </div>

                    <div class="case-meta">
                        Reference case ·
                        Index {result['dataset_index']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Visual similarity",
                    f"{result['similarity'] * 100:.1f}%",
                )

                st.caption(
                    f"Learned prototype "
                    f"P{result['prototype_id']:02d}"
                )

    else:

        st.info(
            "Retrieved cases were found, but their images "
            "could not be resolved."
        )

else:

    st.info(
        "Run the full analysis to retrieve visually similar cases."
    )


# =============================================================================
# 04 — PROTOTYPE EVIDENCE
# =============================================================================

section_header(
    "04",
    "Prototype Evidence",
    "Three representative images from the highest-ranked learned visual prototype.",
)

recommended_prototype = (
    st.session_state.get(
        "recommended_prototype"
    )
)

prototype_similarity = (
    st.session_state.get(
        "prototype_similarity"
    )
)

if recommended_prototype is not None:

    prototype_name = (
        f"P{int(recommended_prototype):02d}"
    )

    description = prototype_description(
        recommended_prototype
    )

    top_left, top_right = st.columns(
        [1.35, 0.65],
        gap="large",
    )

    with top_left:

        st.markdown(
            f"""
            <div class="card">

                <div class="prototype-id">
                    {prototype_name}
                </div>

                <div class="prototype-description">
                    {description}
                </div>

                <br>

                <div class="card-muted">
                    The prototype is a recurring pattern in the
                    learned visual representation space. It is not
                    a clinically validated diagnostic category.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:

        if prototype_similarity is not None:

            st.metric(
                "Prototype similarity",
                f"{prototype_similarity * 100:.1f}%",
            )

        else:

            st.metric(
                "Prototype similarity",
                "N/A",
            )

        st.caption(
            "Cosine similarity between the query representation "
            "and the selected prototype."
        )

    st.write("")

    representative_paths = (
        get_prototype_image_paths(
            recommended_prototype
        )
    )

    if representative_paths:

        image_columns = st.columns(
            len(representative_paths),
            gap="medium",
        )

        for index, (
            column,
            image_path,
        ) in enumerate(
            zip(
                image_columns,
                representative_paths,
            ),
            start=1,
        ):

            with column:

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

                st.caption(
                    f"Prototype {prototype_name} · "
                    f"Representative {index}"
                )

    else:

        st.info(
            "Representative prototype images are not available."
        )

else:

    st.info(
        "The recommended prototype will appear after analysis."
    )


# =============================================================================
# 05 — EXPLAINABILITY
# =============================================================================

section_header(
    "05",
    "Explainability",
    "How the prototype layer provides visual context for the model output.",
)

st.markdown(
    """
    <div class="evidence-box">

        <div class="evidence-title">
            Prototype-Grounded Visual Evidence
        </div>

        The selected image is represented as a visual embedding and
        compared with a reference medical image database.

        <br><br>

        Visually related reference images are organized into learned
        visual prototypes. The highest-ranked prototype provides a
        compact evidence layer that helps contextualize the
        model-generated observation.

        <div class="limitation">

            <b>Interpretation boundary</b><br>

            Prototype similarity indicates visual association in the
            learned representation space. It does not establish a
            causal medical relationship, clinical validity, or
            diagnostic certainty.

            <br><br>

            The prototype retrieval process is also separate from
            MedGemma generation; therefore, retrieval does not prove
            that MedGemma used the retrieved prototype to produce its
            response.

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 06 — ASK ABOUT THIS IMAGE
# =============================================================================

section_header(
    "06",
    "Ask About This Image",
    "Ask a question covered by the precomputed VQA results for this case.",
)

if uploaded_file is not None:

    demo_case = st.session_state.get(
        "demo_case"
    )

    if demo_case is not None:

        qa_data = demo_case.get(
            "qa"
        )

        if qa_data:

            try:

                if isinstance(
                    qa_data,
                    str,
                ):
                    qa_data = json.loads(
                        qa_data
                    )

            except Exception:

                qa_data = {}

            if isinstance(
                qa_data,
                dict,
            ) and qa_data:

                questions = list(
                    qa_data.keys()
                )

                selected_question = st.selectbox(
                    "Precomputed questions",
                    questions,
                )

                if st.button(
                    "Show Answer",
                    use_container_width=True,
                ):

                    answer = qa_data.get(
                        selected_question
                    )

                    if answer:

                        st.session_state[
                            "chat_history"
                        ].append(
                            {
                                "role": "user",
                                "content": selected_question,
                            }
                        )

                        st.session_state[
                            "chat_history"
                        ].append(
                            {
                                "role": "assistant",
                                "content": str(answer),
                            }
                        )

                history = st.session_state.get(
                    "chat_history",
                    [],
                )

                # Only show the question/answer interactions
                # after the initial model response.
                if len(history) > 1:

                    for message in history[1:]:

                        with st.chat_message(
                            message["role"]
                        ):

                            st.write(
                                message["content"]
                            )

            else:

                st.info(
                    "No question-specific precomputed answers "
                    "are available for this case."
                )

        else:

            st.info(
                "No question-specific precomputed answers "
                "are available for this case."
            )

    else:

        st.info(
            "Run the analysis first to enable precomputed questions."
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
    <div class="footer">

        <hr>

        <b>Explainable Medical VLM</b><br>
        MedGemma · Visual Retrieval · Prototype Reasoning

        <br><br>

        Research prototype for interpretable medical image analysis.
        Model outputs are not clinical diagnoses.

    </div>
    """,
    unsafe_allow_html=True,
)

