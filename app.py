import os
import tempfile
from pathlib import Path

import numpy as np
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
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

HF_SPACE = (
    "mdjisanmashrafi/"
    "medgemma-medical-backend"
)


# ============================================================
# LOAD RETRIEVAL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    return load_retrieval_artifacts()


try:

    artifacts = load_artifacts()

except Exception as e:

    st.error(
        "Failed to load retrieval artifacts."
    )

    st.exception(e)

    st.stop()


visual_embeddings = artifacts[
    "visual_embeddings"
]

valid_indices = artifacts[
    "valid_indices"
]

cluster_labels = artifacts[
    "cluster_labels"
]

prototype_summary = artifacts[
    "prototype_summary"
]

prototype_images = artifacts[
    "prototype_images"
]


# ============================================================
# LOAD HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def load_medgemma_client():

    hf_token = st.secrets["HF_TOKEN"]

    return Client(
        HF_SPACE,
        token=hf_token,
    )


# ============================================================
# SAVE TEMP IMAGE
# ============================================================

def save_temp_image(image):

    tmp = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    )

    tmp.close()

    image.save(
        tmp.name,
        format="PNG",
    )

    return tmp.name


# ============================================================
# MEDGEMMA VQA
# ============================================================

def query_medgemma(
    image,
    question,
):

    client = load_medgemma_client()

    image_path = save_temp_image(
        image
    )

    try:

        result = client.predict(
            handle_file(image_path),
            question,
            api_name="/analyze_image",
        )

        return result

    finally:

        try:

            os.remove(
                image_path
            )

        except OSError:

            pass


# ============================================================
# MEDGEMMA VISION EMBEDDING
# ============================================================

def query_embedding(image):

    client = load_medgemma_client()

    image_path = save_temp_image(
        image
    )

    try:

        result = client.predict(
            handle_file(image_path),
            api_name="/embed_image",
        )

        # Gradio JSON output should be a list
        embedding = np.asarray(
            result,
            dtype=np.float32,
        )

        # Flatten possible [1, 1152] shape
        embedding = embedding.reshape(-1)

        return embedding

    finally:

        try:

            os.remove(
                image_path
            )

        except OSError:

            pass


# ============================================================
# HEADER
# ============================================================

st.title(
    "🩺 Explainable Medical VLM"
)

st.markdown(
    """
### MedGemma + Visual Prototype Retrieval

Upload a medical image to explore MedGemma-based visual analysis
and visually similar prototype cases.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.metric(
        "Medical cases",
        f"{len(visual_embeddings):,}",
    )

    st.metric(
        "Visual prototypes",
        f"{len(prototype_summary):,}",
    )

    st.metric(
        "Embedding dimension",
        f"{visual_embeddings.shape[1]:,}",
    )

    st.divider()

    st.markdown(
        """
**Pipeline**

Medical Image  
↓  
MedGemma  
↓  
Medical VQA Answer

**Visual Explanation**

Medical Image  
↓  
MedGemma Vision Encoder  
↓  
1,152-dimensional embedding  
↓  
Similarity Retrieval  
↓  
Prototype Identification  
↓  
Representative Cases
"""
    )

    st.divider()

    st.caption(
        "MedGemma inference is provided through "
        "a GPU-enabled Hugging Face Space."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "1. Upload Medical Image"
)

uploaded_file = st.file_uploader(
    "Choose a medical image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload a medical image to begin "
        "the explainable medical VLM analysis."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    uploaded_file
).convert("RGB")


# ============================================================
# DISPLAY IMAGE
# ============================================================

col_image, col_info = st.columns(
    [1.15, 1]
)


with col_image:

    st.image(
        image,
        caption="Uploaded medical image",
        use_container_width=True,
    )


with col_info:

    st.markdown(
        "### Image Analysis"
    )

    st.write(
        f"**Resolution:** "
        f"{image.width} × {image.height}"
    )

    st.write(
        "**Analysis engine:** "
        "MedGemma 1.5 4B"
    )

    st.write(
        "**Retrieval database:** "
        f"{len(visual_embeddings):,} medical cases"
    )


# ============================================================
# MEDICAL VQA
# ============================================================

st.divider()

st.subheader(
    "2. Medical VQA"
)

question = st.text_input(
    "Ask a question about the image",

    value=(
        "What findings are visible "
        "in this image?"
    ),
)


if st.button(
    "Analyze with MedGemma",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Sending image to MedGemma..."
    ):

        try:

            answer = query_medgemma(
                image,
                question,
            )

            st.success(
                "MedGemma analysis complete."
            )

            st.markdown(
                "### MedGemma Answer"
            )

            st.info(answer)

        except Exception as e:

            st.error(
                "MedGemma inference failed."
            )

            st.exception(e)


# ============================================================
# LIVE VISUAL RETRIEVAL
# ============================================================

st.divider()

st.subheader(
    "3. Visual Retrieval Database"
)

st.info(
    """
The uploaded image is encoded using the same MedGemma vision
encoder used to construct the retrieval database. Its 1,152-dimensional
representation is compared against 1,793 precomputed medical cases
using cosine similarity.
"""
)


# ------------------------------------------------------------
# RETRIEVAL BUTTON
# ------------------------------------------------------------

if st.button(
    "Find Visually Similar Cases",
    use_container_width=True,
):

    with st.spinner(
        "Generating MedGemma visual embedding "
        "and searching the medical case database..."
    ):

        try:

            query_embedding_vector = (
                query_embedding(image)
            )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if (
                query_embedding_vector.shape[0]
                != visual_embeddings.shape[1]
            ):

                raise ValueError(
                    "The returned MedGemma embedding "
                    "has the wrong dimension: "
                    f"{query_embedding_vector.shape[0]} "
                    f"instead of "
                    f"{visual_embeddings.shape[1]}."
                )

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            retrieval_results = (
                find_similar_embeddings(
                    query_embedding_vector,
                    visual_embeddings,
                    valid_indices,
                    cluster_labels,
                    top_k=5,
                )
            )

            # ------------------------------------------------
            # Store in session
            # ------------------------------------------------

            st.session_state[
                "query_embedding"
            ] = query_embedding_vector

            st.session_state[
                "retrieval_results"
            ] = retrieval_results

            # Most similar case determines prototype
            recommended_prototype = (
                retrieval_results[0][
                    "prototype_id"
                ]
            )

            st.session_state[
                "recommended_prototype"
            ] = recommended_prototype

            st.success(
                "Visual retrieval complete."
            )

        except Exception as e:

            st.error(
                "Visual retrieval failed."
            )

            st.exception(e)


# ============================================================
# DATABASE METRICS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Stored embeddings",
        f"{len(visual_embeddings):,}",
    )


with col2:

    st.metric(
        "Embedding dimension",
        f"{visual_embeddings.shape[1]:,}",
    )


with col3:

    st.metric(
        "Visual prototypes",
        f"{len(prototype_summary):,}",
    )


# ============================================================
# RETRIEVAL RESULTS
# ============================================================

retrieval_results = st.session_state.get(
    "retrieval_results"
)


if retrieval_results:

    st.write(
        "### Most Similar Medical Cases"
    )

    for result in retrieval_results:

        st.write(
            f"**Rank {result['rank']}** — "
            f"Dataset index "
            f"`{result['dataset_index']}` — "
            f"Prototype "
            f"`P{result['prototype_id']:02d}` — "
            f"Similarity "
            f"`{result['similarity']:.4f}`"
        )


# ============================================================
# PROTOTYPE EXPLORER
# ============================================================

st.divider()

st.subheader(
    "4. Prototype Explorer"
)


# ------------------------------------------------------------
# Recommended prototype
# ------------------------------------------------------------

recommended_prototype = (
    st.session_state.get(
        "recommended_prototype"
    )
)


prototype_options = sorted(
    prototype_summary[
        "prototype_id"
    ].unique()
)


if recommended_prototype is not None:

    default_index = (
        prototype_options.index(
            recommended_prototype
        )
    )

else:

    default_index = 0


selected_prototype = st.selectbox(
    "Explore a learned visual prototype",

    prototype_options,

    index=default_index,

    format_func=lambda x:
        f"P{x:02d}",
)


# ============================================================
# PROTOTYPE SUMMARY
# ============================================================

selected_summary = prototype_summary[
    prototype_summary[
        "prototype_id"
    ] == selected_prototype
]


if not selected_summary.empty:

    row = selected_summary.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prototype",
            f"P{selected_prototype:02d}",
        )

    with col2:

        st.metric(
            "Cases",
            int(row["num_images"]),
        )

    with col3:

        st.metric(
            "Representative index",
            int(
                row[
                    "representative_dataset_index"
                ]
            ),
        )


# ============================================================
# REPRESENTATIVE IMAGES
# ============================================================

st.write(
    "### Representative Medical Cases"
)


selected_images = prototype_images[
    prototype_images[
        "prototype_id"
    ] == selected_prototype
].sort_values(
    "rank"
)


# ------------------------------------------------------------
# Remove duplicate representative paths
# ------------------------------------------------------------

seen_paths = set()

unique_images = []

for _, image_row in selected_images.iterrows():

    rank = int(
        image_row["rank"]
    )

    image_path = (
        ROOT
        / "representative_images"
        / f"P{selected_prototype:02d}"
        / f"representative_{rank}.png"
    )

    path_key = str(
        image_path.resolve()
    )

    if path_key in seen_paths:

        continue

    seen_paths.add(
        path_key
    )

    unique_images.append(
        (image_row, image_path)
    )

    if len(unique_images) >= 3:

        break


if not unique_images:

    st.warning(
        "No representative images were found "
        "for this prototype."
    )

else:

    columns = st.columns(
        min(
            3,
            len(unique_images)
        )
    )

    for column, (
        image_row,
        image_path,
    ) in zip(
        columns,
        unique_images,
    ):

        with column:

            if image_path.exists():

                representative_image = (
                    Image.open(
                        image_path
                    ).convert("RGB")
                )

                rank = int(
                    image_row["rank"]
                )

                st.image(
                    representative_image,
                    caption=(
                        f"P{selected_prototype:02d} "
                        f"Representative {rank}"
                    ),
                    use_container_width=True,
                )

                if (
                    "distance_to_centroid"
                    in image_row
                ):

                    st.caption(
                        "Distance to prototype "
                        "centroid: "
                        f"{float(image_row['distance_to_centroid']):.4f}"
                    )

            else:

                st.error(
                    f"Image not found: "
                    f"{image_path}"
                )


# ============================================================
# EXPLAINABILITY
# ============================================================

st.divider()

st.subheader(
    "5. Explainability"
)


if recommended_prototype is not None:

    st.markdown(
        f"""
### Retrieved Prototype: P{selected_prototype:02d}

The uploaded medical image was encoded using the MedGemma
vision encoder and compared with 1,793 precomputed medical
visual embeddings.

The most similar cases belong to the learned prototype:

**P{recommended_prototype:02d}**
"""
    )

    if retrieval_results:

        best_match = retrieval_results[0]

        st.metric(
            "Best visual similarity",
            f"{best_match['similarity']:.4f}",
        )

    st.markdown(
        """
The system combines:

- **MedGemma** for medical visual question answering
- **MedGemma vision encoder** for visual representation
- **1,152-dimensional embeddings** for image representation
- **Cosine similarity** for visual retrieval
- **1,793 medical cases** as the retrieval database
- **30 learned prototypes** for organizing the visual space
- **Representative cases** for interpretable comparison

This provides an example-based explanation of the model's
visual representation rather than relying only on a single
black-box prediction.
"""
    )

else:

    st.info(
        "Run visual retrieval above to generate "
        "an image-specific prototype explanation."
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.subheader(
    "6. System Status"
)

status_col1, status_col2 = st.columns(2)


with status_col1:

    st.success(
        "✓ Retrieval artifacts loaded"
    )

    st.success(
        "✓ Prototype database available"
    )


with status_col2:

    st.success(
        "✓ MedGemma VQA backend configured"
    )

    if retrieval_results:

        st.success(
            "✓ Live visual retrieval available"
        )

    else:

        st.info(
            "○ Run visual retrieval to test "
            "live embedding search"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Research prototype — not intended for clinical diagnosis "
    "or treatment. MedGemma outputs should not be used as a "
    "substitute for professional medical judgment."
)
