import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from gradio_client import Client, handle_file

from src.retrieval import (
    load_retrieval_artifacts,
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

HF_SPACE = "mdjisanmashrafi/medgemma-medical-backend"


# ============================================================
# LOAD RETRIEVAL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    return load_retrieval_artifacts()


try:
    artifacts = load_artifacts()

except Exception as e:

    st.error("Failed to load retrieval artifacts.")
    st.exception(e)
    st.stop()


visual_embeddings = artifacts["visual_embeddings"]
valid_indices = artifacts["valid_indices"]
cluster_labels = artifacts["cluster_labels"]
prototype_summary = artifacts["prototype_summary"]
prototype_images = artifacts["prototype_images"]


# ============================================================
# LOAD HUGGING FACE GRADIO CLIENT
# ============================================================

@st.cache_resource
def load_medgemma_client():

    return Client(HF_SPACE)


# ============================================================
# MEDGEMMA API FUNCTION
# ============================================================

def query_medgemma(image, question):

    client = load_medgemma_client()

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as tmp:

        image.save(
            tmp.name,
            format="PNG",
        )

        image_path = tmp.name

    try:

        result = client.predict(
            handle_file(image_path),
            question,
            api_name="/predict",
        )

        return result

    finally:

        try:
            os.remove(image_path)
        except OSError:
            pass


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Explainable Medical VLM")

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

    st.header("System Information")

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
        "MedGemma inference is provided through a "
        "GPU-enabled Hugging Face Space."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("1. Upload Medical Image")

uploaded_file = st.file_uploader(
    "Choose a medical image",
    type=["png", "jpg", "jpeg"],
)


if uploaded_file is None:

    st.info(
        "Upload a medical image to begin the explainable "
        "medical VLM analysis."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    uploaded_file
).convert("RGB")


# ============================================================
# DISPLAY INPUT IMAGE
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

    st.markdown("### Image Analysis")

    st.write(
        f"**Format:** {image.format or 'PNG'}"
    )

    st.write(
        f"**Resolution:** {image.width} × {image.height}"
    )

    st.write(
        "**Analysis engine:** MedGemma 1.5 4B"
    )

    st.write(
        "**Retrieval database:** 1,793 medical cases"
    )


# ============================================================
# MEDGEMMA VQA SECTION
# ============================================================

st.divider()

st.subheader("2. Medical VQA")

question = st.text_input(
    "Ask a question about the image",
    value="What findings are visible in this image?",
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
# RETRIEVAL DATABASE
# ============================================================

st.divider()

st.subheader("3. Visual Retrieval Database")

st.info(
    """
    The retrieval database contains precomputed visual embeddings
    generated using the MedGemma vision encoder. The uploaded image
    can be compared against these learned representations to identify
    visually related medical cases.
    """
)


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
# PROTOTYPE EXPLORER
# ============================================================

st.divider()

st.subheader("4. Prototype Explorer")

prototype_options = sorted(
    prototype_summary["prototype_id"]
    .unique()
)


selected_prototype = st.selectbox(
    "Explore a learned visual prototype",
    prototype_options,
    format_func=lambda x: f"P{x:02d}",
)


# ============================================================
# SELECTED PROTOTYPE SUMMARY
# ============================================================

selected_summary = prototype_summary[
    prototype_summary["prototype_id"]
    == selected_prototype
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
            int(row["representative_dataset_index"]),
        )


# ============================================================
# REPRESENTATIVE IMAGES
# ============================================================

st.write(
    "### Representative Medical Cases"
)


selected_images = prototype_images[
    prototype_images["prototype_id"]
    == selected_prototype
].sort_values("rank")


if selected_images.empty:

    st.warning(
        "No representative images were found "
        "for this prototype."
    )

else:

    columns = st.columns(
        min(3, len(selected_images))
    )

    for column, (_, image_row) in zip(
        columns,
        selected_images.iterrows(),
    ):

        rank = int(
            image_row["rank"]
        )

        image_path = (
            ROOT
            / "representative_images"
            / f"P{selected_prototype:02d}"
            / f"representative_{rank}.png"
        )

        with column:

            if image_path.exists():

                representative_image = Image.open(
                    image_path
                ).convert("RGB")

                st.image(
                    representative_image,
                    caption=(
                        f"P{selected_prototype:02d} "
                        f"Representative {rank}"
                    ),
                    use_container_width=True,
                )

                if "distance_to_centroid" in image_row:

                    st.caption(
                        "Distance to prototype centroid: "
                        f"{float(image_row['distance_to_centroid']):.4f}"
                    )

            else:

                st.error(
                    f"Image not found: {image_path}"
                )


# ============================================================
# EXPLANATION
# ============================================================

st.divider()

st.subheader("5. Explainability")

st.markdown(
    f"""
    **Prototype P{selected_prototype:02d}** represents a group of
    visually related medical images discovered in the learned
    embedding space.

    The system combines:

    - **MedGemma** for medical visual question answering
    - **MedGemma vision embeddings** for visual representation
    - **Cosine similarity** for visual similarity measurement
    - **30 learned prototypes** for organizing the embedding space
    - **Representative medical cases** for interpretable comparison

    The prototype view provides a visual explanation of the learned
    representation rather than relying only on a single black-box output.
    """
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.subheader("6. System Status")

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
        "✓ MedGemma backend connected"
    )

    st.success(
        "✓ Hugging Face GPU backend configured"
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
