import streamlit as st
from pathlib import Path
from PIL import Image

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


# ============================================================
# LOAD ARTIFACTS
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
# HEADER
# ============================================================

st.title("🩺 Explainable Medical VLM")

st.markdown(
    """
### MedGemma + Visual Prototype Retrieval

Upload a medical image to explore visually similar cases
and understand which learned prototype is most similar.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Information")

    st.metric(
        "Medical cases",
        f"{len(visual_embeddings):,}"
    )

    st.metric(
        "Visual prototypes",
        f"{len(prototype_summary):,}"
    )

    st.metric(
        "Embedding dimension",
        f"{visual_embeddings.shape[1]:,}"
    )

    st.divider()

    st.markdown(
        """
        **Pipeline**

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
        "Upload a medical image to begin the explainable retrieval analysis."
    )

    st.stop()


# ============================================================
# DISPLAY INPUT IMAGE
# ============================================================

image = Image.open(uploaded_file).convert("RGB")

st.image(
    image,
    caption="Uploaded medical image",
    width=500,
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
):

    st.warning(
        "MedGemma inference is not connected yet. "
        "The retrieval system below is ready."
    )

    st.write("Question:", question)


# ============================================================
# RETRIEVAL SECTION
# ============================================================

st.divider()

st.subheader("3. Explainable Visual Retrieval")

st.info(
    """
    The stored medical-image embeddings were generated using
    the MedGemma vision encoder. Once the same encoder processes
    the uploaded image, its embedding can be compared against
    the 1,793 stored cases.
    """
)


# ============================================================
# CURRENT STATUS
# ============================================================

st.write("### Retrieval Database")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Stored embeddings",
        f"{len(visual_embeddings):,}"
    )

with col2:
    st.metric(
        "Embedding dimension",
        f"{visual_embeddings.shape[1]:,}"
    )

with col3:
    st.metric(
        "Prototypes",
        f"{len(prototype_summary):,}"
    )


# ============================================================
# PROTOTYPE EXPLORER
# ============================================================

st.divider()

st.subheader("4. Prototype Explorer")

prototype_options = sorted(
    prototype_summary["prototype_id"].unique()
)

selected_prototype = st.selectbox(
    "Explore a learned visual prototype",
    prototype_options,
    format_func=lambda x: f"P{x:02d}",
)


# Prototype summary
selected_summary = prototype_summary[
    prototype_summary["prototype_id"] == selected_prototype
]


if not selected_summary.empty:

    row = selected_summary.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prototype",
            f"P{selected_prototype:02d}"
        )

    with col2:
        st.metric(
            "Cases",
            int(row["num_images"])
        )

    with col3:
        st.metric(
            "Representative index",
            int(row["representative_dataset_index"])
        )


# ============================================================
# REPRESENTATIVE IMAGES
# ============================================================

st.write("### Representative Medical Cases")

selected_images = prototype_images[
    prototype_images["prototype_id"] == selected_prototype
].sort_values("rank")


if selected_images.empty:

    st.warning(
        "No representative images were found for this prototype."
    )

else:

    columns = st.columns(3)

    for column, (_, image_row) in zip(
        columns,
        selected_images.iterrows()
    ):

        rank = int(image_row["rank"])

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

            else:

                st.error(
                    f"Image not found: {image_path}"
                )


# ============================================================
# EXPLANATION
# ============================================================

st.divider()

st.subheader("5. How the Explanation Works")

st.markdown(
    f"""
    **Prototype P{selected_prototype:02d}** represents a group of
    visually related medical images discovered from the embedding space.

    The system uses:

    - **MedGemma vision embeddings** to represent image content
    - **Cosine similarity** to measure visual similarity
    - **30 learned prototypes** to organize the embedding space
    - **Representative cases** to provide a human-interpretable explanation

    This allows the system to move beyond a single black-box prediction
    and show visually similar cases associated with the learned prototype.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype — not intended for clinical diagnosis or treatment."
)
