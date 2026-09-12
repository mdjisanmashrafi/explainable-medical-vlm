from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_retrieval_artifacts():

    artifacts = {
        "visual_embeddings": np.load(
            ROOT / "visual_embeddings.npy"
        ),

        "valid_indices": np.load(
            ROOT / "valid_indices.npy"
        ),

        "cluster_labels": np.load(
            ROOT / "cluster_labels.npy"
        ),

        "cluster_centroids": np.load(
            ROOT / "cluster_centroids.npy"
        ),

        "pca_model": joblib.load(
            ROOT / "pca_model.pkl"
        ),

        "pca_scaler": joblib.load(
            ROOT / "pca_scaler.pkl"
        ),

        "prototype_summary": pd.read_csv(
            ROOT / "prototype_summary.csv"
        ),

        "prototype_images": pd.read_csv(
            ROOT / "prototype_representative_images.csv"
        ),
    }

    return artifacts


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query,
    matrix,
):

    query = np.asarray(
        query,
        dtype=np.float32,
    )

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(query)

    if query_norm == 0:

        return np.zeros(
            len(matrix),
            dtype=np.float32,
        )

    matrix_norm = np.linalg.norm(
        matrix,
        axis=1,
    )

    denominator = (
        matrix_norm * query_norm
    )

    denominator[
        denominator == 0
    ] = 1e-12

    scores = np.dot(
        matrix,
        query,
    ) / denominator

    return scores


# ============================================================
# FIND SIMILAR CASES
# ============================================================

def find_similar_embeddings(
    query_embedding,
    embeddings,
    valid_indices,
    cluster_labels,
    top_k=5,
):

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate embedding dimension
    # --------------------------------------------------------

    if query_embedding.shape[-1] != embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"query={query_embedding.shape[-1]}, "
            f"database={embeddings.shape[1]}"
        )

    # --------------------------------------------------------
    # Compute cosine similarity
    # --------------------------------------------------------

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    # --------------------------------------------------------
    # Sort all candidates
    # --------------------------------------------------------

    sorted_rows = np.argsort(scores)[::-1]

    results = []

    seen_dataset_indices = set()

    # --------------------------------------------------------
    # Select distinct cases
    # --------------------------------------------------------

    for row_index in sorted_rows:

        dataset_index = int(
            valid_indices[row_index]
        )

        # Prevent duplicate cases
        if dataset_index in seen_dataset_indices:
            continue

        seen_dataset_indices.add(
            dataset_index
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
            }
        )

        if len(results) >= top_k:
            break

    return results
