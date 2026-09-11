from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent


def load_retrieval_artifacts():
    """Load all precomputed retrieval artifacts."""

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


def cosine_similarity(query, matrix):
    """Calculate cosine similarity between query and matrix rows."""

    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        return np.zeros(len(matrix))

    matrix_norm = np.linalg.norm(matrix, axis=1)

    denominator = matrix_norm * query_norm
    denominator[denominator == 0] = 1e-12

    return np.dot(matrix, query) / denominator


def find_similar_embeddings(
    query_embedding,
    embeddings,
    valid_indices,
    cluster_labels,
    top_k=5,
):
    """
    Find the most visually similar precomputed cases.
    """

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )

    top_rows = np.argsort(scores)[::-1][:top_k]

    results = []

    for row_index in top_rows:
        results.append({
            "embedding_row": int(row_index),
            "dataset_index": int(valid_indices[row_index]),
            "prototype_id": int(cluster_labels[row_index]),
            "similarity": float(scores[row_index]),
        })

    return results
