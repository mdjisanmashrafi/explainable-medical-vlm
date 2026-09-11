from pathlib import Path

import numpy as np
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parent.parent


def load_retrieval_artifacts():
    """Load prototype/retrieval artifacts."""

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
    """Calculate cosine similarity between query and rows of matrix."""

    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        return np.zeros(len(matrix))

    matrix_norm = np.linalg.norm(matrix, axis=1)

    denominator = matrix_norm * query_norm

    denominator[denominator == 0] = 1e-12

    return np.dot(matrix, query) / denominator


def find_similar_embeddings(query_embedding, embeddings, top_k=5):
    """Return indices and similarity scores for nearest embeddings."""

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "index": int(i),
            "similarity": float(scores[i]),
        }
        for i in top_indices
    ]
