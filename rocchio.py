import numpy as np

def rocchio(query_vec, relevant_vecs, non_relevant_vecs,
            alpha=1.0, beta=0.75, gamma=0.15):
    """
    Adjusts query vector toward relevant docs and away from non-relevant docs.
    alpha: weight of original query
    beta:  weight of relevant centroid
    gamma: weight of non-relevant centroid
    """
    new_query = alpha * query_vec

    if len(relevant_vecs) > 0:
        relevant_centroid = np.mean(relevant_vecs, axis=0)
        new_query += beta * relevant_centroid

    if len(non_relevant_vecs) > 0:
        non_relevant_centroid = np.mean(non_relevant_vecs, axis=0)
        new_query -= gamma * non_relevant_centroid

    # Renormalize
    new_query = new_query / np.linalg.norm(new_query)
    return new_query