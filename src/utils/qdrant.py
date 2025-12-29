"""Qdrant-specific utilities for vector operations."""

from qdrant_client.models import SparseVector

# Vector name constants for Qdrant collections
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def convert_sparse_dict_to_qdrant_sparsevector(
    sparse_dict: dict[int, float],
) -> SparseVector:
    """
    Convert dict[int, float] sparse vector to Qdrant SparseVector object.

    Args:
        sparse_dict: Dictionary mapping token IDs to weights.

    Returns:
        SparseVector object with sorted indices and matching values.
    """
    if not sparse_dict:
        return SparseVector(indices=[], values=[])

    # Sort indices ascending and reorder values to match
    sorted_items = sorted(sparse_dict.items(), key=lambda x: x[0])
    indices = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    return SparseVector(indices=indices, values=values)
