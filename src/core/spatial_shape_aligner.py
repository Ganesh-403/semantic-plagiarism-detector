"""
src/core/spatial_shape_aligner.py
---------------------------------
Spatial Shape Alignment and Hausdorff Distance Engine.

Computes structural similarity and Hausdorff distance between normalized
3D meshes to detect cloned CAD models.
"""

import math
import logging
from typing import List, Dict, Any
from src.core.mesh_geometry_extractor import MeshDescriptor

logger = logging.getLogger(__name__)


def compute_hausdorff_distance(grid_a: List[int], grid_b: List[int]) -> float:
    """Compute a simplified discrete Hausdorff distance between two voxel grids."""
    if not grid_a or not grid_b or len(grid_a) != len(grid_b):
        return float("inf")

    # Simplified Hamming-like distance for voxel occupancy
    diff = sum(1 for a, b in zip(grid_a, grid_b) if a != b)
    return diff / len(grid_a)


def compute_cad_similarity(
    desc_a: MeshDescriptor, desc_b: MeshDescriptor
) -> Dict[str, Any]:
    """Compute structural similarity between two 3D mesh descriptors."""
    if desc_a.vertex_count == 0 or desc_b.vertex_count == 0:
        return {
            "hausdorff_distance": 1.0,
            "bbox_similarity": 0.0,
            "overall_score": 0.0,
            "is_cloned_geometry": False,
        }

    # 1. Bounding Box Similarity (Rotation Invariant)
    bbox_a = desc_a.bounding_box_dimensions
    bbox_b = desc_b.bounding_box_dimensions

    bbox_diffs = [abs(a - b) / max(a, b, 1e-6) for a, b in zip(bbox_a, bbox_b)]
    bbox_sim = max(0.0, 1.0 - (sum(bbox_diffs) / 3.0))

    # 2. Voxel Grid Hausdorff Distance
    hausdorff = compute_hausdorff_distance(desc_a.voxel_grid, desc_b.voxel_grid)
    voxel_sim = max(0.0, 1.0 - hausdorff)

    overall_score = (bbox_sim * 0.4) + (voxel_sim * 0.6)
    is_cloned = overall_score > 0.85

    return {
        "hausdorff_distance": round(hausdorff, 4),
        "bbox_similarity": round(bbox_sim, 4),
        "voxel_similarity": round(voxel_sim, 4),
        "overall_score": round(overall_score, 4),
        "is_cloned_geometry": is_cloned,
    }
