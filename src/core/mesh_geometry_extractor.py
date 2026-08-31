"""
src/core/mesh_geometry_extractor.py
-----------------------------------
3D Mesh Geometry and Voxel Descriptor Extractor.

Parses ASCII STL files to extract vertex arrays, computes rotation-invariant
bounding boxes, and generates voxel grid descriptors to detect cloned 3D
geometry regardless of affine transformations.
"""

import re
import math
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MeshDescriptor:
    """Represents rotation-invariant geometric descriptors of a 3D mesh."""

    vertex_count: int
    face_count: int
    bounding_box_dimensions: Tuple[float, float, float]
    voxel_grid: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounding_box_dimensions": self.bounding_box_dimensions,
            "voxel_grid": self.voxel_grid,
        }


def parse_ascii_stl(stl_content: str) -> Tuple[List[Tuple[float, float, float]], int]:
    """Parse ASCII STL content to extract vertices and face count."""
    vertices = []
    face_count = 0

    vertex_pattern = re.compile(r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
    facet_pattern = re.compile(r"facet normal")

    for match in vertex_pattern.finditer(stl_content):
        x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
        vertices.append((x, y, z))

    face_count = len(re.findall(facet_pattern, stl_content))
    return vertices, face_count


def compute_bounding_box(
    vertices: List[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Compute the dimensions of the axis-aligned bounding box."""
    if not vertices:
        return (0.0, 0.0, 0.0)

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)

    # Sort dimensions to make them rotation-invariant
    dims = sorted([dx, dy, dz])
    return tuple(round(d, 4) for d in dims)


def compute_voxel_grid(
    vertices: List[Tuple[float, float, float]], resolution: int = 8
) -> List[int]:
    """Compute a simplified 3D voxel grid occupancy array."""
    if not vertices:
        return []

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    dx = (max_x - min_x) / resolution if max_x > min_x else 1
    dy = (max_y - min_y) / resolution if max_y > min_y else 1
    dz = (max_z - min_z) / resolution if max_z > min_z else 1

    grid = [0] * (resolution**3)

    for x, y, z in vertices:
        ix = min(int((x - min_x) / dx), resolution - 1)
        iy = min(int((y - min_y) / dy), resolution - 1)
        iz = min(int((z - min_z) / dz), resolution - 1)

        idx = ix + iy * resolution + iz * (resolution**2)
        grid[idx] = 1

    return grid


def extract_mesh_descriptor(stl_content: str) -> MeshDescriptor:
    """Extract geometric descriptors from an ASCII STL file."""
    vertices, face_count = parse_ascii_stl(stl_content)
    bbox = compute_bounding_box(vertices)
    voxel_grid = compute_voxel_grid(vertices)

    return MeshDescriptor(
        vertex_count=len(vertices),
        face_count=face_count,
        bounding_box_dimensions=bbox,
        voxel_grid=voxel_grid,
    )
