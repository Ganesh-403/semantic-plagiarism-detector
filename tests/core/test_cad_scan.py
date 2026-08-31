"""
tests/core/test_cad_scan.py
---------------------------
Unit tests for 3D Mesh and CAD Model Structural Plagiarism Detection.
"""

import pytest
from src.core.mesh_geometry_extractor import (
    parse_ascii_stl,
    compute_bounding_box,
    extract_mesh_descriptor,
)
from src.core.spatial_shape_aligner import compute_cad_similarity

MOCK_STL = """
solid mock
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid mock
"""


class TestMeshGeometryExtractor:
    def test_parse_ascii_stl(self):
        verts, faces = parse_ascii_stl(MOCK_STL)
        assert len(verts) == 3
        assert faces == 1

    def test_compute_bounding_box(self):
        verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        bbox = compute_bounding_box(verts)
        assert bbox == (0.0, 1.0, 1.0)

    def test_extract_mesh_descriptor(self):
        desc = extract_mesh_descriptor(MOCK_STL)
        assert desc.vertex_count == 3
        assert len(desc.voxel_grid) > 0


class TestSpatialShapeAligner:
    def test_compute_cad_similarity_identical(self):
        desc_a = extract_mesh_descriptor(MOCK_STL)
        desc_b = extract_mesh_descriptor(MOCK_STL)
        result = compute_cad_similarity(desc_a, desc_b)
        assert result["overall_score"] == 1.0
        assert result["is_cloned_geometry"] is True
