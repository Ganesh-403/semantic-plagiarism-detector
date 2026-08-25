import pytest
from PIL import Image

from src.core.image_phash_engine import ImagePHashEngine
from src.core.equation_ast_parser import EquationASTParser

class TestMultimodalDetection:

    def test_phash_collision_resistance(self):
        """
        Verify that similar images (e.g., rotated) produce the same rotation-invariant pHash,
        and drastically different images produce different hashes.
        """
        # Create a simple test image (red square)
        img1 = Image.new('RGB', (100, 100), color = 'red')
        
        # Create a rotated version (a red square rotated is still the same, but let's draw something)
        from PIL import ImageDraw
        img2 = Image.new('RGB', (100, 100), color = 'white')
        d = ImageDraw.Draw(img2)
        d.rectangle([20, 20, 80, 80], fill='blue')
        
        # Rotate img2
        img2_rotated = img2.rotate(90)
        
        # Their rotation-invariant hashes should match
        hash_img2 = ImagePHashEngine.compute_rotation_invariant_phash(img2)
        hash_img2_rot = ImagePHashEngine.compute_rotation_invariant_phash(img2_rotated)
        
        assert hash_img2 == hash_img2_rot
        
        # Distance should be 0
        dist = ImagePHashEngine.phash_distance(hash_img2, hash_img2_rot)
        assert dist == 0

        # Different images should have different hashes
        hash_img1 = ImagePHashEngine.compute_rotation_invariant_phash(img1)
        assert hash_img1 != hash_img2
        assert ImagePHashEngine.phash_distance(hash_img1, hash_img2) > 0

    def test_equation_ast_tree_edit_distance(self):
        """
        Verify that the AST parser and edit distance logic correctly measures structural similarity.
        """
        eq1 = "E = mc^2"
        eq2 = "E = mc^2"
        eq3 = "E = m c ^ 2"  # Structurally identical, just spaces added
        eq4 = "F = ma"
        
        ast1 = EquationASTParser.parse_latex_to_ast(eq1)
        ast2 = EquationASTParser.parse_latex_to_ast(eq2)
        ast3 = EquationASTParser.parse_latex_to_ast(eq3)
        ast4 = EquationASTParser.parse_latex_to_ast(eq4)
        
        # Identical equations should have 0.0 distance
        dist1_2 = EquationASTParser.tree_edit_distance(ast1, ast2)
        assert dist1_2 == 0.0
        
        # Spacing should not affect the parsed AST structure
        dist1_3 = EquationASTParser.tree_edit_distance(ast1, ast3)
        assert dist1_3 == 0.0
        
        # Different equations should have > 0 distance
        dist1_4 = EquationASTParser.tree_edit_distance(ast1, ast4)
        assert dist1_4 > 0.0
