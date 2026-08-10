--- a/tests/test_cross_lingual.py
@@ -10,6 +10,7 @@ def test_translate_to_english():
     translation = cross_lingual.translate("Bonjour le monde", "en")
     assert translation == "Hello world"
 
+@pytest.mark.parametrize("input_text, expected_translation, expected_confidence", [
+    ("Bonjour le monde", "Hello world", 0.95),
+])
 def test_translate_to_english_with_confidence():
-    translation = cross_lingual.translate("Bonjour le monde", "en")
-    assert translation == "Hello world"
+    result = cross_lingual.translate_with_confidence("Bonjour le monde", "en")
+    assert result["translation"] == "Hello world"
+    assert round(result["confidence"], 2) == 0.95
