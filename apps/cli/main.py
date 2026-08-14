--- a/apps/cli/main.py
@@ -123,6 +123,10 @@ def extract_text_from_docx(docx_path):
     text = ""
     document = Document(docx_path)
     for paragraph in document.paragraphs:
+        if table_cells(paragraph):
+            for row in paragraph._element.getchildren():
+                for cell in row.iterchildren():
+                    text += cell.text.strip() + " "
         text += paragraph.text.strip() + "\n"
     return text

@@ -145,6 +149,18 @@ def table_cells(paragraph):
     """
     Helper function to check if a paragraph is part of a table
     """
-    return False
+    return any("w:tbl" in element.tag for element in paragraph._element.getchildren())
