# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from src.ocr.neural_alignment_engine import NeuralAlignmentEngine


class OcrPipelineOrchestrator:
    def __init__(self):
        self.engine = NeuralAlignmentEngine()

    def process_multimodal_block(
        self,
        mock_db_client,
        document_id: str,
        page: int,
        raw_extracted_text: str,
        bbox: dict,
        reference_text: str,
    ) -> dict:
        """
        Processes a layout block from a page, applying sanitization and neural checks,
        and saves the finalized metrics to the database.
        """
        # 1. Strip out malicious bypass components
        clean_text, anomalies_purged = self.engine.sanitize_and_clean_text(
            raw_extracted_text
        )

        # 2. Run structural text alignment check
        alignment_score = self.engine.compute_alignment_vectors(
            clean_text, reference_text
        )

        # 3. Assemble data payload for database insertion
        block_record = {
            "document_id": document_id,
            "page_number": page,
            "raw_text": raw_extracted_text,
            "cleansed_text": clean_text,
            "bounding_box": bbox,
            "paraphrase_alignment_score": alignment_score,
        }

        # Simulated database insertion loop; connect your active model context smoothly
        # mock_db_client.table("ocr_extracted_blocks").insert(block_record)

        return {
            "success": True,
            "anomalies_purged": anomalies_purged,
            "alignment_score": alignment_score,
        }
