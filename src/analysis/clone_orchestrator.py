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

from src.analysis.ast_hashing_engine import ASTHashingEngine


class CloneOrchestrator:
    def __init__(self):
        self.hash_engine = ASTHashingEngine()

    def process_and_classify_submission(
        self, current_source: str, historical_records: list[dict]
    ) -> list[dict]:
        """
        Compares new code blocks against past submissions to classify similarity thresholds.

        Historical record structure example:
        {"submission_id": "uuid", "source_code": "str", "tokens": ["list"]}
        """
        current_profile = self.hash_engine.generate_fingerprint(current_source)
        if not current_profile["success"]:
            return []

        matched_clones = []

        for record in historical_records:
            # 1. Check for Type 1 Clones (Exact literal matches)
            if current_source.strip() == record["source_code"].strip():
                matched_clones.append(
                    {
                        "matched_submission_id": record["submission_id"],
                        "similarity_score": 100.00,
                        "classification": "type_1_exact",
                    }
                )
                continue

            # 2. Check for Type 2/3 Clones (Renamed variables or structural code restructuring)
            similarity = self.hash_engine.calculate_token_similarity(
                current_profile["tokens"], record["tokens"]
            )

            if similarity >= 85.00:
                classification = (
                    "type_2_renamed"
                    if current_profile["ast_hash"] == record["ast_hash"]
                    else "type_3_restructured"
                )
                matched_clones.append(
                    {
                        "matched_submission_id": record["submission_id"],
                        "similarity_score": similarity,
                        "classification": classification,
                    }
                )

        return sorted(matched_clones, key=lambda x: x["similarity_score"], reverse=True)
