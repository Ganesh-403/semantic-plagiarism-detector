from src.analysis.ast_hashing_engine import ASTHashingEngine

class CloneOrchestrator:
    def __init__(self):
        self.hash_engine = ASTHashingEngine()

    def process_and_classify_submission(self, current_source: str, historical_records: list[dict]) -> list[dict]:
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
                matched_clones.append({
                    "matched_submission_id": record["submission_id"],
                    "similarity_score": 100.00,
                    "classification": "type_1_exact"
                })
                continue

            # 2. Check for Type 2/3 Clones (Renamed variables or structural code restructuring)
            similarity = self.hash_engine.calculate_token_similarity(
                current_profile["tokens"], 
                record["tokens"]
            )

            if similarity >= 85.00:
                classification = "type_2_renamed" if current_profile["ast_hash"] == record["ast_hash"] else "type_3_restructured"
                matched_clones.append({
                    "matched_submission_id": record["submission_id"],
                    "similarity_score": similarity,
                    "classification": classification
                })

        return sorted(matched_clones, key=lambda x: x["similarity_score"], reverse=True)
