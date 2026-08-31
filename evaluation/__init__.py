"""
evaluation
----------
Benchmark and evaluation tools for the Semantic Plagiarism Detection System.

Modules:
    evaluate            Run precision/recall/F1 evaluation against a labelled
                        benchmark dataset.  Compares Sentence Transformer
                        embeddings vs a TF-IDF lexical baseline.

    adversarial_generator
                        Generate controlled adversarial variants using 8
                        transformation categories: synonym substitution,
                        sentence reordering, clause restructuring, local
                        paraphrasing, compression/expansion, mixed sections,
                        and lexical noise.

    adversarial_benchmark
                        Run the adversarial benchmark and produce metrics
                        identifying which transformations cause the largest
                        detection degradation.  Includes CI regression checks.

Data:
    benchmark_dataset.json   25 labelled text pairs (10 plagiarized,
                             15 not plagiarized) spanning heavy paraphrases,
                             light paraphrases, same-topic originals,
                             and different-topic negatives.

Usage (from project root):
    python -m evaluation.evaluate
    python -m evaluation.adversarial_generator
    python -m evaluation.adversarial_benchmark
"""
