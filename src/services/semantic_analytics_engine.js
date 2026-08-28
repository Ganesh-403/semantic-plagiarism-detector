/**
 * Enterprise Semantic Plagiarism Analytics Engine
 * Class-based Business Logic, Matrix Operations, and Dynamic Telemetry Processing
 */

class SemanticAnalyticsEngine {
    constructor(config = {}) {
        this.similarityThreshold = config.similarityThreshold || 0.85;
        this.minChunkSize = config.minChunkSize || 64;
        this.embeddingModel = config.embeddingModel || 'all-mpnet-base-v2';
        this.enableCrossLingual = config.enableCrossLingual !== undefined ? config.enableCrossLingual : true;

        this.documentCorpus = [];
        this.flaggedPairs = [];
        this.vectorDistribution = [15, 45, 78, 32, 18];

        this.initDefaultDataset();
    }

    initDefaultDataset() {
        this.documentCorpus = [
            { id: 'DOC-8901', name: 'neural_transformer_analysis.pdf', author: 'Dr. A. Vance', vectors: 768, tokens: 4120 },
            { id: 'DOC-8902', name: 'deep_learning_paraphrase_v2.docx', author: 'M. K. Sterling', vectors: 768, tokens: 3890 },
            { id: 'DOC-8903', name: 'attention_mechanisms_survey.txt', author: 'Dr. A. Vance', vectors: 768, tokens: 6200 },
            { id: 'DOC-8904', name: 'semantic_similarity_benchmarks.pdf', author: 'R. H. Chen', vectors: 768, tokens: 2950 },
            { id: 'DOC-8905', name: 'natural_language_clustering_final.pdf', author: 'J. L. Miller', vectors: 768, tokens: 5410 }
        ];

        this.flaggedPairs = [
            {
                pairId: 'PAIR-9012',
                sourceDoc: 'neural_transformer_analysis.pdf',
                targetDoc: 'deep_learning_paraphrase_v2.docx',
                cosineSim: 0.942,
                jaccardIndex: 0.485,
                risk: 'HIGH',
                status: 'Under Audit'
            },
            {
                pairId: 'PAIR-9013',
                sourceDoc: 'attention_mechanisms_survey.txt',
                targetDoc: 'semantic_similarity_benchmarks.pdf',
                cosineSim: 0.887,
                jaccardIndex: 0.312,
                risk: 'HIGH',
                status: 'Flagged'
            },
            {
                pairId: 'PAIR-9014',
                sourceDoc: 'natural_language_clustering_final.pdf',
                targetDoc: 'clustering_techniques_draft.docx',
                cosineSim: 0.764,
                jaccardIndex: 0.220,
                risk: 'MEDIUM',
                status: 'Reviewed'
            },
            {
                pairId: 'PAIR-9015',
                sourceDoc: 'cross_lingual_embeddings_v1.pdf',
                targetDoc: 'multilingual_bert_adaptation.txt',
                cosineSim: 0.915,
                jaccardIndex: 0.185,
                risk: 'HIGH',
                status: 'Under Audit'
            }
        ];
    }

    calculateCosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) {
            throw new Error("Vector dimension mismatch or null vector supplied.");
        }
        let dotProduct = 0.0;
        let normA = 0.0;
        let normB = 0.0;

        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        if (normA === 0 || normB === 0) return 0.0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    calculateJaccardIndex(setA, setB) {
        const intersection = new Set([...setA].filter(x => setB.has(x)));
        const union = new Set([...setA], [...setB]);
        if (union.size === 0) return 0.0;
        return intersection.size / union.size;
    }

    evaluateRiskScore(cosineSim, jaccardSim) {
        const weightedScore = (cosineSim * 0.7) + (jaccardSim * 0.3);
        if (weightedScore >= this.similarityThreshold) {
            return { risk: 'HIGH', score: weightedScore, badgeClass: 'badge-danger' };
        } else if (weightedScore >= 0.65) {
            return { risk: 'MEDIUM', score: weightedScore, badgeClass: 'badge-accent' };
        } else {
            return { risk: 'LOW', score: weightedScore, badgeClass: 'badge-success' };
        }
    }

    updateConfig(newConfig) {
        if (newConfig.similarityThreshold !== undefined) {
            this.similarityThreshold = parseFloat(newConfig.similarityThreshold);
        }
        if (newConfig.minChunkSize !== undefined) {
            this.minChunkSize = parseInt(newConfig.minChunkSize, 10);
        }
        if (newConfig.embeddingModel) {
            this.embeddingModel = newConfig.embeddingModel;
        }
        if (newConfig.enableCrossLingual !== undefined) {
            this.enableCrossLingual = newConfig.enableCrossLingual;
        }
    }

    getFlaggedPairsFiltered(searchQuery = '') {
        if (!searchQuery) return this.flaggedPairs;
        const q = searchQuery.toLowerCase();
        return this.flaggedPairs.filter(pair =>
            pair.pairId.toLowerCase().includes(q) ||
            pair.sourceDoc.toLowerCase().includes(q) ||
            pair.targetDoc.toLowerCase().includes(q) ||
            pair.risk.toLowerCase().includes(q)
        );
    }
}

// UI Controller Binding
document.addEventListener('DOMContentLoaded', () => {
    const engine = new SemanticAnalyticsEngine();

    const cutoffRange = document.getElementById('range-similarity-cutoff');
    const cutoffLabel = document.getElementById('lbl-cutoff');
    const thresholdBadge = document.getElementById('badge-threshold');
    const chunkSizeRange = document.getElementById('range-min-chunk-size');
    const chunkSizeLabel = document.getElementById('lbl-chunk-size');
    const searchInput = document.getElementById('corpus-search-input');
    const tableBody = document.getElementById('telemetry-table-body');
    const btnReindex = document.getElementById('btn-recalculate-corpus');
    const btnSync = document.getElementById('btn-refresh-telemetry');

    function renderTable(data) {
        if (!tableBody) return;
        tableBody.innerHTML = '';

        data.forEach(item => {
            const riskEval = engine.evaluateRiskScore(item.cosineSim, item.jaccardIndex);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${item.pairId}</strong></td>
                <td>${item.sourceDoc}</td>
                <td>${item.targetDoc}</td>
                <td>${(item.cosineSim * 100).toFixed(1)}%</td>
                <td>${(item.jaccardIndex * 100).toFixed(1)}%</td>
                <td><span class="badge ${riskEval.badgeClass}">${riskEval.risk}</span></td>
                <td>${item.status}</td>
                <td>
                    <button class="btn btn-sm btn-secondary" onclick="alert('Viewing Alignment for ${item.pairId}')">Inspect</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    if (cutoffRange && cutoffLabel) {
        cutoffRange.addEventListener('input', (e) => {
            const val = e.target.value;
            cutoffLabel.textContent = `${val}%`;
            if (thresholdBadge) thresholdBadge.textContent = `Threshold: ${val}%`;
            engine.updateConfig({ similarityThreshold: val / 100 });
            renderTable(engine.getFlaggedPairsFiltered(searchInput ? searchInput.value : ''));
        });
    }

    if (chunkSizeRange && chunkSizeLabel) {
        chunkSizeRange.addEventListener('input', (e) => {
            chunkSizeLabel.textContent = `${e.target.value} tokens`;
        });
    }

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            renderTable(engine.getFlaggedPairsFiltered(e.target.value));
        });
    }

    if (btnReindex) {
        btnReindex.addEventListener('click', () => {
            btnReindex.classList.add('loading');
            setTimeout(() => {
                btnReindex.classList.remove('loading');
                renderTable(engine.getFlaggedPairsFiltered());
                alert('Semantic Vector Index successfully re-calculated across corpus.');
            }, 600);
        });
    }

    if (btnSync) {
        btnSync.addEventListener('click', () => {
            renderTable(engine.getFlaggedPairsFiltered());
        });
    }

    // Initial render
    renderTable(engine.getFlaggedPairsFiltered());
});
