/**
 * Enterprise Cross-Lingual Paraphrase Detection Engine
 * Multilingual Embedding Space Operations, Back-Translation Alignment, and Semantic Shift Metrics
 */

class CrossLingualEngine {
    constructor(config = {}) {
        this.crossCutoffThreshold = config.crossCutoffThreshold || 0.82;
        this.sourceLangFilter = config.sourceLangFilter || 'all';
        this.targetLangFilter = config.targetLangFilter || 'all';
        this.autoBacktranslation = config.autoBacktranslation !== undefined ? config.autoBacktranslation : true;

        this.crossLingualPairs = [];
        this.initDefaultPairs();
    }

    initDefaultPairs() {
        this.crossLingualPairs = [
            {
                pairId: 'XL-801',
                sourceText: 'Quantum computing revolutionizes cryptography. (EN)',
                targetText: 'La computación cuántica revoluciona la criptografía. (ES)',
                vectorDist: 0.962,
                backTransScore: 0.941,
                risk: 'CRITICAL',
                status: 'Under Audit'
            },
            {
                pairId: 'XL-802',
                sourceText: 'Neural networks optimize global trade. (EN)',
                targetText: 'Les réseaux de neurones optimisent le commerce mondial. (FR)',
                vectorDist: 0.884,
                backTransScore: 0.860,
                risk: 'HIGH',
                status: 'Flagged'
            },
            {
                pairId: 'XL-803',
                sourceText: 'Autonomous driving safety protocols. (EN)',
                targetText: 'Sicherheitsprotokolle für autonomes Fahren. (DE)',
                vectorDist: 0.915,
                backTransScore: 0.892,
                risk: 'CRITICAL',
                status: 'Under Audit'
            },
            {
                pairId: 'XL-804',
                sourceText: 'Climate change impact on marine biology. (EN)',
                targetText: '气候变化对海洋生物学的影响。 (ZH)',
                vectorDist: 0.835,
                backTransScore: 0.790,
                risk: 'MEDIUM',
                status: 'Reviewed'
            }
        ];
    }

    calculateMultilingualDistance(vecA, vecB) {
        if (!vecA || !vecB) return 0;
        let dot = 0.0, normA = 0.0, normB = 0.0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    evaluateCrossLingualRisk(vecDist, backTransScore) {
        const composite = (vecDist * 0.6) + (backTransScore * 0.4);
        if (composite >= this.crossCutoffThreshold) {
            return { risk: 'HIGH PARAPHRASE RISK', badgeClass: 'badge-danger' };
        } else if (composite >= 0.70) {
            return { risk: 'MODERATE RISK', badgeClass: 'badge-warning' };
        } else {
            return { risk: 'LOW RISK', badgeClass: 'badge-success' };
        }
    }

    updateConfig(newConfig) {
        if (newConfig.crossCutoffThreshold !== undefined) {
            this.crossCutoffThreshold = parseFloat(newConfig.crossCutoffThreshold);
        }
        if (newConfig.sourceLangFilter) {
            this.sourceLangFilter = newConfig.sourceLangFilter;
        }
        if (newConfig.targetLangFilter) {
            this.targetLangFilter = newConfig.targetLangFilter;
        }
        if (newConfig.autoBacktranslation !== undefined) {
            this.autoBacktranslation = newConfig.autoBacktranslation;
        }
    }

    getPairsFiltered(query = '') {
        if (!query) return this.crossLingualPairs;
        const q = query.toLowerCase();
        return this.crossLingualPairs.filter(p => 
            p.pairId.toLowerCase().includes(q) ||
            p.sourceText.toLowerCase().includes(q) ||
            p.targetText.toLowerCase().includes(q) ||
            p.risk.toLowerCase().includes(q)
        );
    }
}

// UI Controller Binding
document.addEventListener('DOMContentLoaded', () => {
    const engine = new CrossLingualEngine();

    const cutoffRange = document.getElementById('range-cross-cutoff');
    const cutoffLabel = document.getElementById('lbl-cross-cutoff');
    const cutoffBadge = document.getElementById('badge-cross-cutoff');
    const searchInput = document.getElementById('cross-lingual-search-input');
    const tableBody = document.getElementById('cross-table-body');
    const btnReindex = document.getElementById('btn-reindex-cross');
    const btnSync = document.getElementById('btn-sync-cross-telemetry');

    function renderTable(data) {
        if (!tableBody) return;
        tableBody.innerHTML = '';

        data.forEach(item => {
            const riskEval = engine.evaluateCrossLingualRisk(item.vectorDist, item.backTransScore);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${item.pairId}</strong></td>
                <td>${item.sourceText}</td>
                <td>${item.targetText}</td>
                <td>${(item.vectorDist * 100).toFixed(1)}%</td>
                <td>${(item.backTransScore * 100).toFixed(1)}%</td>
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
            if (cutoffBadge) cutoffBadge.textContent = `Threshold: ${val}%`;
            engine.updateConfig({ crossCutoffThreshold: val / 100 });
            renderTable(engine.getPairsFiltered(searchInput ? searchInput.value : ''));
        });
    }

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            renderTable(engine.getPairsFiltered(e.target.value));
        });
    }

    if (btnReindex) {
        btnReindex.addEventListener('click', () => {
            btnReindex.classList.add('loading');
            setTimeout(() => {
                btnReindex.classList.remove('loading');
                renderTable(engine.getPairsFiltered());
                alert('Multilingual vector space projection complete.');
            }, 500);
        });
    }

    if (btnSync) {
        btnSync.addEventListener('click', () => {
            renderTable(engine.getPairsFiltered());
        });
    }

    renderTable(engine.getPairsFiltered());
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CrossLingualEngine };
}
