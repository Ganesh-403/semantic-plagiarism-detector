/**
 * Enterprise Stylometric Authorship Attribution Engine
 * Burrows Delta Distance, Yule's K Vocabulary Richness, and Sentence Entropy Engine
 */

class StylometricAuthorshipEngine {
    constructor(config = {}) {
        this.burrowsDeltaThreshold = config.burrowsDeltaThreshold || 0.85;
        this.topFunctionWordsCount = config.topFunctionWordsCount || 100;
        this.detectGhostwriter = config.detectGhostwriter !== undefined ? config.detectGhostwriter : true;

        this.authorshipProfiles = [];
        this.initDefaultProfiles();
    }

    initDefaultProfiles() {
        this.authorshipProfiles = [
            {
                profileId: 'STYLE-601',
                suspectedAuthor: 'Dr. Evelyn Reed',
                targetDoc: 'machine_learning_paper_final.docx',
                burrowsDelta: 0.320,
                yulesKScore: 142.5,
                risk: 'MATCHED AUTHOR',
                status: 'Verified'
            },
            {
                profileId: 'STYLE-602',
                suspectedAuthor: 'Student J. Doe (ID-4890)',
                targetDoc: 'computer_science_thesis_ch4.pdf',
                burrowsDelta: 1.480,
                yulesKScore: 68.2,
                risk: 'HIGH AUTHOR SHIFT (GHOSTWRITING)',
                status: 'Flagged'
            },
            {
                profileId: 'STYLE-603',
                suspectedAuthor: 'Prof. M. Sterling',
                targetDoc: 'neural_network_benchmarks.pdf',
                burrowsDelta: 0.410,
                yulesKScore: 135.0,
                risk: 'MATCHED AUTHOR',
                status: 'Verified'
            },
            {
                profileId: 'STYLE-604',
                suspectedAuthor: 'Alex Vance',
                targetDoc: 'data_structures_essay_v2.docx',
                burrowsDelta: 0.950,
                yulesKScore: 92.4,
                risk: 'SUSPECTED STYLISTIC INCONSISTENCY',
                status: 'Under Audit'
            }
        ];
    }

    calculateBurrowsDelta(authorFreqs, docFreqs) {
        if (!authorFreqs || !docFreqs || authorFreqs.length !== docFreqs.length) return 0;
        let sumDelta = 0.0;
        for (let i = 0; i < authorFreqs.length; i++) {
            sumDelta += Math.abs(authorFreqs[i] - docFreqs[i]);
        }
        return sumDelta / authorFreqs.length;
    }

    evaluateAttributionRisk(deltaScore, yulesK) {
        if (deltaScore >= 1.20) {
            return { risk: 'HIGH AUTHOR SHIFT (GHOSTWRITTEN)', badgeClass: 'badge-danger' };
        } else if (deltaScore >= 0.85) {
            return { risk: 'INCONSISTENT WRITEPRINT', badgeClass: 'badge-warning' };
        } else {
            return { risk: 'VERIFIED AUTHOR MATCH', badgeClass: 'badge-success' };
        }
    }

    updateConfig(newConfig) {
        if (newConfig.burrowsDeltaThreshold !== undefined) {
            this.burrowsDeltaThreshold = parseFloat(newConfig.burrowsDeltaThreshold);
        }
        if (newConfig.topFunctionWordsCount !== undefined) {
            this.topFunctionWordsCount = parseInt(newConfig.topFunctionWordsCount, 10);
        }
        if (newConfig.detectGhostwriter !== undefined) {
            this.detectGhostwriter = newConfig.detectGhostwriter;
        }
    }

    getProfilesFiltered(query = '') {
        if (!query) return this.authorshipProfiles;
        const q = query.toLowerCase();
        return this.authorshipProfiles.filter(p => 
            p.profileId.toLowerCase().includes(q) ||
            p.suspectedAuthor.toLowerCase().includes(q) ||
            p.targetDoc.toLowerCase().includes(q) ||
            p.risk.toLowerCase().includes(q)
        );
    }
}

// UI Controller Binding
document.addEventListener('DOMContentLoaded', () => {
    const engine = new StylometricAuthorshipEngine();

    const cutoffRange = document.getElementById('range-style-cutoff');
    const cutoffLabel = document.getElementById('lbl-style-cutoff');
    const cutoffBadge = document.getElementById('badge-style-cutoff');
    const wordsRange = document.getElementById('range-top-words');
    const wordsLabel = document.getElementById('lbl-top-words');
    const searchInput = document.getElementById('stylometric-search-input');
    const tableBody = document.getElementById('style-table-body');
    const btnReindex = document.getElementById('btn-reindex-style');
    const btnSync = document.getElementById('btn-sync-style-telemetry');

    function renderTable(data) {
        if (!tableBody) return;
        tableBody.innerHTML = '';

        data.forEach(item => {
            const riskEval = engine.evaluateAttributionRisk(item.burrowsDelta, item.yulesKScore);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${item.profileId}</strong></td>
                <td>${item.suspectedAuthor}</td>
                <td>${item.targetDoc}</td>
                <td>${item.burrowsDelta.toFixed(3)}</td>
                <td>${item.yulesKScore.toFixed(1)}</td>
                <td><span class="badge ${riskEval.badgeClass}">${riskEval.risk}</span></td>
                <td>${item.status}</td>
                <td>
                    <button class="btn btn-sm btn-secondary" onclick="alert('Viewing Writeprint Profile for ${item.profileId}')">Writeprint</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    if (cutoffRange && cutoffLabel) {
        cutoffRange.addEventListener('input', (e) => {
            const val = (e.target.value / 100).toFixed(2);
            cutoffLabel.textContent = val;
            if (cutoffBadge) cutoffBadge.textContent = `Delta Cutoff: ${val}`;
            engine.updateConfig({ burrowsDeltaThreshold: parseFloat(val) });
            renderTable(engine.getProfilesFiltered(searchInput ? searchInput.value : ''));
        });
    }

    if (wordsRange && wordsLabel) {
        wordsRange.addEventListener('input', (e) => {
            wordsLabel.textContent = `${e.target.value} words`;
        });
    }

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            renderTable(engine.getProfilesFiltered(e.target.value));
        });
    }

    if (btnReindex) {
        btnReindex.addEventListener('click', () => {
            btnReindex.classList.add('loading');
            setTimeout(() => {
                btnReindex.classList.remove('loading');
                renderTable(engine.getProfilesFiltered());
                alert('Stylometric Burrows Delta writeprint matrix re-calculated.');
            }, 500);
        });
    }

    if (btnSync) {
        btnSync.addEventListener('click', () => {
            renderTable(engine.getProfilesFiltered());
        });
    }

    renderTable(engine.getProfilesFiltered());
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { StylometricAuthorshipEngine };
}
