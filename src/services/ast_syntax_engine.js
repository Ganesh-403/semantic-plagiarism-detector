/**
 * Enterprise Abstract Syntax Tree (AST) Plagiarism Engine
 * Subtree Isomorphism, Tree Edit Distance (TED) Algorithms, and Node Alignment Engine
 */

class ASTSyntaxEngine {
    constructor(config = {}) {
        this.treeCutoffThreshold = config.treeCutoffThreshold || 0.80;
        this.minSubtreeNodes = config.minSubtreeNodes || 12;
        this.targetLanguage = config.targetLanguage || 'python';
        this.ignoreIdentifiers = config.ignoreIdentifiers !== undefined ? config.ignoreIdentifiers : true;

        this.astMatches = [];
        this.initDefaultMatches();
    }

    initDefaultMatches() {
        this.astMatches = [
            {
                matchId: 'AST-4011',
                primaryFile: 'auth_middleware_v1.py',
                compareFile: 'jwt_security_copy.py',
                treeSimilarity: 0.945,
                editDistance: 4,
                cloneClass: 'Type-2 (Renamed Identifiers)',
                status: 'High Confidence Clone'
            },
            {
                matchId: 'AST-4012',
                primaryFile: 'sorting_algorithms.js',
                compareFile: 'quicksort_implementation.js',
                treeSimilarity: 0.872,
                editDistance: 12,
                cloneClass: 'Type-3 (Subtree Insertion)',
                status: 'Under Audit'
            },
            {
                matchId: 'AST-4013',
                primaryFile: 'database_pool.cpp',
                compareFile: 'connection_manager.cpp',
                treeSimilarity: 0.798,
                editDistance: 28,
                cloneClass: 'Type-3 (Reordered Statements)',
                status: 'Reviewed'
            },
            {
                matchId: 'AST-4014',
                primaryFile: 'tree_traversal_utils.py',
                compareFile: 'binary_tree_helpers.py',
                treeSimilarity: 0.910,
                editDistance: 6,
                cloneClass: 'Type-2 (Refactored Subtree)',
                status: 'Flagged'
            }
        ];
    }

    calculateTreeEditDistance(treeA, treeB) {
        if (!treeA || !treeB) return 0;
        // Zhang-Shasha Tree Edit Distance algorithm simulation
        const nodesA = treeA.nodeCount || 10;
        const nodesB = treeB.nodeCount || 10;
        const maxNodes = Math.max(nodesA, nodesB);
        const sim = 1.0 - (Math.abs(nodesA - nodesB) / maxNodes);
        return Math.max(0.0, Math.min(1.0, sim));
    }

    evaluateCloneCategory(similarity, editDist) {
        if (similarity >= 0.92 && editDist <= 5) {
            return { category: 'Type-1 / Type-2 Exact Clone', risk: 'CRITICAL', badgeClass: 'badge-danger' };
        } else if (similarity >= 0.80) {
            return { category: 'Type-3 Near-Miss Subtree Clone', risk: 'HIGH', badgeClass: 'badge-warning' };
        } else {
            return { category: 'Type-4 Semantic Alternative', risk: 'LOW', badgeClass: 'badge-success' };
        }
    }

    updateConfig(newConfig) {
        if (newConfig.treeCutoffThreshold !== undefined) {
            this.treeCutoffThreshold = parseFloat(newConfig.treeCutoffThreshold);
        }
        if (newConfig.minSubtreeNodes !== undefined) {
            this.minSubtreeNodes = parseInt(newConfig.minSubtreeNodes, 10);
        }
        if (newConfig.targetLanguage) {
            this.targetLanguage = newConfig.targetLanguage;
        }
        if (newConfig.ignoreIdentifiers !== undefined) {
            this.ignoreIdentifiers = newConfig.ignoreIdentifiers;
        }
    }

    getMatchesFiltered(query = '') {
        if (!query) return this.astMatches;
        const q = query.toLowerCase();
        return this.astMatches.filter(m => 
            m.matchId.toLowerCase().includes(q) ||
            m.primaryFile.toLowerCase().includes(q) ||
            m.compareFile.toLowerCase().includes(q) ||
            m.cloneClass.toLowerCase().includes(q)
        );
    }
}

// UI Binding Logic
document.addEventListener('DOMContentLoaded', () => {
    const engine = new ASTSyntaxEngine();

    const cutoffRange = document.getElementById('range-ast-cutoff');
    const cutoffLabel = document.getElementById('lbl-ast-cutoff');
    const cutoffBadge = document.getElementById('badge-ast-cutoff');
    const minNodesRange = document.getElementById('range-min-nodes');
    const minNodesLabel = document.getElementById('lbl-min-nodes');
    const searchInput = document.getElementById('ast-search-input');
    const tableBody = document.getElementById('ast-table-body');
    const btnReparse = document.getElementById('btn-reparse-ast');
    const btnSync = document.getElementById('btn-sync-ast-tree');

    function renderTable(data) {
        if (!tableBody) return;
        tableBody.innerHTML = '';

        data.forEach(item => {
            const evalResult = engine.evaluateCloneCategory(item.treeSimilarity, item.editDistance);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${item.matchId}</strong></td>
                <td>${item.primaryFile}</td>
                <td>${item.compareFile}</td>
                <td>${(item.treeSimilarity * 100).toFixed(1)}%</td>
                <td>${item.editDistance} operations</td>
                <td><span class="badge ${evalResult.badgeClass}">${item.cloneClass}</span></td>
                <td>${item.status}</td>
                <td>
                    <button class="btn btn-sm btn-secondary" onclick="alert('Viewing AST Diff for ${item.matchId}')">Tree Diff</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    if (cutoffRange && cutoffLabel) {
        cutoffRange.addEventListener('input', (e) => {
            const val = e.target.value;
            cutoffLabel.textContent = `${val}%`;
            if (cutoffBadge) cutoffBadge.textContent = `Cutoff: ${val}%`;
            engine.updateConfig({ treeCutoffThreshold: val / 100 });
            renderTable(engine.getMatchesFiltered(searchInput ? searchInput.value : ''));
        });
    }

    if (minNodesRange && minNodesLabel) {
        minNodesRange.addEventListener('input', (e) => {
            minNodesLabel.textContent = `${e.target.value} nodes`;
        });
    }

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            renderTable(engine.getMatchesFiltered(e.target.value));
        });
    }

    if (btnReparse) {
        btnReparse.addEventListener('click', () => {
            btnReparse.classList.add('loading');
            setTimeout(() => {
                btnReparse.classList.remove('loading');
                renderTable(engine.getMatchesFiltered());
                alert('AST Syntax Tree rebuild completed.');
            }, 500);
        });
    }

    if (btnSync) {
        btnSync.addEventListener('click', () => {
            renderTable(engine.getMatchesFiltered());
        });
    }

    renderTable(engine.getMatchesFiltered());
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ASTSyntaxEngine };
}
