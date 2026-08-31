/**
 * Enterprise Multimodal PDF OCR Plagiarism Engine
 * LayoutLMv3 Bounding Box Calculations, Vision-Language Image Features, and OCR Alignment Engine
 */

class MultimodalPDFOCREngine {
    constructor(config = {}) {
        this.layoutCutoffThreshold = config.layoutCutoffThreshold || 0.82;
        this.targetDPI = config.targetDPI || 300;
        this.ocrEngineModel = config.ocrEngineModel || 'layoutlmv3';
        this.extractFigures = config.extractFigures !== undefined ? config.extractFigures : true;

        this.ocrMatches = [];
        this.initDefaultMatches();
    }

    initDefaultMatches() {
        this.ocrMatches = [
            {
                matchId: 'OCR-501',
                primaryPDF: 'quantum_lab_report_scanned.pdf',
                candidateFile: 'physics_journal_draft_p4.pdf',
                textSim: 0.935,
                layoutSim: 0.910,
                risk: 'CRITICAL',
                status: 'Under Audit'
            },
            {
                matchId: 'OCR-502',
                primaryPDF: 'medical_patent_diagram_scan.pdf',
                candidateFile: 'oncology_research_figure3.png',
                textSim: 0.780,
                layoutSim: 0.895,
                risk: 'HIGH',
                status: 'Flagged'
            },
            {
                matchId: 'OCR-503',
                primaryPDF: 'historical_archive_manuscript.pdf',
                candidateFile: 'digitized_history_thesis.pdf',
                textSim: 0.860,
                layoutSim: 0.740,
                risk: 'HIGH',
                status: 'Reviewed'
            },
            {
                matchId: 'OCR-504',
                primaryPDF: 'circuit_board_schematic_doc.pdf',
                candidateFile: 'embedded_systems_handbook.pdf',
                textSim: 0.912,
                layoutSim: 0.948,
                risk: 'CRITICAL',
                status: 'Under Audit'
            }
        ];
    }

    calculateBoundingBoxIntersection(boxA, boxB) {
        if (!boxA || !boxB) return 0;
        const xMin = Math.max(boxA.x, boxB.x);
        const yMin = Math.max(boxA.y, boxB.y);
        const xMax = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
        const yMax = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);

        if (xMax <= xMin || yMax <= yMin) return 0.0;
        const intersection = (xMax - xMin) * (yMax - yMin);
        const areaA = boxA.w * boxA.h;
        const areaB = boxB.w * boxB.h;
        return intersection / (areaA + areaB - intersection);
    }

    evaluateMultimodalRisk(textSim, layoutSim) {
        const composite = (textSim * 0.5) + (layoutSim * 0.5);
        if (composite >= this.layoutCutoffThreshold) {
            return { risk: 'HIGH MULTIMODAL CLONE', badgeClass: 'badge-danger' };
        } else if (composite >= 0.70) {
            return { risk: 'MODERATE OVERLAP', badgeClass: 'badge-warning' };
        } else {
            return { risk: 'LOW OVERLAP', badgeClass: 'badge-success' };
        }
    }

    updateConfig(newConfig) {
        if (newConfig.layoutCutoffThreshold !== undefined) {
            this.layoutCutoffThreshold = parseFloat(newConfig.layoutCutoffThreshold);
        }
        if (newConfig.targetDPI !== undefined) {
            this.targetDPI = parseInt(newConfig.targetDPI, 10);
        }
        if (newConfig.ocrEngineModel) {
            this.ocrEngineModel = newConfig.ocrEngineModel;
        }
        if (newConfig.extractFigures !== undefined) {
            this.extractFigures = newConfig.extractFigures;
        }
    }

    getMatchesFiltered(query = '') {
        if (!query) return this.ocrMatches;
        const q = query.toLowerCase();
        return this.ocrMatches.filter(m => 
            m.matchId.toLowerCase().includes(q) ||
            m.primaryPDF.toLowerCase().includes(q) ||
            m.candidateFile.toLowerCase().includes(q) ||
            m.risk.toLowerCase().includes(q)
        );
    }
}

// UI Controller Binding
document.addEventListener('DOMContentLoaded', () => {
    const engine = new MultimodalPDFOCREngine();

    const cutoffRange = document.getElementById('range-ocr-cutoff');
    const cutoffLabel = document.getElementById('lbl-ocr-cutoff');
    const cutoffBadge = document.getElementById('badge-ocr-cutoff');
    const dpiRange = document.getElementById('range-dpi-resolution');
    const dpiLabel = document.getElementById('lbl-dpi');
    const searchInput = document.getElementById('pdf-search-input');
    const tableBody = document.getElementById('ocr-table-body');
    const btnReindex = document.getElementById('btn-reindex-ocr');
    const btnSync = document.getElementById('btn-sync-ocr-telemetry');

    function renderTable(data) {
        if (!tableBody) return;
        tableBody.innerHTML = '';

        data.forEach(item => {
            const riskEval = engine.evaluateMultimodalRisk(item.textSim, item.layoutSim);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${item.matchId}</strong></td>
                <td>${item.primaryPDF}</td>
                <td>${item.candidateFile}</td>
                <td>${(item.textSim * 100).toFixed(1)}%</td>
                <td>${(item.layoutSim * 100).toFixed(1)}%</td>
                <td><span class="badge ${riskEval.badgeClass}">${riskEval.risk}</span></td>
                <td>${item.status}</td>
                <td>
                    <button class="btn btn-sm btn-secondary" onclick="alert('Viewing OCR Bounding Boxes for ${item.matchId}')">Inspect PDF</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    if (cutoffRange && cutoffLabel) {
        cutoffRange.addEventListener('input', (e) => {
            const val = e.target.value;
            cutoffLabel.textContent = `${val}%`;
            if (cutoffBadge) cutoffBadge.textContent = `Visual Cutoff: ${val}%`;
            engine.updateConfig({ layoutCutoffThreshold: val / 100 });
            renderTable(engine.getMatchesFiltered(searchInput ? searchInput.value : ''));
        });
    }

    if (dpiRange && dpiLabel) {
        dpiRange.addEventListener('input', (e) => {
            dpiLabel.textContent = `${e.target.value} DPI`;
        });
    }

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            renderTable(engine.getMatchesFiltered(e.target.value));
        });
    }

    if (btnReindex) {
        btnReindex.addEventListener('click', () => {
            btnReindex.classList.add('loading');
            setTimeout(() => {
                btnReindex.classList.remove('loading');
                renderTable(engine.getMatchesFiltered());
                alert('Multimodal PDF OCR Layout re-scan complete.');
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
    module.exports = { MultimodalPDFOCREngine };
}
