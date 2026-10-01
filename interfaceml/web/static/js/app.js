/**
 * InterfaceML Web Application
 * Client-side JavaScript for heterojunction modeling interface
 */

// State management
const state = {
    uploadedFiles: {},
    currentSection: 'structure-builder',
    sidebarCollapsed: false,
    mobileSidebarOpen: false,
    builderMode: 'adsorbate',
    analysisMode: 'fix',
    layerMode: 'fix',  // 'fix', 'split', or 'pdos'
    pdosFiles: [],
    pdosCounter: 0,
    aiAvailable: false,
    selectedModel: 'egnn',
    availableModels: {},
    currentStructures: [],
    generationHistory: [],
    pipelineSource: 'upload',
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeSidebar();
    initializeFileUploads();
    initializeButtons();
    initializeSubModeToggles();
    initializeModeSwitcher();
    initializePdosUploads();
    checkServerHealth();
    initializeAITab();
    initializePipelineTab();
    initializeTerminationToggle();
});

function initializeTerminationToggle() {
    const sel = document.getElementById('termination');
    const custom = document.getElementById('termination-custom');
    if (!sel || !custom) return;
    const update = () => {
        if (sel.value === '__custom__') {
            custom.classList.remove('hidden');
            custom.focus();
        } else {
            custom.classList.add('hidden');
        }
    };
    sel.addEventListener('change', update);
    update();
}

// Section metadata for page header
const sectionMeta = {
    'structure-builder': { title: 'Structure Builder', desc: 'Build heterojunction interfaces and adsorbate models' },
    'ai-generation': { title: 'AI Generation', desc: 'Generate fullerene structures with deep learning models' },
    'analysis': { title: 'Analysis Tools', desc: 'Fix layers, split structures, and compute density/DOS' },
    'dft-pipeline': { title: 'DFT Pipeline', desc: 'Build interfaces and generate CP2K input files' },
    'algorithm': { title: 'Algorithm Overview', desc: 'Model architectures, equations, and pipeline diagrams' },
};

function updatePageHeader(sectionId) {
    const meta = sectionMeta[sectionId];
    if (!meta) return;
    const titleEl = document.getElementById('page-header-title');
    const descEl = document.getElementById('page-header-desc');
    if (!titleEl || !descEl) return;
    // Fade out, swap text, fade in
    titleEl.style.opacity = '0';
    descEl.style.opacity = '0';
    setTimeout(() => {
        titleEl.textContent = meta.title;
        descEl.textContent = meta.desc;
        titleEl.style.opacity = '1';
        descEl.style.opacity = '1';
    }, 150);
}

// ============================================
// SIDEBAR NAVIGATION
// ============================================

function initializeSidebar() {
    const navItems = document.querySelectorAll('.sidebar-nav-item[data-section]');
    const sections = document.querySelectorAll('.page-section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const sectionId = item.dataset.section;

            // Update nav active state
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Switch section with animation
            sections.forEach(s => s.classList.remove('active'));
            const target = document.getElementById(sectionId);
            if (target) {
                target.classList.add('active');
                // Scroll main area to top on section switch
                const mainArea = document.querySelector('.main-area');
                if (mainArea) mainArea.scrollTo({ top: 0, behavior: 'smooth' });
                else window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            state.currentSection = sectionId;

            // Update page header
            updatePageHeader(sectionId);

            // Close mobile sidebar
            if (state.mobileSidebarOpen) {
                closeMobileSidebar();
            }
        });
    });

    // Collapse toggle
    const collapseBtn = document.getElementById('sidebar-collapse-btn');
    if (collapseBtn) {
        // Restore saved state
        const saved = localStorage.getItem('sidebar-collapsed');
        if (saved === 'true') {
            document.body.classList.add('sidebar-collapsed');
            state.sidebarCollapsed = true;
        }

        collapseBtn.addEventListener('click', () => {
            state.sidebarCollapsed = !state.sidebarCollapsed;
            document.body.classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
            localStorage.setItem('sidebar-collapsed', state.sidebarCollapsed);
        });
    }

    // Mobile hamburger
    const hamburger = document.getElementById('mobile-hamburger');
    if (hamburger) {
        hamburger.addEventListener('click', openMobileSidebar);
    }

    // Mobile overlay
    const overlay = document.getElementById('sidebar-overlay');
    if (overlay) {
        overlay.addEventListener('click', closeMobileSidebar);
    }
}

function openMobileSidebar() {
    state.mobileSidebarOpen = true;
    document.body.classList.add('mobile-open');
}

function closeMobileSidebar() {
    state.mobileSidebarOpen = false;
    document.body.classList.remove('mobile-open');
}

// ============================================
// SUB-MODE TOGGLES
// ============================================

function initializeSubModeToggles() {
    // Structure Builder sub-modes
    const builderBtns = document.querySelectorAll('[data-builder-mode]');
    builderBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.builderMode;
            builderBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('#structure-builder .sub-mode-content').forEach(c => c.classList.remove('active'));
            const target = document.getElementById(`builder-${mode}`);
            if (target) target.classList.add('active');

            state.builderMode = mode;
        });
    });

    // Analysis sub-modes
    const analysisBtns = document.querySelectorAll('[data-analysis-mode]');
    analysisBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.analysisMode;
            analysisBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('#analysis .sub-mode-content').forEach(c => c.classList.remove('active'));
            const target = document.getElementById(`analysis-${mode}`);
            if (target) target.classList.add('active');

            state.analysisMode = mode;
            // Map analysis mode to legacy layerMode
            if (mode === 'fix' || mode === 'split' || mode === 'pdos') {
                state.layerMode = mode;
            }
        });
    });
}

// ============================================
// FILE UPLOADS
// ============================================

function initializeFileUploads() {
    setupFileUpload('base-file', 'base', displayFileInfo);
    setupFileUpload('adsorbate-file', 'adsorbate', (fileType, data) => {
        if (!Array.isArray(state.uploadedFiles.adsorbateList)) {
            state.uploadedFiles.adsorbateList = [];
        }
        state.uploadedFiles.adsorbateList.push(data);
        displayAdsorbateList();
    });
    setupFileUpload('substrate-file', 'substrate', displayFileInfo);
    setupFileUpload('film-file', 'film', displayFileInfo);
    setupFileUpload('layers-file', 'layers', displayLayersFileInfo);
    setupFileUpload('tdos-file', 'tdos', displayDosFileInfo);
}

function setupFileUpload(inputId, fileType, callback) {
    const input = document.getElementById(inputId);
    if (!input) return;

    input.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (files.length === 0) return;
        for (let file of files) {
            await uploadFile(file, fileType, callback);
        }
    });

    const uploadArea = input.closest('.file-upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            for (let file of files) {
                await uploadFile(file, fileType, callback);
            }
        });
    }
}

async function uploadFile(file, fileType, callback) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', fileType);

    try {
        showLoading(true);
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const payload = await response.json();

        if (response.ok) {
            const data = payload && payload.file_info ? payload.file_info : payload;
            state.uploadedFiles[fileType] = data;
            callback(fileType, data);
            showToast(`${file.name} uploaded successfully`, 'success');
        } else {
            showToast(`Upload failed: ${payload.error}`, 'error');
        }
    } catch (error) {
        showToast(`Upload error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

function displayFileInfo(fileType, data) {
    const infoDiv = document.getElementById(`${fileType}-info`);
    if (!infoDiv) return;

    infoDiv.classList.remove('hidden');
    infoDiv.innerHTML = `
        <strong>${data.filename}</strong><br>
        ${data.composition ? `Composition: ${data.composition}<br>` : ''}
        ${data.n_atoms ? `Atoms: ${data.n_atoms}<br>` : ''}
        ${data.lattice_abc ? `Cell: ${data.lattice_abc.join(' x ')} A<br>` : ''}
        ${data.parse_error ? `<span style="color: #ef4444;">Parse error: ${data.parse_error}</span>` : ''}
    `;
}

function displayLayersFileInfo(fileType, data) {
    const infoDiv = document.getElementById('layers-file-info');
    if (!infoDiv) return;

    infoDiv.classList.remove('hidden');
    infoDiv.innerHTML = `
        <strong>${data.filename}</strong><br>
        ${data.composition ? `Composition: ${data.composition}<br>` : ''}
        ${data.n_atoms ? `Total atoms: ${data.n_atoms}` : ''}
    `;
}

function displayDosFileInfo(fileType, data) {
    const infoDiv = document.getElementById(`${fileType}-info`);
    if (!infoDiv) return;

    infoDiv.classList.remove('hidden');
    infoDiv.innerHTML = `
        <strong>${data.filename}</strong><br>
        ${data.file_hash ? `SHA-256: ${data.file_hash}<br>` : ''}
        ${data.parse_error ? `<span style="color: #ef4444;">Parse error: ${data.parse_error}</span>` : ''}
    `;
}

// ============================================
// MODE SWITCHER (legacy layer management)
// ============================================

function initializeModeSwitcher() {
    // Legacy mode-btn handling is now done via analysis sub-mode toggles
    // but keep the mode-btn handler for any remaining mode-selector usage
    const modeButtons = document.querySelectorAll('.mode-selector:not(.ai-mode-selector):not(.pipeline-source-selector) .mode-btn[data-mode]');

    modeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const mode = button.dataset.mode;
            const parent = button.closest('.mode-selector');
            if (!parent) return;

            parent.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            const fixParams = document.getElementById('fix-mode-params');
            const splitParams = document.getElementById('split-mode-params');
            const pdosParams = document.getElementById('pdos-mode-params');

            if (fixParams && splitParams && pdosParams) {
                if (mode === 'fix') {
                    fixParams.classList.remove('hidden');
                    splitParams.classList.add('hidden');
                    pdosParams.classList.add('hidden');
                    state.layerMode = 'fix';
                } else if (mode === 'split') {
                    fixParams.classList.add('hidden');
                    splitParams.classList.remove('hidden');
                    pdosParams.classList.add('hidden');
                    state.layerMode = 'split';
                } else if (mode === 'pdos') {
                    fixParams.classList.add('hidden');
                    splitParams.classList.add('hidden');
                    pdosParams.classList.remove('hidden');
                    state.layerMode = 'pdos';
                }
            }

            const resultDiv = document.getElementById('layers-result');
            if (resultDiv) resultDiv.classList.add('hidden');
        });
    });
}

// ============================================
// ACTION BUTTONS
// ============================================

function initializeButtons() {
    const btnMap = {
        'build-adsorbate-btn': buildAdsorbate,
        'build-interface-btn': buildInterface,
        'iface-scan-terms-btn': scanInterfaceTerminations,
        'fix-layers-btn': fixLayers,
        'split-layers-btn': splitLayers,
        'pdos-layers-btn': generatePdosLayers,
        'compute-density-btn': computeDensity,
        'plot-dos-btn': plotDos,
    };

    Object.entries(btnMap).forEach(([id, handler]) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', handler);
    });
}

// ============================================
// BUILD ACTIONS
// ============================================

function displayAdsorbateList() {
    const infoDiv = document.getElementById('adsorbate-info');
    if (!infoDiv) return;
    const list = state.uploadedFiles.adsorbateList || [];
    infoDiv.replaceChildren();
    if (list.length === 0) {
        infoDiv.classList.add('hidden');
        return;
    }
    infoDiv.classList.remove('hidden');
    list.forEach((d, i) => {
        const row = document.createElement('div');
        row.append(`${i + 1}. `);
        const name = document.createElement('strong');
        name.textContent = String(d.filename ?? '(unnamed)');
        row.append(name);
        if (d.n_atoms) row.append(` — ${Number(d.n_atoms)} atoms `);
        const rm = document.createElement('a');
        rm.href = '#';
        rm.textContent = '×';
        rm.style.cssText = 'color:#ef4444;margin-left:6px;';
        rm.addEventListener('click', (e) => {
            e.preventDefault();
            state.uploadedFiles.adsorbateList.splice(i, 1);
            displayAdsorbateList();
        });
        row.append(rm);
        infoDiv.append(row);
    });
}

async function buildAdsorbate() {
    const adsList = state.uploadedFiles.adsorbateList || [];
    if (!state.uploadedFiles.base || adsList.length === 0) {
        showToast('Please upload a base file and at least one adsorbate', 'error');
        return;
    }

    const xyRaw = (document.getElementById('xy-frac')?.value || '0.5 0.5').trim();
    const xyParts = xyRaw.replace(',', ' ').split(/\s+/).map(Number);
    if (xyParts.length !== 2 || xyParts.some(isNaN)) {
        showToast('XY frac must be two numbers (e.g. "0.5 0.5")', 'error');
        return;
    }

    const termSelect = document.getElementById('termination').value;
    let terminationAtoms = null;
    let terminationLabel = 'auto';
    if (termSelect === '__custom__') {
        const custom = (document.getElementById('termination-custom')?.value || '').trim();
        if (!custom) {
            showToast('Enter custom termination elements (e.g. "Cs,I")', 'error');
            return;
        }
        terminationAtoms = custom;
        terminationLabel = custom;
    } else if (termSelect && termSelect !== 'auto') {
        terminationAtoms = termSelect;
        terminationLabel = termSelect;
    }

    const data = {
        base_file: state.uploadedFiles.base.filepath,
        adsorbate_files: adsList.map(d => d.filepath),
        termination: terminationLabel,
        termination_atoms: terminationAtoms,
        distance: parseFloat(document.getElementById('distance').value),
        miller: document.getElementById('miller').value.trim().split(/\s+/).map(Number),
        supercell: document.getElementById('supercell').value,
        rotation_deg: parseFloat(document.getElementById('rotation-deg')?.value || '0'),
        xy_frac: xyParts,
        stack_mode: document.getElementById('stack-mode')?.value || 'separate'
    };

    try {
        showLoading(true);
        const response = await fetch('/api/build-adsorbate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        renderAdsorbateResult('adsorbate-result', result, response.ok);
    } catch (error) {
        renderAdsorbateResult('adsorbate-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

function _findSlabTopZ(atoms) {
    if (!atoms.length) return null;
    const zs = atoms.map(a => a.z).slice().sort((a, b) => a - b);
    let maxGap = 0, gapIdx = -1;
    for (let i = 1; i < zs.length; i++) {
        const g = zs[i] - zs[i - 1];
        if (g > maxGap) { maxGap = g; gapIdx = i; }
    }
    // Gap > 2Å usually means slab-vacuum-adsorbate split
    if (maxGap > 2.0 && gapIdx > 0) return zs[gapIdx - 1];
    return zs[zs.length - 1];
}

function applyTopLayerHighlight(viewer, on) {
    const atoms = viewer.selectedAtoms({});
    const baseStyle = { sphere: { scale: 0.28 }, stick: { radius: 0.12 } };
    viewer.setStyle({}, baseStyle);
    if (on) {
        const slabTopZ = _findSlabTopZ(atoms);
        if (slabTopZ != null) {
            const topIdx = atoms
                .filter(a => slabTopZ - a.z >= -0.05 && slabTopZ - a.z < 1.5)
                .map(a => a.index);
            if (topIdx.length) {
                viewer.addStyle({ index: topIdx },
                    { sphere: { scale: 0.55, color: '#fbbf24', opacity: 0.35 } });
            }
        }
    }
    viewer.render();
}

async function loadStructureViewer(div, url) {
    if (!window.$3Dmol) throw new Error('3Dmol.js not loaded');
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const viewer = $3Dmol.createViewer(div, { backgroundColor: '#0f172a' });
    viewer.addModel(text, 'vasp');
    viewer.setStyle({}, { sphere: { scale: 0.28 }, stick: { radius: 0.12 } });
    try { viewer.addUnitCell(); } catch (_) {}
    try { viewer.setProjection('orthographic'); } catch (_) {}
    viewer.zoomTo();
    viewer.rotate(-90, 'x');
    viewer.rotate(-8, 'y');
    viewer.zoom(1.15);
    viewer.render();
    return viewer;
}

function renderAdsorbateResult(elementId, result, success) {
    const root = document.getElementById(elementId);
    if (!root) return;
    root.classList.remove('hidden');
    root.replaceChildren();

    if (!success || (result && result.status === 'failed')) {
        root.classList.add('error');
        const h = document.createElement('h4'); h.textContent = 'Error';
        const p = document.createElement('p');
        p.textContent = (result && (result.error || result.message)) || 'Build failed';
        root.append(h, p);
        if (Array.isArray(result?.results)) {
            const ul = document.createElement('ul');
            result.results.filter(r => r.status === 'failed').forEach(r => {
                const li = document.createElement('li');
                li.textContent = `${r.adsorbate}: ${r.error}`;
                ul.append(li);
            });
            if (ul.children.length) root.append(ul);
        }
        return;
    }
    root.classList.remove('error');

    const header = document.createElement('div');
    header.className = 'result-header';
    const title = document.createElement('h4');
    title.textContent = result.message || 'Build complete';
    header.append(title);
    const meta = document.createElement('div');
    meta.className = 'result-meta';
    meta.textContent = [
        result.run_id ? `run ${result.run_id}` : null,
        result.stack_mode ? `mode ${result.stack_mode}` : null,
        result.n_built != null ? `built ${result.n_built}/${result.n_total}` : null,
    ].filter(Boolean).join(' · ');
    header.append(meta);
    root.append(header);
    (result.warnings || []).forEach(message => {
        const warning = document.createElement('p');
        warning.className = 'result-meta';
        warning.textContent = message;
        root.append(warning);
    });

    const items = [];
    if (Array.isArray(result.results)) {
        result.results.forEach(r => {
            if (r.status === 'success' && r.download_url) {
                items.push({
                    title: r.adsorbate || r.filename || 'structure',
                    n_atoms: r.n_atoms, supercell: r.supercell, termination: r.termination,
                    strain_pct: r.strain_pct, area: r.area, rank: r.rank,
                    download_url: r.download_url, filename: r.filename,
                });
            }
        });
    }
    if (result.stack_mode === 'stacked' && result.download_url) {
        items.push({
            title: `Stacked × ${result.n_stacked || ''}`.trim(),
            n_atoms: result.n_atoms, supercell: result.supercell, termination: result.termination,
            download_url: result.download_url, filename: result.filename,
        });
    }

    if (items.length === 0) {
        const p = document.createElement('p');
        p.textContent = 'No structure files produced.';
        root.append(p);
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'struct-grid';
    root.append(grid);

    items.forEach((it, idx) => {
        const card = document.createElement('div');
        card.className = 'struct-card';

        const cardHeader = document.createElement('div');
        cardHeader.className = 'struct-card-header';
        const t = document.createElement('strong');
        t.textContent = String(it.title);
        cardHeader.append(t);
        const btnGroup = document.createElement('div');
        btnGroup.style.cssText = 'display:flex;gap:0.4rem;align-items:center;';

        const hl = document.createElement('button');
        hl.type = 'button';
        hl.className = 'btn btn-sm struct-highlight-btn';
        hl.textContent = 'Highlight top';
        hl.style.cssText = 'background:#1e293b;color:#e2e8f0;border:1px solid #334155;padding:0.32rem 0.6rem;border-radius:8px;font-size:0.72rem;cursor:pointer;';
        let hlOn = false;
        hl.dataset.viewerIdx = String(idx);
        btnGroup.append(hl);

        const dl = document.createElement('a');
        dl.href = it.download_url;
        dl.className = 'btn btn-primary btn-sm';
        if (it.filename) dl.download = String(it.filename);
        dl.textContent = '↓ Download';
        btnGroup.append(dl);
        cardHeader.append(btnGroup);
        card.append(cardHeader);

        const sub = document.createElement('div');
        sub.className = 'struct-meta';
        sub.textContent = [
            it.n_atoms != null ? `${it.n_atoms} atoms` : null,
            it.supercell ? `cell ${it.supercell}` : null,
            it.strain_pct != null ? `strain ${it.strain_pct}%` : null,
            it.area != null ? `area ${it.area}Å²` : null,
            it.termination ? `term ${it.termination}` : null,
        ].filter(Boolean).join(' · ');
        card.append(sub);

        const viewer = document.createElement('div');
        viewer.className = 'struct-viewer';
        viewer.id = `viewer-${idx}-${Date.now()}`;
        card.append(viewer);
        grid.append(card);

        loadStructureViewer(viewer, it.download_url).then(viewerInstance => {
            hl.addEventListener('click', () => {
                hlOn = !hlOn;
                applyTopLayerHighlight(viewerInstance, hlOn);
                hl.textContent = hlOn ? '✦ Top layer on' : 'Highlight top';
                hl.style.background = hlOn ? '#fbbf24' : '#1e293b';
                hl.style.color = hlOn ? '#0f172a' : '#e2e8f0';
            });
        }).catch(e => {
            hl.disabled = true;
            const err = document.createElement('div');
            err.className = 'struct-error';
            err.textContent = `Render failed: ${e.message}`;
            viewer.replaceChildren(err);
        });
    });
}

async function scanInterfaceTerminations() {
    if (!state.uploadedFiles.substrate || !state.uploadedFiles.film) {
        showToast('Upload substrate and film first', 'error');
        return;
    }
    const status = document.getElementById('iface-term-status');
    if (status) status.textContent = 'Scanning…';

    const payload = {
        base_file: state.uploadedFiles.substrate.filepath,
        film_file: state.uploadedFiles.film.filepath,
        miller_base: document.getElementById('miller-sub').value.trim().split(/\s+/).map(Number),
        miller_film: document.getElementById('miller-film').value.trim().split(/\s+/).map(Number),
        max_area: parseFloat(document.getElementById('max-area').value),
        matching_mode: document.getElementById('strain-mode').value,
    };

    try {
        showLoading(true);
        const r = await fetch('/api/list-interface-terminations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const j = await r.json();
        if (!r.ok || j.status !== 'success') {
            const msg = (j && (j.error || j.message)) || `HTTP ${r.status}`;
            if (status) status.textContent = `Failed: ${msg}`;
            showToast(`Scan failed: ${msg}`, 'error');
            return;
        }
        const fillSelect = (id, terms) => {
            const sel = document.getElementById(id);
            if (!sel) return;
            const cur = sel.value;
            sel.replaceChildren();
            const auto = document.createElement('option');
            auto.value = ''; auto.textContent = 'Auto (scan top pairs)';
            sel.append(auto);
            terms.forEach(t => {
                const o = document.createElement('option');
                o.value = String(t.label);
                const f = String(t.formula || t.label);
                const sg = t.spacegroup ? ` · ${t.spacegroup}` : '';
                const inst = t.instance ? ` #${t.instance}` : '';
                o.textContent = `${f}${sg}${inst}`;
                sel.append(o);
            });
            sel.value = cur || '';
        };
        fillSelect('iface-sub-term', j.substrate_terminations || []);
        fillSelect('iface-film-term', j.film_terminations || []);
        if (status) {
            status.textContent = `${j.n_pairs} pairs · ${j.n_zsl_matches} ZSL match(es) · ${(j.substrate_terminations||[]).length} sub × ${(j.film_terminations||[]).length} film terms`;
        }
    } catch (err) {
        if (status) status.textContent = `Error: ${err.message}`;
        showToast(`Scan error: ${err.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function buildInterface() {
    if (!state.uploadedFiles.substrate || !state.uploadedFiles.film) {
        showToast('Please upload both substrate and film files', 'error');
        return;
    }

    const data = {
        base_file: state.uploadedFiles.substrate.filepath,
        film_file: state.uploadedFiles.film.filepath,
        miller_base: document.getElementById('miller-sub').value.trim().split(/\s+/).map(Number),
        miller_film: document.getElementById('miller-film').value.trim().split(/\s+/).map(Number),
        max_area: parseFloat(document.getElementById('max-area').value),
        matching_mode: document.getElementById('strain-mode').value,
        gap: parseFloat(document.getElementById('iface-gap')?.value || '2.5'),
        vacuum: parseFloat(document.getElementById('iface-vacuum')?.value || '20.0'),
        top_k: parseInt(document.getElementById('iface-topk')?.value || '3', 10),
        max_termination_pairs: parseInt(document.getElementById('iface-term-pairs')?.value || '3', 10),
        substrate_termination: document.getElementById('iface-sub-term')?.value || '',
        film_termination: document.getElementById('iface-film-term')?.value || '',
        twist_deg: parseFloat(document.getElementById('iface-twist-deg')?.value || '0'),
        xy_shift_x: parseFloat(document.getElementById('iface-xy-shift-x')?.value || '0'),
        xy_shift_y: parseFloat(document.getElementById('iface-xy-shift-y')?.value || '0'),
        xy_units: document.getElementById('iface-xy-units')?.value || 'angstrom',
    };

    try {
        showLoading(true);
        const response = await fetch('/api/build-interface', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        renderAdsorbateResult('interface-result', result, response.ok);
    } catch (error) {
        renderAdsorbateResult('interface-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

// ============================================
// LAYER OPERATIONS
// ============================================

async function fixLayers() {
    if (!state.uploadedFiles.layers) {
        showToast('Please upload a structure file', 'error');
        return;
    }

    const data = {
        structure_file: state.uploadedFiles.layers.filepath,
        mode: 'by_z_layers',
        n_fix_layers: parseInt(document.getElementById('n-fix-layers').value),
        include_molecules: document.getElementById('include-molecules').checked
    };

    try {
        showLoading(true);
        const response = await fetch('/api/fix-layers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (response.ok && result.status === 'success') {
            displayLayersResult(result);
        } else {
            displayResult('layers-result', result, false);
        }
    } catch (error) {
        displayResult('layers-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

function displayLayersResult(result) {
    const resultDiv = document.getElementById('layers-result');
    resultDiv.classList.remove('hidden', 'error');
    resultDiv.innerHTML = `
        <h4>Selective Dynamics Added</h4>
        <p><strong>Total layers detected:</strong> ${result.n_layers_total}</p>
        <p><strong>Layers fixed:</strong> ${result.n_layers_fixed}</p>
        <p><strong>Atoms fixed:</strong> ${result.n_atoms_fixed}</p>
        <p><strong>Output file:</strong> <a href="${result.download_url}" download>Download POSCAR</a></p>
        <details>
            <summary>Fixed atom indices (0-based)</summary>
            <pre style="max-height: 200px; overflow-y: auto; background: white; padding: 1rem; border-radius: 0.25rem;">${result.fixed_indices.join(', ')}</pre>
        </details>
    `;
}

async function splitLayers() {
    if (!state.uploadedFiles.layers) {
        showToast('Please upload a structure file', 'error');
        return;
    }

    const data = {
        structure_file: state.uploadedFiles.layers.filepath,
        n_interfaces: parseInt(document.getElementById('n-interfaces').value),
        min_gap: parseFloat(document.getElementById('min-gap').value),
        use_smart_detection: document.getElementById('smart-detection').checked
    };

    try {
        showLoading(true);
        const response = await fetch('/api/split-layers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (response.ok && result.status === 'success') {
            displaySplitResult(result);
        } else {
            displayResult('layers-result', result, false);
        }
    } catch (error) {
        displayResult('layers-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

async function generatePdosLayers() {
    if (!state.uploadedFiles.layers) {
        showToast('Please upload a structure file', 'error');
        return;
    }
    const filenameValue = document.getElementById('pdos-filename').value.trim();

    const data = {
        structure_file: state.uploadedFiles.layers.filepath,
        n_interfaces: parseInt(document.getElementById('pdos-n-interfaces').value, 10),
        min_gap: parseFloat(document.getElementById('pdos-min-gap').value),
        use_smart_detection: document.getElementById('pdos-smart-detection').checked,
        pdos_filename: filenameValue === '' ? null : filenameValue,
        nlumo: parseInt(document.getElementById('pdos-nlumo').value, 10)
    };

    try {
        showLoading(true);
        const response = await fetch('/api/pdos-layers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (response.ok && result.status === 'success') {
            displayPdosResult(result);
        } else {
            displayResult('layers-result', result, false);
        }
    } catch (error) {
        displayResult('layers-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

function displaySplitResult(result) {
    const resultDiv = document.getElementById('layers-result');
    resultDiv.classList.remove('hidden', 'error');

    let layersHTML = '';
    result.layers.forEach((layer, index) => {
        let layerIcon = '&#9632;';
        if (layer.layer_type && layer.layer_type.includes('Perovskite')) layerIcon = '&#9670;';
        else if (layer.layer_type && layer.layer_type.includes('C60')) layerIcon = '&#9679;';

        layersHTML += `
            <div class="layer-item">
                <div class="layer-header">
                    <span class="layer-icon">${layerIcon}</span>
                    <strong>Layer ${layer.layer_number}</strong>
                </div>
                <div class="layer-details">
                    <div class="layer-type">${layer.layer_type || 'Unknown'}</div>
                    <div class="layer-stats">
                        <span>Atoms: ${layer.n_atoms}</span>
                        ${layer.z_range ? `<span>Height: ${layer.z_range}</span>` : ''}
                    </div>
                    <div class="layer-composition">${layer.composition}</div>
                </div>
                <a href="${layer.download_url}" class="download-link" download>
                    Download ${layer.filename}
                </a>
            </div>
        `;
    });

    resultDiv.innerHTML = `
        <h4>Structure Split Successfully</h4>
        ${result.run_id ? `<p><strong>Run ID:</strong> ${result.run_id}</p>` : ''}
        <p><strong>Total interfaces detected:</strong> ${result.n_interfaces}</p>
        <p><strong>Layers created:</strong> ${result.n_layers}</p>
        <div class="layers-grid">${layersHTML}</div>
        <p class="result-note"><em>Each layer preserves the original lattice parameters.</em></p>
    `;
}

function displayPdosResult(result) {
    const resultDiv = document.getElementById('layers-result');
    resultDiv.classList.remove('hidden', 'error');

    const pdosBlock = result.cp2k_pdos || '';
    const rangesText = result.layers
        .map(layer => `Layer ${layer.layer_number}: ${layer.ranges_1}`)
        .join('\n');

    let layersHTML = '';
    result.layers.forEach(layer => {
        layersHTML += `
            <div class="layer-item">
                <div class="layer-header">
                    <span class="layer-icon">&#9670;</span>
                    <strong>Layer ${layer.layer_number}</strong>
                </div>
                <div class="layer-details">
                    <div class="layer-type">Atoms: ${layer.n_atoms}</div>
                    <div class="layer-stats">
                        <span>Range: ${layer.ranges_1 || 'N/A'}</span>
                        ${layer.z_range ? `<span>Height: ${layer.z_range}</span>` : ''}
                    </div>
                </div>
                <div class="layer-actions">
                    <button class="btn btn-secondary btn-sm" data-copy-layer="${layer.layer_number}">Copy Layer LIST</button>
                </div>
                <div class="layer-code">${escapeHtml(layer.ranges_1)}</div>
            </div>
        `;
    });

    resultDiv.innerHTML = `
        <h4>PDOS Layer Indices Ready</h4>
        <p><strong>Interfaces:</strong> ${result.n_interfaces}</p>
        <p><strong>Layers detected:</strong> ${result.n_layers}</p>
        <div class="pdos-actions">
            <button class="btn btn-secondary btn-sm" data-copy="pdos">Copy CP2K PDOS Block</button>
            <button class="btn btn-secondary btn-sm" data-copy="ranges">Copy Layer Ranges</button>
        </div>
        <div class="code-block" id="pdos-block">${escapeHtml(pdosBlock)}</div>
        <div class="layers-grid">${layersHTML}</div>
    `;

    const pdosButton = resultDiv.querySelector('[data-copy="pdos"]');
    if (pdosButton) {
        pdosButton.addEventListener('click', () => {
            copyToClipboard(pdosBlock);
            showToast('CP2K PDOS block copied', 'success');
        });
    }

    const rangesButton = resultDiv.querySelector('[data-copy="ranges"]');
    if (rangesButton) {
        rangesButton.addEventListener('click', () => {
            copyToClipboard(rangesText);
            showToast('Layer ranges copied', 'success');
        });
    }

    const layerButtons = resultDiv.querySelectorAll('[data-copy-layer]');
    layerButtons.forEach(button => {
        button.addEventListener('click', () => {
            const layerNumber = button.getAttribute('data-copy-layer');
            const layer = result.layers.find(l => String(l.layer_number) === String(layerNumber));
            if (layer) {
                copyToClipboard(layer.ranges_1 || '');
                showToast(`Layer ${layer.layer_number} LIST copied`, 'success');
            }
        });
    });
}

async function computeDensity() {
    showToast('Density analysis functionality coming soon', 'info');
}

// ============================================
// PDOS DYNAMIC UPLOADS
// ============================================

function initializePdosUploads() {
    const addPdosBtn = document.getElementById('add-pdos-btn');
    if (addPdosBtn) {
        addPdosBtn.addEventListener('click', addPdosUpload);
    }
    addPdosUpload();
    addPdosUpload();
}

function addPdosUpload() {
    state.pdosCounter++;
    const pdosId = state.pdosCounter;
    const container = document.getElementById('pdos-upload-container');
    if (!container) return;

    const pdosItem = document.createElement('div');
    pdosItem.className = 'pdos-upload-item';
    pdosItem.id = `pdos-item-${pdosId}`;
    pdosItem.innerHTML = `
        <div class="pdos-upload-row">
            <div class="file-upload-area pdos-upload-area" id="pdos${pdosId}-upload">
                <input type="file" id="pdos${pdosId}-file" accept=".pdos" hidden>
                <label for="pdos${pdosId}-file" class="file-upload-label">
                    <span class="upload-text">Upload PDOS list ${pdosId} (.pdos)</span>
                </label>
            </div>
            <button type="button" class="btn btn-danger btn-sm btn-remove-pdos" data-pdos-id="${pdosId}" title="Remove">
                &times;
            </button>
        </div>
        <div id="pdos${pdosId}-info" class="file-info hidden"></div>
    `;

    container.appendChild(pdosItem);

    const fileInput = document.getElementById(`pdos${pdosId}-file`);
    const uploadArea = document.getElementById(`pdos${pdosId}-upload`);

    fileInput.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (files.length === 0) return;
        await uploadPdosFile(files[0], pdosId);
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });
    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) await uploadPdosFile(files[0], pdosId);
    });

    const removeBtn = pdosItem.querySelector('.btn-remove-pdos');
    removeBtn.addEventListener('click', () => removePdosUpload(pdosId));
}

async function uploadPdosFile(file, pdosId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', `pdos${pdosId}`);

    try {
        showLoading(true);
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok) {
            const existing = state.pdosFiles.findIndex(p => p.id === pdosId);
            if (existing >= 0) {
                state.pdosFiles[existing] = { id: pdosId, ...data };
            } else {
                state.pdosFiles.push({ id: pdosId, ...data });
            }
            state.uploadedFiles[`pdos${pdosId}`] = data;
            displayDosFileInfo(`pdos${pdosId}`, data);
            showToast(`${file.name} uploaded successfully`, 'success');
        } else {
            showToast(`Upload failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Upload error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

function removePdosUpload(pdosId) {
    const pdosItem = document.getElementById(`pdos-item-${pdosId}`);
    if (pdosItem) pdosItem.remove();
    state.pdosFiles = state.pdosFiles.filter(p => p.id !== pdosId);
    delete state.uploadedFiles[`pdos${pdosId}`];
}

// ============================================
// DOS PLOTTING
// ============================================

async function plotDos() {
    if (!state.uploadedFiles.tdos) {
        showToast('Please upload a TDOS file', 'error');
        return;
    }
    if (state.pdosFiles.length === 0) {
        showToast('Please upload at least one PDOS file', 'error');
        return;
    }

    const pdosFilePaths = state.pdosFiles.sort((a, b) => a.id - b.id).map(p => p.filepath);

    const data = {
        tdos_file: state.uploadedFiles.tdos.filepath,
        pdos_files: pdosFilePaths,
        title: document.getElementById('dos-title').value.trim() || null,
        output_name: document.getElementById('dos-output-name').value.trim() || null,
        sigma: parseFloat(document.getElementById('dos-sigma').value),
        grid_step: parseFloat(document.getElementById('dos-grid-step').value),
        normalize: document.getElementById('dos-normalize').checked,
        x_min: document.getElementById('dos-x-min').value.trim() || null,
        x_max: document.getElementById('dos-x-max').value.trim() || null,
        tdos_source: document.getElementById('dos-tdos-source').value,
        pdos_scale: document.getElementById('dos-pdos-scale').value,
        tdos_scale: document.getElementById('dos-tdos-scale').value,
        tdos_total_atoms: document.getElementById('dos-tdos-atoms').value.trim() || null
    };

    try {
        showLoading(true);
        const response = await fetch('/api/plot-dos', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (response.ok && result.status === 'success') {
            displayDosResult(result);
        } else {
            displayResult('dos-result', result, false);
        }
    } catch (error) {
        displayResult('dos-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

function displayDosResult(result) {
    const resultDiv = document.getElementById('dos-result');
    if (!resultDiv) return;

    resultDiv.classList.remove('hidden', 'error');
    const efText = result.ef !== null && result.ef !== undefined
        ? `Fermi energy: ${result.ef.toFixed(4)} eV`
        : 'Fermi energy: not found';

    const metaLines = [];
    if (result.tdos_source_used) metaLines.push(`<p><strong>TDOS source:</strong> ${result.tdos_source_used}</p>`);
    if (result.tdos_peak !== null && result.tdos_peak !== undefined) metaLines.push(`<p><strong>TDOS peak:</strong> ${result.tdos_peak} states/eV</p>`);
    if (result.pdos_peaks) {
        const pdosPeakStr = Object.entries(result.pdos_peaks).map(([label, peak]) => `${label}: ${peak}`).join(', ');
        metaLines.push(`<p><strong>PDOS peaks:</strong> ${pdosPeakStr}</p>`);
    }

    resultDiv.innerHTML = `
        <h4>DOS Plot Ready</h4>
        <p>${efText}</p>
        ${metaLines.join('')}
        <p><strong>Download:</strong> <a href="${result.download_url}" download>PNG file</a></p>
        <div class="result-image">
            <img src="${result.image_url}" alt="DOS plot preview">
        </div>
    `;
}

// ============================================
// DISPLAY / UTILITIES
// ============================================

function displayResult(elementId, result, success) {
    const resultDiv = document.getElementById(elementId);
    if (!resultDiv) return;

    resultDiv.classList.remove('hidden');

    if (success) {
        resultDiv.classList.remove('error');
        resultDiv.innerHTML = `
            <h4>Success</h4>
            <pre>${JSON.stringify(result, null, 2)}</pre>
        `;
    } else {
        resultDiv.classList.add('error');
        resultDiv.innerHTML = `
            <h4>Error</h4>
            <p>${result.error || result.message || 'Unknown error'}</p>
        `;
    }
}

async function checkServerHealth() {
    const statusEl = document.getElementById('server-status');
    const statusText = statusEl?.querySelector('.server-status-text');
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        if (statusEl) {
            statusEl.className = 'server-status connected';
            if (statusText) {
                const parts = ['Server OK'];
                if (data.version) parts[0] = `v${data.version}`;
                if (data.core_available) parts.push('Core');
                statusText.textContent = parts.join(' · ');
            }
        }
        if (!data.core_available) {
            console.warn('Core modules not available');
        }
    } catch (error) {
        console.error('Failed to connect to server:', error);
        if (statusEl) {
            statusEl.className = 'server-status disconnected';
            if (statusText) statusText.textContent = 'Disconnected';
        }
    }
}

function showLoading(show) {
    document.body.style.cursor = show ? 'wait' : '';
}

/**
 * CSS-class-based toast notification
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Legacy alias
function showMessage(message, type) {
    showToast(message, type);
}

function copyToClipboard(text) {
    if (!text) return;
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
    } else {
        fallbackCopy(text);
    }
}

function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/**
 * Toggle advanced settings using max-height CSS transition
 */
function toggleAdvanced(id) {
    const content = document.getElementById(id);
    if (!content) return;
    const header = content.previousElementSibling;

    if (content.style.maxHeight && content.style.maxHeight !== '0px') {
        content.style.maxHeight = '0px';
        content.style.overflow = 'hidden';
        if (header) header.classList.remove('open');
    } else {
        content.style.maxHeight = content.scrollHeight + 'px';
        content.style.overflow = 'visible';
        if (header) header.classList.add('open');
    }
}

// ============================================
// AI STRUCTURE GENERATION
// ============================================

function initializeAITab() {
    checkAIAvailability();
    initializeModelSelector();

    const aiModeButtons = document.querySelectorAll('.ai-mode-selector .mode-btn');
    const fullereneMode = document.getElementById('ai-fullerene-mode');
    const interfaceMode = document.getElementById('ai-interface-mode');

    aiModeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.aiMode;
            aiModeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (mode === 'fullerene') {
                fullereneMode.style.display = 'block';
                interfaceMode.style.display = 'none';
            } else {
                fullereneMode.style.display = 'none';
                interfaceMode.style.display = 'block';
            }
        });
    });

    setupFileUpload('ai-interface-base-file', 'ai-interface-base', (fileType, data) => {
        const infoDiv = document.getElementById('ai-interface-base-info');
        if (!infoDiv) return;
        const fi = data.file_info || {};
        infoDiv.classList.remove('hidden');
        infoDiv.innerHTML = `
            <div class="file-info-content">
                <strong>${fi.filename || 'Uploaded'}</strong>
                ${fi.composition ? ` - ${fi.composition}` : ''}
                ${fi.n_atoms ? ` (${fi.n_atoms} atoms)` : ''}
            </div>
        `;
        state.uploadedFiles['ai-interface-base'] = fi;
        const genBtn = document.getElementById('generate-interface-ai-btn');
        if (genBtn) genBtn.disabled = false;
        const step1 = document.getElementById('pipe-step-1');
        if (step1) step1.classList.add('completed');
    });

    const genIfaceBtn = document.getElementById('generate-interface-ai-btn');
    if (genIfaceBtn) genIfaceBtn.addEventListener('click', handleAIInterfaceGeneration);

    const numAtomsSelect = document.getElementById('ai-num-atoms');
    const customAtomsContainer = document.getElementById('custom-atoms-container');
    if (numAtomsSelect) {
        numAtomsSelect.addEventListener('change', (e) => {
            customAtomsContainer.style.display = e.target.value === 'custom' ? 'block' : 'none';
        });
    }

    const tempSlider = document.getElementById('ai-temperature');
    const tempOutput = document.getElementById('temp-value');
    if (tempSlider && tempOutput) {
        tempSlider.addEventListener('input', (e) => { tempOutput.textContent = e.target.value; });
    }

    const generateBtn = document.getElementById('generate-ai-btn');
    if (generateBtn) generateBtn.addEventListener('click', handleAIGeneration);

    const batchBtn = document.getElementById('batch-generate-btn');
    if (batchBtn) batchBtn.addEventListener('click', handleBatchGeneration);
}

function initializeModelSelector() {
    const modelOptions = document.querySelectorAll('.model-option');
    modelOptions.forEach(opt => {
        opt.addEventListener('click', () => {
            const modelKey = opt.dataset.model;
            const entry = state.availableModels[modelKey];
            if (entry && !entry.available) {
                showToast(`Model "${entry.label}" is not available: ${entry.error || 'unknown error'}`, 'error');
                return;
            }
            modelOptions.forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            const radio = opt.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
            state.selectedModel = modelKey;
            updateModelInfoPanel(modelKey);
            const pipeLabel = document.getElementById('pipe-model-label');
            if (pipeLabel) pipeLabel.textContent = entry ? entry.label.split('+')[0].trim() : modelKey;
        });
    });
}

function updateModelInfoPanel(modelKey) {
    const entry = state.availableModels[modelKey];
    if (!entry) return;

    const elModel = document.getElementById('info-active-model');
    const elArch = document.getElementById('info-architecture');
    const elSampling = document.getElementById('info-sampling');
    const elSpeed = document.getElementById('info-gen-speed');

    if (elModel) elModel.textContent = entry.label;
    if (elArch) elArch.textContent = entry.architecture === 'painn' ? 'PaiNN (angular-aware)' : 'EGNN (distance-based)';
    if (elSampling) elSampling.textContent = entry.diffusion_type === 'flow_matching' ? 'Flow Matching (50-step ODE)' : 'DDPM / DDIM (200 steps)';
    if (elSpeed) elSpeed.textContent = entry.diffusion_type === 'flow_matching' ? '~1-3 seconds/structure' : '~2-5 seconds/structure';
}

function updateModelStatusDots(modelsData) {
    if (!modelsData) return;
    for (const m of modelsData) {
        state.availableModels[m.key] = m;
        const statusEl = document.getElementById(`model-status-${m.key}`);
        const optEl = document.getElementById(`model-opt-${m.key}`);
        if (statusEl) {
            const dot = statusEl.querySelector('.status-dot');
            const text = statusEl.querySelector('.status-text');
            if (m.available) {
                dot.classList.add('available');
                dot.classList.remove('unavailable');
                const params = m.num_parameters ? ` (${(m.num_parameters / 1e6).toFixed(1)}M params)` : '';
                const epoch = m.epoch ? `, epoch ${m.epoch}` : '';
                text.textContent = `Ready${params}${epoch}`;
            } else {
                dot.classList.add('unavailable');
                dot.classList.remove('available');
                text.textContent = m.error || 'Not available';
                if (optEl) optEl.classList.add('disabled');
            }
        }
    }
    const countEl = document.getElementById('info-models-count');
    const nAvail = modelsData.filter(m => m.available).length;
    if (countEl) countEl.textContent = `${nAvail} of ${modelsData.length}`;
}

async function checkAIAvailability() {
    const statusDiv = document.getElementById('ai-status');
    const contentDiv = document.getElementById('ai-content-main');
    const unavailableDiv = document.getElementById('ai-unavailable');
    const reasonEl = document.getElementById('ai-unavailable-reason');
    const checkpointEl = document.getElementById('ai-unavailable-checkpoint');
    const basePathEl = document.getElementById('ai-unavailable-basepath');

    try {
        const response = await fetch('/api/ai/info');
        let data = null;
        try { data = await response.json(); } catch (e) { data = null; }

        if (response.ok && data && data.available) {
            state.aiAvailable = true;
            if (statusDiv) {
                statusDiv.innerHTML = `<div class="status-indicator available"><span class="status-dot"></span><span class="status-text">AI Ready</span></div>`;
            }
            if (contentDiv) contentDiv.style.display = 'block';
            if (unavailableDiv) unavailableDiv.style.display = 'none';

            updateGNNStatus(data);

            if (data.available_models) {
                updateModelStatusDots(data.available_models);
                const activeEntry = state.availableModels[state.selectedModel];
                if (!activeEntry || !activeEntry.available) {
                    const defaultKey = data.default_model || 'egnn';
                    state.selectedModel = defaultKey;
                    document.querySelectorAll('.model-option').forEach(o => {
                        o.classList.toggle('active', o.dataset.model === defaultKey);
                        const r = o.querySelector('input[type="radio"]');
                        if (r) r.checked = o.dataset.model === defaultKey;
                    });
                }
                updateModelInfoPanel(state.selectedModel);
            }
        } else {
            const errorDetail = (data && data.error) ? data.error : `AI module not available (HTTP ${response.status})`;
            const checkpointPath = data && data.checkpoint_path ? data.checkpoint_path : null;
            const basePath = data && data.base_path ? data.base_path : null;
            throw { message: errorDetail, checkpointPath, basePath };
        }
    } catch (error) {
        state.aiAvailable = false;
        const errorMessage = error && error.message ? error.message : 'AI module not available';
        const checkpointPath = error && error.checkpointPath ? error.checkpointPath : null;
        const basePath = error && error.basePath ? error.basePath : null;

        if (statusDiv) {
            statusDiv.innerHTML = `<div class="status-indicator unavailable"><span class="status-dot"></span><span class="status-text">Unavailable</span></div>`;
        }
        if (contentDiv) contentDiv.style.display = 'none';
        if (unavailableDiv) unavailableDiv.style.display = 'block';
        if (reasonEl) reasonEl.textContent = errorMessage;
        if (checkpointEl && checkpointPath) checkpointEl.textContent = checkpointPath;
        if (basePathEl && basePath) basePathEl.textContent = basePath;
    }
}

async function handleAIGeneration() {
    if (!state.aiAvailable) { showToast('AI module is not available', 'error'); return; }

    const numAtomsSelect = document.getElementById('ai-num-atoms');
    const customAtoms = document.getElementById('ai-custom-atoms');
    const numSamples = parseInt(document.getElementById('ai-num-samples').value);
    const outputFormat = document.getElementById('ai-output-format').value;

    let numAtoms = numAtomsSelect.value === 'custom' ? parseInt(customAtoms.value) : parseInt(numAtomsSelect.value);

    if (numAtoms < 20 || numAtoms > 240) { showToast('Number of atoms must be between 20 and 240', 'error'); return; }
    if (numAtoms % 2 !== 0) { showToast('Number of atoms must be even', 'error'); return; }
    if (numSamples < 1 || numSamples > 50) { showToast('Number of samples must be between 1 and 50', 'error'); return; }

    const progressDiv = document.getElementById('ai-progress');
    const progressFill = document.getElementById('ai-progress-fill');
    const progressText = document.getElementById('ai-progress-text');
    const resultsDiv = document.getElementById('ai-results');

    progressDiv.style.display = 'block';
    resultsDiv.classList.add('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = `Generating ${numSamples} structures with ${numAtoms} carbon atoms (${state.selectedModel})...`;

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 2;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
    }, 100);

    try {
        const response = await fetch('/api/ai/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_atoms: numAtoms, num_samples: numSamples, output_format: outputFormat, model: state.selectedModel })
        });

        const data = await response.json();
        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        if (response.ok && data.status === 'success') {
            progressText.textContent = `${data.message}`;
            state.currentStructures = data.structures || [];
            state.generationHistory.push(data);

            setTimeout(() => {
                progressDiv.style.display = 'none';
                displayGenerationResults(data);
                if (state.currentStructures.length > 0) viewStructure(0);
            }, 1000);

            showToast(`Generated ${data.num_generated} structures successfully!`, 'success');
        } else {
            throw new Error(data.error || 'Generation failed');
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressDiv.style.display = 'none';
        showToast(`Generation failed: ${error.message}`, 'error');
    }
}

function displayGenerationResults(data) {
    const resultsDiv = document.getElementById('ai-results');
    resultsDiv.classList.remove('hidden');

    const successRate = (data.success_rate * 100).toFixed(1);
    const avgTime = (data.generation_time / data.num_generated).toFixed(2);

    let structuresHTML = '';
    if (data.structures && data.structures.length > 0) {
        structuresHTML = `
            <div class="results-grid">
                ${data.structures.map((struct, idx) => `
                    <div class="result-card">
                        <div class="result-card-header">Structure ${idx + 1}</div>
                        <div class="result-card-body">
                            <div class="result-metric">
                                <span class="metric-label">Filename:</span>
                                <span class="metric-value">${struct.filename}</span>
                            </div>
                        </div>
                        <div class="result-card-actions">
                            <a href="${struct.download_url}" class="btn btn-secondary btn-sm" download>Download</a>
                            <button class="btn btn-secondary btn-sm" onclick="viewStructure(${idx})">View</button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    resultsDiv.innerHTML = `
        <h3>Generation Results</h3>
        <div class="info-grid" style="margin-bottom: 1.5rem;">
            <div class="info-item"><span class="info-label">Model Used:</span><span class="info-value">${data.model_label || data.model || 'EGNN + DDPM'}</span></div>
            <div class="info-item"><span class="info-label">Structures Generated:</span><span class="info-value">${data.num_generated}</span></div>
            <div class="info-item"><span class="info-label">Success Rate:</span><span class="info-value">${successRate}%</span></div>
            <div class="info-item"><span class="info-label">Total Time:</span><span class="info-value">${data.generation_time}s</span></div>
            <div class="info-item"><span class="info-label">Avg Time/Structure:</span><span class="info-value">${avgTime}s</span></div>
        </div>
        ${structuresHTML}
    `;
}

async function handleBatchGeneration() {
    if (!state.aiAvailable) { showToast('AI module is not available', 'error'); return; }
    showToast('Batch generation feature coming soon!', 'info');
}

function updateGNNStatus(data) {
    const badge = document.getElementById('gnn-status-badge');
    const infoText = document.getElementById('gnn-info-text');

    const hasRefiner = data && data.local_refiner_available;
    const isTrained = data && data.local_refiner_trained;
    const refinerError = data && data.local_refiner_error;

    if (badge) {
        badge.classList.remove('gnn-available', 'gnn-untrained', 'gnn-unavailable');
        if (hasRefiner && isTrained) {
            badge.classList.add('gnn-available');
            badge.textContent = 'GNN OK';
        } else if (hasRefiner && !isTrained) {
            badge.classList.add('gnn-untrained');
            badge.textContent = 'GNN (untrained)';
        } else {
            badge.classList.add('gnn-unavailable');
            badge.textContent = 'GNN --';
        }
    }

    if (infoText) {
        if (hasRefiner && isTrained) {
            infoText.textContent = 'Local GNN refiner is loaded with a trained checkpoint.';
        } else if (hasRefiner && !isTrained) {
            infoText.textContent = 'Local GNN is loaded but no trained checkpoint found. Defaults to near-zero displacements.';
        } else if (refinerError) {
            infoText.textContent = `Local GNN unavailable: ${refinerError}.`;
        } else {
            infoText.textContent = 'Local GNN not configured.';
        }
    }
}

// ============================================
// AI INTERFACE GENERATION
// ============================================

async function handleAIInterfaceGeneration() {
    if (!state.aiAvailable) { showToast('AI module is not available', 'error'); return; }

    const baseInfo = state.uploadedFiles['ai-interface-base'];
    if (!baseInfo || !baseInfo.filename) { showToast('Please upload a perovskite base structure first', 'error'); return; }

    const millerStr = document.getElementById('ai-iface-miller').value.trim();
    const millerParts = millerStr.split(/[\s,]+/).map(Number);
    if (millerParts.length !== 3 || millerParts.some(isNaN)) { showToast('Miller index must be three integers', 'error'); return; }

    const supercellVal = document.getElementById('ai-iface-supercell').value;
    let supercellXY = null;
    if (supercellVal !== 'auto') {
        const parts = supercellVal.split(',').map(Number);
        if (parts.length === 2) supercellXY = parts;
    }

    const xyFracX = parseFloat(document.getElementById('ai-iface-xy-frac-x').value) || 0.5;
    const xyFracY = parseFloat(document.getElementById('ai-iface-xy-frac-y').value) || 0.5;
    const terminationVal = document.getElementById('ai-iface-termination').value;
    const termination = terminationVal === 'auto' ? null : terminationVal;

    const payload = {
        base_filename: baseInfo.filename,
        num_atoms: parseInt(document.getElementById('ai-iface-num-atoms').value),
        num_samples: parseInt(document.getElementById('ai-iface-num-samples').value) || 1,
        miller: millerParts,
        slab_thickness: parseFloat(document.getElementById('ai-iface-slab-thickness').value),
        vacuum: parseFloat(document.getElementById('ai-iface-vacuum').value),
        separation: parseFloat(document.getElementById('ai-iface-separation').value),
        buffer: parseFloat(document.getElementById('ai-iface-buffer').value),
        layer_tol: parseFloat(document.getElementById('ai-iface-layer-tol').value),
        xy_frac: [xyFracX, xyFracY],
        supercell_xy: supercellXY,
        termination: termination,
        ddim: document.getElementById('ai-iface-ddim').checked,
        local_refine: document.getElementById('ai-iface-local-refine').checked,
        refine_scope: document.getElementById('ai-iface-refine-scope').value,
        model: state.selectedModel,
    };

    const progressDiv = document.getElementById('ai-iface-progress');
    const progressFill = document.getElementById('ai-iface-progress-fill');
    const progressText = document.getElementById('ai-iface-progress-text');
    const resultsDiv = document.getElementById('ai-iface-results');

    progressDiv.style.display = 'block';
    resultsDiv.classList.add('hidden');
    progressFill.style.width = '0%';
    const modelLabel = (state.availableModels[state.selectedModel] || {}).label || state.selectedModel;
    progressText.textContent = `Generating fullerene via ${modelLabel}...`;
    setPipelineStep(2);

    let progress = 0;
    const descriptions = [
        { at: 10, text: `Generating fullerene via ${modelLabel}...` },
        { at: 35, text: 'Building perovskite slab...' },
        { at: 55, text: 'Placing fullerene on slab surface...' },
        { at: 75, text: 'Running local GNN refinement...' },
        { at: 90, text: 'Finalizing interface structure...' },
    ];
    const progressInterval = setInterval(() => {
        progress += 1.5;
        if (progress > 95) progress = 95;
        progressFill.style.width = progress + '%';
        for (const d of descriptions) {
            if (progress >= d.at && progress < d.at + 2) {
                progressText.textContent = d.text;
                if (d.at === 10) setPipelineStep(2);
                if (d.at === 35) setPipelineStep(3);
                if (d.at === 75) setPipelineStep(4);
            }
        }
    }, 150);

    try {
        const numSamples = parseInt(document.getElementById('ai-iface-num-samples').value) || 1;
        const useAsync = numSamples > 1;

        if (useAsync) {
            const submitResp = await fetch('/api/ai/generate-interface-async', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const submitData = await submitResp.json();
            if (!submitResp.ok || submitData.status !== 'accepted') throw new Error(submitData.error || 'Failed to submit async task');

            const pollUrl = submitData.poll_url;
            progressText.textContent = `Task submitted (${numSamples} samples). Polling...`;

            const pollResult = await new Promise((resolve, reject) => {
                const pollTimer = setInterval(async () => {
                    try {
                        const pollResp = await fetch(pollUrl);
                        const pollData = await pollResp.json();
                        if (pollData.state === 'STARTED') {
                            progressText.textContent = `Generating ${numSamples} interface structures...`;
                            if (progress < 80) { progress = 50; progressFill.style.width = '50%'; }
                            setPipelineStep(3);
                        } else if (pollData.state === 'SUCCESS') {
                            clearInterval(pollTimer);
                            resolve(pollData.result);
                        } else if (pollData.state === 'FAILURE') {
                            clearInterval(pollTimer);
                            reject(new Error(pollData.error || 'Task failed'));
                        } else if (pollData.state === 'NOT_FOUND') {
                            clearInterval(pollTimer);
                            reject(new Error('Task not found'));
                        }
                    } catch (e) { clearInterval(pollTimer); reject(e); }
                }, 2000);
            });

            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (pollResult.status === 'success') {
                progressText.textContent = 'Interface structures generated!';
                setPipelineStep(5);
                setTimeout(() => { progressDiv.style.display = 'none'; displayInterfaceResults(pollResult); }, 1000);
                showToast(`Interface generated (${pollResult.num_samples} samples, ${pollResult.n_atoms} atoms) in ${pollResult.generation_time}s`, 'success');
            } else {
                throw new Error(pollResult.error || 'Generation failed');
            }
        } else {
            const response = await fetch('/api/ai/generate-interface', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (response.ok && data.status === 'success') {
                progressText.textContent = 'Interface generated successfully!';
                setPipelineStep(5);
                setTimeout(() => { progressDiv.style.display = 'none'; displayInterfaceResults(data); }, 1000);
                showToast(`Interface generated (${data.num_samples} samples, ${data.n_atoms} atoms) in ${data.generation_time}s`, 'success');
            } else {
                throw new Error(data.error || 'Interface generation failed');
            }
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressDiv.style.display = 'none';
        resetPipelineSteps();
        showToast(`Interface generation failed: ${error.message}`, 'error');
    }
}

function setPipelineStep(step) {
    for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`pipe-step-${i}`);
        if (!el) continue;
        el.classList.remove('active', 'completed');
        if (i < step) el.classList.add('completed');
        else if (i === step) el.classList.add('active');
    }
}

function resetPipelineSteps() {
    for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`pipe-step-${i}`);
        if (el) el.classList.remove('active', 'completed');
    }
    if (state.uploadedFiles['ai-interface-base']) {
        const step1 = document.getElementById('pipe-step-1');
        if (step1) step1.classList.add('completed');
    }
}

function displayInterfaceResults(data) {
    const resultsDiv = document.getElementById('ai-iface-results');
    resultsDiv.classList.remove('hidden');

    const samples = data.samples || [{ sample_index: 0, download_url: data.download_url, n_atoms: data.n_atoms, metadata: data.metadata }];
    const numSamples = samples.length;
    const batchHeader = numSamples > 1
        ? `<div class="batch-header"><h3>${numSamples} Interface Structures Generated</h3><span class="meta-value">${data.generation_time}s total</span></div>`
        : '';

    let cardsHtml = '';
    for (const sample of samples) {
        const meta = sample.metadata || {};
        const supercell = meta.supercell || {};
        const refinerUsed = meta.local_refiner_used ? 'Yes' : 'No';
        const refinerTrained = meta.local_refiner_trained ? 'Yes (trained)' : 'No (pass-through)';
        const termTop = (meta.termination_top || []).join(', ') || '--';
        const termBot = (meta.termination_bottom || []).join(', ') || '--';
        const sampleLabel = numSamples > 1 ? ` - Sample #${sample.sample_index + 1}` : '';
        const refinerWarn = data.local_refiner_error
            ? `<div class="gnn-info-panel" style="margin-top:0.5rem; border-left-color: #f59e0b;">Local GNN note: ${data.local_refiner_error}</div>`
            : '';

        cardsHtml += `
            <div class="interface-result-card">
                <h3>Interface Structure${sampleLabel}</h3>
                <div class="interface-meta-grid">
                    <div class="interface-meta-item"><span class="meta-label">Model</span><span class="meta-value">${data.model_label || data.model || '--'}</span></div>
                    <div class="interface-meta-item"><span class="meta-label">Total Atoms</span><span class="meta-value">${sample.n_atoms}</span></div>
                    ${numSamples === 1 ? `<div class="interface-meta-item"><span class="meta-label">Generation Time</span><span class="meta-value">${data.generation_time}s</span></div>` : ''}
                    <div class="interface-meta-item"><span class="meta-label">Supercell</span><span class="meta-value">${supercell.nx || '?'}x${supercell.ny || '?'}</span></div>
                    <div class="interface-meta-item"><span class="meta-label">Termination</span><span class="meta-value">${meta.termination || 'auto'}</span></div>
                    <div class="interface-meta-item"><span class="meta-label">Top Species</span><span class="meta-value">${termTop}</span></div>
                    <div class="interface-meta-item"><span class="meta-label">Bottom Species</span><span class="meta-value">${termBot}</span></div>
                    <div class="interface-meta-item"><span class="meta-label">GNN Refiner</span><span class="meta-value">${refinerUsed}</span></div>
                    <div class="interface-meta-item"><span class="meta-label">GNN Trained</span><span class="meta-value">${refinerTrained}</span></div>
                </div>
                ${refinerWarn}
                <div class="interface-download-bar">
                    <a href="${sample.download_url}" class="btn btn-primary btn-sm" download>Download VASP (POSCAR)</a>
                    <span style="color: var(--text-secondary); font-size: 0.8rem;">${sample.download_url.split('/').pop()}</span>
                </div>
            </div>
        `;
    }

    resultsDiv.innerHTML = batchHeader + cardsHtml;
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ============================================
// THREE.JS 3D MOLECULE VIEWER
// ============================================

let viewer3D = {
    scene: null, camera: null, renderer: null, controls: null,
    animationId: null, autoRotate: false, moleculeGroup: null,
    initialized: false, renderMode: 'ball-stick',
    currentPositions: null, currentEdges: null,
};

function initViewer() {
    if (viewer3D.initialized) return;
    const canvas = document.getElementById('structure-canvas');
    if (!canvas) { showToast('Structure viewer canvas not found', 'error'); return; }
    if (typeof THREE === 'undefined') { showToast('3D viewer library failed to load', 'error'); return; }

    try {
        const container = canvas.parentElement;
        const w = container.clientWidth || 800;
        const h = 500;

        viewer3D.scene = new THREE.Scene();
        viewer3D.scene.background = new THREE.Color(0xf0f2f5);
        viewer3D.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
        viewer3D.camera.position.set(0, 0, 15);
        viewer3D.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        viewer3D.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        viewer3D.renderer.setSize(w, h);

        if (typeof THREE.OrbitControls === 'undefined') { showToast('3D viewer controls failed to load', 'error'); return; }
        viewer3D.controls = new THREE.OrbitControls(viewer3D.camera, canvas);
        viewer3D.controls.enableDamping = true;
        viewer3D.controls.dampingFactor = 0.08;
        viewer3D.controls.rotateSpeed = 0.8;

        viewer3D.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 7);
        viewer3D.scene.add(dirLight);
        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
        backLight.position.set(-5, -5, -5);
        viewer3D.scene.add(backLight);

        viewer3D.moleculeGroup = new THREE.Group();
        viewer3D.scene.add(viewer3D.moleculeGroup);
        viewer3D.initialized = true;

        function animate() {
            viewer3D.animationId = requestAnimationFrame(animate);
            if (viewer3D.autoRotate) viewer3D.moleculeGroup.rotation.y += 0.005;
            viewer3D.controls.update();
            viewer3D.renderer.render(viewer3D.scene, viewer3D.camera);
        }
        animate();
    } catch (err) {
        showToast('Failed to initialize 3D viewer: ' + err.message, 'error');
    }
}

function loadMolecule(positions, edges) {
    if (!viewer3D.initialized) initViewer();
    if (!viewer3D.initialized || !viewer3D.moleculeGroup) return;

    viewer3D.currentPositions = positions;
    viewer3D.currentEdges = edges;
    updateRenderModeButtons();

    try {
        while (viewer3D.moleculeGroup.children.length > 0) {
            const child = viewer3D.moleculeGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            viewer3D.moleculeGroup.remove(child);
        }

        if (!positions || positions.length === 0) return;

        const n = positions.length;
        let cx = 0, cy = 0, cz = 0;
        for (const p of positions) { cx += p[0]; cy += p[1]; cz += p[2]; }
        cx /= n; cy /= n; cz /= n;

        let resolvedEdges = edges && edges.length > 0 ? edges : null;
        if (!resolvedEdges) {
            resolvedEdges = [];
            const maxBondDist2 = 1.85 * 1.85;
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    const dx = positions[i][0] - positions[j][0];
                    const dy = positions[i][1] - positions[j][1];
                    const dz = positions[i][2] - positions[j][2];
                    if (dx*dx + dy*dy + dz*dz <= maxBondDist2) resolvedEdges.push([i, j]);
                }
            }
        }

        const mode = viewer3D.renderMode;

        if (mode === 'wireframe') {
            const lineColor = 0x5b8def;
            const vertexColor = 0x667eea;
            const lineVerts = [];
            for (const [i, j] of resolvedEdges) {
                if (i >= n || j >= n) continue;
                const p1x = positions[i][0]-cx, p1y = positions[i][1]-cy, p1z = positions[i][2]-cz;
                const p2x = positions[j][0]-cx, p2y = positions[j][1]-cy, p2z = positions[j][2]-cz;
                const len2 = (p1x-p2x)**2 + (p1y-p2y)**2 + (p1z-p2z)**2;
                if (len2 < 0.0001 || len2 > 12.25) continue;
                lineVerts.push(p1x, p1y, p1z, p2x, p2y, p2z);
            }
            if (lineVerts.length > 0) {
                const lineGeo = new THREE.BufferGeometry();
                lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(lineVerts, 3));
                const lineMat = new THREE.LineBasicMaterial({ color: lineColor, linewidth: 2 });
                viewer3D.moleculeGroup.add(new THREE.LineSegments(lineGeo, lineMat));
            }
            const dotGeo = new THREE.SphereGeometry(0.25, 16, 12);
            const dotMat = new THREE.MeshPhongMaterial({ color: vertexColor, shininess: 80, specular: 0x8899cc });
            for (const p of positions) {
                const dot = new THREE.Mesh(dotGeo, dotMat);
                dot.position.set(p[0]-cx, p[1]-cy, p[2]-cz);
                viewer3D.moleculeGroup.add(dot);
            }
        } else {
            const atomMat = new THREE.MeshPhongMaterial({ color: 0x333333, shininess: 80, specular: 0x666666 });
            const atomGeo = new THREE.SphereGeometry(0.25, 16, 12);
            const bondMat = new THREE.MeshPhongMaterial({ color: 0x888888, shininess: 40 });

            for (const p of positions) {
                const mesh = new THREE.Mesh(atomGeo, atomMat);
                mesh.position.set(p[0]-cx, p[1]-cy, p[2]-cz);
                viewer3D.moleculeGroup.add(mesh);
            }

            for (const [i, j] of resolvedEdges) {
                if (i >= n || j >= n) continue;
                const p1 = new THREE.Vector3(positions[i][0]-cx, positions[i][1]-cy, positions[i][2]-cz);
                const p2 = new THREE.Vector3(positions[j][0]-cx, positions[j][1]-cy, positions[j][2]-cz);
                const len = p1.distanceTo(p2);
                if (len < 0.01 || len > 3.5) continue;
                const bondGeo = new THREE.CylinderGeometry(0.06, 0.06, len, 6, 1);
                bondGeo.translate(0, len/2, 0);
                bondGeo.rotateX(Math.PI/2);
                const bond = new THREE.Mesh(bondGeo, bondMat);
                bond.position.copy(p1);
                bond.lookAt(p2);
                viewer3D.moleculeGroup.add(bond);
            }
        }

        let maxR = 0;
        for (const p of positions) {
            const r = Math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2);
            if (r > maxR) maxR = r;
        }
        viewer3D.camera.position.set(0, 0, Math.max(maxR * 2.8, 5));
        viewer3D.camera.lookAt(0, 0, 0);
        viewer3D.controls.target.set(0, 0, 0);
        viewer3D.controls.update();
        viewer3D.moleculeGroup.rotation.set(0, 0, 0);
    } catch (err) {
        showToast('Error rendering 3D structure: ' + err.message, 'error');
    }
}

function viewStructure(index) {
    if (index >= state.currentStructures.length) return;
    const struct = state.currentStructures[index];
    const viewerDiv = document.getElementById('ai-viewer');
    const viewerInfo = document.getElementById('viewer-info');

    viewerDiv.classList.remove('hidden');
    viewerInfo.innerHTML = `
        <div class="info-grid">
            <div class="info-item"><span class="info-label">Filename:</span><span class="info-value">${struct.filename}</span></div>
            <div class="info-item"><span class="info-label">Atoms:</span><span class="info-value">${struct.num_atoms || '?'}</span></div>
        </div>
    `;
    viewerDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    requestAnimationFrame(() => {
        if (struct.positions && struct.positions.length > 0) loadMolecule(struct.positions, struct.edges);
        else showToast('No position data available for 3D view', 'info');
    });
}

function rotateStructure() {
    if (!viewer3D.initialized) return;
    viewer3D.autoRotate = !viewer3D.autoRotate;
    showToast(viewer3D.autoRotate ? 'Auto-rotation enabled' : 'Auto-rotation disabled', 'info');
}

function resetView() {
    if (!viewer3D.initialized) return;
    viewer3D.moleculeGroup.rotation.set(0, 0, 0);
    viewer3D.camera.position.set(0, 0, 15);
    viewer3D.camera.lookAt(0, 0, 0);
    viewer3D.controls.target.set(0, 0, 0);
    viewer3D.controls.update();
    viewer3D.autoRotate = false;
}

function downloadCurrent() {
    if (state.currentStructures.length > 0) window.location.href = state.currentStructures[0].download_url;
}

function setRenderMode(mode) {
    if (mode !== 'ball-stick' && mode !== 'wireframe') return;
    viewer3D.renderMode = mode;
    updateRenderModeButtons();
    if (viewer3D.currentPositions && viewer3D.currentPositions.length > 0) {
        loadMolecule(viewer3D.currentPositions, viewer3D.currentEdges);
    }
}

function updateRenderModeButtons() {
    const btnBall = document.getElementById('btn-ball-stick');
    const btnWire = document.getElementById('btn-wireframe');
    if (btnBall) btnBall.classList.toggle('viewer-btn-active', viewer3D.renderMode === 'ball-stick');
    if (btnWire) btnWire.classList.toggle('viewer-btn-active', viewer3D.renderMode === 'wireframe');
}

// ============================================
// DFT PIPELINE TAB
// ============================================

function initializePipelineTab() {
    const sourceButtons = document.querySelectorAll('.pipeline-source-selector .mode-btn');
    const uploadPanel = document.getElementById('pipeline-upload-mode');
    const aiPanel = document.getElementById('pipeline-ai-mode');

    sourceButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const src = btn.dataset.pipelineSource;
            sourceButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.pipelineSource = src;
            if (src === 'upload') {
                uploadPanel.style.display = 'block';
                aiPanel.style.display = 'none';
            } else {
                uploadPanel.style.display = 'none';
                aiPanel.style.display = 'block';
            }
        });
    });

    setupFileUpload('pipeline-perovskite-file', 'pipeline-perovskite', (fileType, data) => {
        const fi = data.file_info || data;
        const infoDiv = document.getElementById('pipeline-perovskite-info');
        if (infoDiv) { infoDiv.classList.remove('hidden'); infoDiv.innerHTML = `<strong>${fi.filename || 'Uploaded'}</strong>${fi.composition ? ` - ${fi.composition}` : ''}${fi.n_atoms ? ` (${fi.n_atoms} atoms)` : ''}`; }
        state.uploadedFiles['pipeline-perovskite'] = fi;
    });

    setupFileUpload('pipeline-fullerene-file', 'pipeline-fullerene', (fileType, data) => {
        const fi = data.file_info || data;
        const infoDiv = document.getElementById('pipeline-fullerene-info');
        if (infoDiv) { infoDiv.classList.remove('hidden'); infoDiv.innerHTML = `<strong>${fi.filename || 'Uploaded'}</strong>${fi.composition ? ` - ${fi.composition}` : ''}${fi.n_atoms ? ` (${fi.n_atoms} atoms)` : ''}`; }
        state.uploadedFiles['pipeline-fullerene'] = fi;
    });

    setupFileUpload('pipeline-ai-perovskite-file', 'pipeline-ai-perovskite', (fileType, data) => {
        const fi = data.file_info || data;
        const infoDiv = document.getElementById('pipeline-ai-perovskite-info');
        if (infoDiv) { infoDiv.classList.remove('hidden'); infoDiv.innerHTML = `<strong>${fi.filename || 'Uploaded'}</strong>${fi.composition ? ` - ${fi.composition}` : ''}${fi.n_atoms ? ` (${fi.n_atoms} atoms)` : ''}`; }
        state.uploadedFiles['pipeline-ai-perovskite'] = fi;
    });

    setupFileUpload('pipeline-dft-structure-file', 'pipeline-dft-structure', (fileType, data) => {
        const fi = data.file_info || data;
        const infoDiv = document.getElementById('pipeline-dft-structure-info');
        if (infoDiv) { infoDiv.classList.remove('hidden'); infoDiv.innerHTML = `<strong>${fi.filename || 'Uploaded'}</strong>${fi.composition ? ` - ${fi.composition}` : ''}${fi.n_atoms ? ` (${fi.n_atoms} atoms)` : ''}`; }
        state.uploadedFiles['pipeline-dft-structure'] = fi;
    });

    const runBtn = document.getElementById('pipeline-run-btn');
    if (runBtn) runBtn.addEventListener('click', handlePipelineRun);

    const dftOnlyBtn = document.getElementById('pipeline-dft-only-btn');
    if (dftOnlyBtn) dftOnlyBtn.addEventListener('click', handleDftPrepOnly);
}

function collectDftSlurm() {
    return {
        dft: {
            functional: document.getElementById('pipeline-functional').value,
            cutoff: parseInt(document.getElementById('pipeline-cutoff').value),
            rel_cutoff: parseInt(document.getElementById('pipeline-rel-cutoff').value),
            max_scf: parseInt(document.getElementById('pipeline-max-scf').value),
            dispersion: document.getElementById('pipeline-dispersion').value,
            geo_opt_max_iter: parseInt(document.getElementById('pipeline-geo-opt-max').value),
            geo_opt_convergence: parseFloat(document.getElementById('pipeline-geo-opt-conv').value),
        },
        slurm: {
            partition: document.getElementById('pipeline-partition').value,
            nodes: parseInt(document.getElementById('pipeline-nodes').value),
            ntasks_per_node: parseInt(document.getElementById('pipeline-ntasks').value),
            time: document.getElementById('pipeline-time').value,
            cp2k_module: document.getElementById('pipeline-cp2k-module').value,
            cp2k_binary: document.getElementById('pipeline-cp2k-binary').value,
        },
    };
}

async function handlePipelineRun() {
    let perovskiteFile, payload;
    const millerStr = document.getElementById('pipeline-miller').value.trim();
    const millerParts = millerStr.split(/[\s,]+/).map(Number);
    if (millerParts.length !== 3 || millerParts.some(isNaN)) { showToast('Miller index must be three integers', 'error'); return; }

    const supercellVal = document.getElementById('pipeline-supercell').value;
    const dftSlurm = collectDftSlurm();

    if (state.pipelineSource === 'upload') {
        perovskiteFile = state.uploadedFiles['pipeline-perovskite'];
        const fullereneFile = state.uploadedFiles['pipeline-fullerene'];
        if (!perovskiteFile || !perovskiteFile.filename) { showToast('Please upload a perovskite structure', 'error'); return; }
        if (!fullereneFile || !fullereneFile.filename) { showToast('Please upload a fullerene structure', 'error'); return; }
        payload = { source: 'upload', perovskite_filename: perovskiteFile.filename, fullerene_filename: fullereneFile.filename, fullerene_source: 'upload' };
    } else {
        perovskiteFile = state.uploadedFiles['pipeline-ai-perovskite'];
        if (!perovskiteFile || !perovskiteFile.filename) { showToast('Please upload a perovskite structure', 'error'); return; }
        payload = {
            source: 'ai', perovskite_filename: perovskiteFile.filename, fullerene_source: 'ai',
            ai_num_carbon: parseInt(document.getElementById('pipeline-ai-num-carbon').value),
            ai_num_samples: parseInt(document.getElementById('pipeline-ai-num-samples').value),
            ai_model: document.getElementById('pipeline-ai-model').value,
        };
    }

    payload.miller = millerParts;
    payload.slab_thickness = parseFloat(document.getElementById('pipeline-slab-thickness').value);
    payload.vacuum = parseFloat(document.getElementById('pipeline-vacuum').value);
    payload.separation = parseFloat(document.getElementById('pipeline-separation').value);
    payload.supercell = supercellVal;
    payload.termination = document.getElementById('pipeline-termination').value;
    payload.fix_bottom_layers = parseInt(document.getElementById('pipeline-fix-layers').value);
    payload.dft = dftSlurm.dft;
    payload.slurm = dftSlurm.slurm;

    const progressDiv = document.getElementById('pipeline-progress');
    const progressFill = document.getElementById('pipeline-progress-fill');
    const progressText = document.getElementById('pipeline-progress-text');
    const resultDiv = document.getElementById('pipeline-result');

    progressDiv.style.display = 'block';
    resultDiv.classList.add('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = 'Building interfaces...';

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 2;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
        if (progress > 30 && progress < 33) progressText.textContent = 'Generating DFT input files...';
        if (progress > 60 && progress < 63) progressText.textContent = 'Creating ZIP archive...';
    }, 150);

    try {
        const response = await fetch('/api/pipeline/build-and-prep', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        if (response.ok && data.status === 'success') {
            progressText.textContent = 'Pipeline complete!';
            setTimeout(() => { progressDiv.style.display = 'none'; displayPipelineResult('pipeline-result', data); }, 800);
            showToast(`Built ${data.n_interfaces} interface(s), prepared ${data.n_jobs} DFT job(s)`, 'success');
        } else {
            throw new Error(data.error || 'Pipeline failed');
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressDiv.style.display = 'none';
        showToast(`Pipeline failed: ${error.message}`, 'error');
        displayResult('pipeline-result', { error: error.message }, false);
    }
}

async function handleDftPrepOnly() {
    const structFile = state.uploadedFiles['pipeline-dft-structure'];
    if (!structFile || !structFile.filename) { showToast('Please upload a structure file', 'error'); return; }

    const dftSlurm = collectDftSlurm();
    const payload = { structure_filename: structFile.filename, dft: dftSlurm.dft, slurm: dftSlurm.slurm };

    try {
        showLoading(true);
        const response = await fetch('/api/pipeline/dft-prep', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (response.ok && data.status === 'success') {
            displayPipelineResult('pipeline-dft-result', data);
            showToast(`Prepared ${data.n_jobs} DFT job(s)`, 'success');
        } else {
            displayResult('pipeline-dft-result', data, false);
        }
    } catch (error) {
        displayResult('pipeline-dft-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

function displayPipelineResult(elementId, data) {
    const resultDiv = document.getElementById(elementId);
    if (!resultDiv) return;

    resultDiv.classList.remove('hidden', 'error');

    let jobsHTML = '';
    if (data.jobs_summary && data.jobs_summary.length > 0) {
        jobsHTML = `<div class="results-grid">
            ${data.jobs_summary.map(job => `
                <div class="result-card">
                    <div class="result-card-header">${escapeHtml(job.job_id)}</div>
                    <div class="result-card-body">
                        <div class="result-metric"><span class="metric-label">Atoms:</span><span class="metric-value">${job.n_atoms || '?'}</span></div>
                        <div class="result-metric"><span class="metric-label">Elements:</span><span class="metric-value">${(job.elements || []).join(', ')}</span></div>
                        <div class="result-metric"><span class="metric-label">Status:</span><span class="metric-value">${job.status || '?'}</span></div>
                    </div>
                </div>
            `).join('')}
        </div>`;
    }

    resultDiv.innerHTML = `
        <h4>DFT Pipeline Complete</h4>
        <div class="info-grid" style="margin-bottom: 1.25rem;">
            <div class="info-item"><span class="info-label">Interfaces Built:</span><span class="info-value">${data.n_interfaces}</span></div>
            <div class="info-item"><span class="info-label">DFT Jobs Prepared:</span><span class="info-value">${data.n_jobs}</span></div>
        </div>
        ${jobsHTML}
        <div class="action-bar" style="margin-top: 1.25rem;">
            <a href="${data.download_url}" class="btn btn-primary btn-lg" download>Download ZIP (CP2K + SLURM)</a>
        </div>
    `;
}
