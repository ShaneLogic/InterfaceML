/**
 * InterfaceML Web Application
 * Client-side JavaScript for heterojunction modeling interface
 */

// State management
const state = {
    uploadedFiles: {},
    currentTab: 'adsorbate',
    layerMode: 'fix',  // 'fix', 'split', or 'pdos'
    pdosFiles: [],     // Array of PDOS file data
    pdosCounter: 0,    // Counter for unique PDOS IDs
    aiAvailable: false,
    currentStructures: [],
    generationHistory: []
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeFileUploads();
    initializeButtons();
    initializeModeSwitcher();
    initializePdosUploads();
    checkServerHealth();
    initializeAITab();
});

/**
 * Tab switching functionality
 */
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;
            
            // Update button states
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update content visibility
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            state.currentTab = tabId;
        });
    });
}

/**
 * File upload handling
 */
function initializeFileUploads() {
    // Base structure upload
    setupFileUpload('base-file', 'base', displayFileInfo);
    
    // Adsorbate upload(s)
    setupFileUpload('adsorbate-file', 'adsorbate', displayFileInfo);
    
    // Substrate upload
    setupFileUpload('substrate-file', 'substrate', displayFileInfo);
    
    // Film upload
    setupFileUpload('film-file', 'film', displayFileInfo);
    
    // Layers file upload
    setupFileUpload('layers-file', 'layers', displayLayersFileInfo);

    // DOS uploads - only TDOS, PDOS is handled dynamically
    setupFileUpload('tdos-file', 'tdos', displayDosFileInfo);
}

/**
 * Setup file upload for an input element
 */
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
    
    // Drag and drop support
    const uploadArea = input.closest('.file-upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--primary-color)';
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '';
        });
        
        uploadArea.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '';
            
            const files = e.dataTransfer.files;
            for (let file of files) {
                await uploadFile(file, fileType, callback);
            }
        });
    }
}

/**
 * Upload file to server
 */
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
        
        const data = await response.json();
        
        if (response.ok) {
            state.uploadedFiles[fileType] = data;
            callback(fileType, data);
            showMessage(`✓ ${file.name} uploaded successfully`, 'success');
        } else {
            showMessage(`✗ Upload failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage(`✗ Upload error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Display uploaded file information
 */
function displayFileInfo(fileType, data) {
    const infoDiv = document.getElementById(`${fileType}-info`);
    if (!infoDiv) return;
    
    infoDiv.classList.remove('hidden');
    infoDiv.innerHTML = `
        <strong>${data.filename}</strong><br>
        ${data.composition ? `Composition: ${data.composition}<br>` : ''}
        ${data.n_atoms ? `Atoms: ${data.n_atoms}<br>` : ''}
        ${data.lattice_abc ? `Cell: ${data.lattice_abc.join(' × ')} Å<br>` : ''}
        ${data.parse_error ? `<span style="color: #ef4444;">Parse error: ${data.parse_error}</span>` : ''}
    `;
}

/**
 * Display layers file info
 */
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

/**
 * Display DOS file info
 */
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

/**
 * Mode switcher for layer management
 */
function initializeModeSwitcher() {
    const modeButtons = document.querySelectorAll('.mode-btn');
    
    modeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const mode = button.dataset.mode;
            
            // Update button states
            modeButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show/hide appropriate parameter sections
            const fixParams = document.getElementById('fix-mode-params');
            const splitParams = document.getElementById('split-mode-params');
            const pdosParams = document.getElementById('pdos-mode-params');
            
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
            
            // Clear previous results
            const resultDiv = document.getElementById('layers-result');
            resultDiv.classList.add('hidden');
        });
    });
}

/**
 * Initialize action buttons
 */
function initializeButtons() {
    // Build adsorbate button
    const buildAdsorbateBtn = document.getElementById('build-adsorbate-btn');
    if (buildAdsorbateBtn) {
        buildAdsorbateBtn.addEventListener('click', buildAdsorbate);
    }
    
    // Build interface button
    const buildInterfaceBtn = document.getElementById('build-interface-btn');
    if (buildInterfaceBtn) {
        buildInterfaceBtn.addEventListener('click', buildInterface);
    }
    
    // Fix layers button
    const fixLayersBtn = document.getElementById('fix-layers-btn');
    if (fixLayersBtn) {
        fixLayersBtn.addEventListener('click', fixLayers);
    }
    
    // Split layers button
    const splitLayersBtn = document.getElementById('split-layers-btn');
    if (splitLayersBtn) {
        splitLayersBtn.addEventListener('click', splitLayers);
    }

    // Generate PDOS layers button
    const pdosLayersBtn = document.getElementById('pdos-layers-btn');
    if (pdosLayersBtn) {
        pdosLayersBtn.addEventListener('click', generatePdosLayers);
    }
    
    // Compute density button
    const computeDensityBtn = document.getElementById('compute-density-btn');
    if (computeDensityBtn) {
        computeDensityBtn.addEventListener('click', computeDensity);
    }

    const plotDosBtn = document.getElementById('plot-dos-btn');
    if (plotDosBtn) {
        plotDosBtn.addEventListener('click', plotDos);
    }
}

/**
 * Build adsorbate structure
 */
async function buildAdsorbate() {
    if (!state.uploadedFiles.base || !state.uploadedFiles.adsorbate) {
        showMessage('Please upload both base and adsorbate files', 'error');
        return;
    }
    
    const data = {
        base_file: state.uploadedFiles.base.filepath,
        adsorbate_files: [state.uploadedFiles.adsorbate.filepath],
        termination: document.getElementById('termination').value,
        distance: parseFloat(document.getElementById('distance').value),
        miller: document.getElementById('miller').value.split(' ').map(Number),
        supercell: document.getElementById('supercell').value
    };
    
    try {
        showLoading(true);
        const response = await fetch('/api/build-adsorbate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        displayResult('adsorbate-result', result, response.ok);
    } catch (error) {
        displayResult('adsorbate-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

/**
 * Build interface structure
 */
async function buildInterface() {
    if (!state.uploadedFiles.substrate || !state.uploadedFiles.film) {
        showMessage('Please upload both substrate and film files', 'error');
        return;
    }
    
    const data = {
        base_file: state.uploadedFiles.substrate.filepath,
        film_file: state.uploadedFiles.film.filepath,
        miller_base: document.getElementById('miller-sub').value.split(' ').map(Number),
        miller_film: document.getElementById('miller-film').value.split(' ').map(Number),
        max_area: parseFloat(document.getElementById('max-area').value),
        strain_mode: document.getElementById('strain-mode').value
    };
    
    try {
        showLoading(true);
        const response = await fetch('/api/build-interface', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        displayResult('interface-result', result, response.ok);
    } catch (error) {
        displayResult('interface-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

/**
 * Fix layers with selective dynamics
 */
async function fixLayers() {
    if (!state.uploadedFiles.layers) {
        showMessage('Please upload a structure file', 'error');
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

/**
 * Display layers fixing result
 */
function displayLayersResult(result) {
    const resultDiv = document.getElementById('layers-result');
    resultDiv.classList.remove('hidden', 'error');
    resultDiv.innerHTML = `
        <h4>✓ Selective Dynamics Added</h4>
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

/**
 * Split layers
 */
async function splitLayers() {
    if (!state.uploadedFiles.layers) {
        showMessage('Please upload a structure file', 'error');
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

/**
 * Generate PDOS layer indices and CP2K input blocks
 */
async function generatePdosLayers() {
    if (!state.uploadedFiles.layers) {
        showMessage('Please upload a structure file', 'error');
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

/**
 * Display layer splitting result
 */
function displaySplitResult(result) {
    const resultDiv = document.getElementById('layers-result');
    resultDiv.classList.remove('hidden', 'error');
    
    let layersHTML = '';
    result.layers.forEach((layer, index) => {
        // Determine layer icon based on type
        let layerIcon = '📦';
        if (layer.layer_type && layer.layer_type.includes('Perovskite')) {
            layerIcon = '🔷';
        } else if (layer.layer_type && layer.layer_type.includes('C60')) {
            layerIcon = '⚫';
        } else if (layer.layer_type && layer.layer_type.includes('C70')) {
            layerIcon = '🟤';
        } else if (layer.layer_type && layer.layer_type.includes('Oxide')) {
            layerIcon = '🔶';
        }
        
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
                    📥 Download ${layer.filename}
                </a>
            </div>
        `;
    });
    
    resultDiv.innerHTML = `
        <h4>✓ Structure Split Successfully</h4>
        ${result.run_id ? `<p><strong>Run ID:</strong> ${result.run_id}</p>` : ''}
        ${result.input_hash ? `<p><strong>Input Hash:</strong> ${result.input_hash}</p>` : ''}
        <p><strong>Total interfaces detected:</strong> ${result.n_interfaces}</p>
        <p><strong>Layers created:</strong> ${result.n_layers}</p>
        <div class="layers-grid">
            ${layersHTML}
        </div>
        <p class="result-note">
            <em>Note: Each layer preserves the original lattice parameters (a, b, c, α, β, γ).</em>
        </p>
    `;
}

/**
 * Display PDOS layer indices and CP2K block
 */
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
                    <span class="layer-icon">🧩</span>
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
                    <button class="btn btn-secondary" data-copy-layer="${layer.layer_number}">
                        Copy Layer LIST
                    </button>
                </div>
                <div class="layer-code">${escapeHtml(layer.ranges_1)}</div>
            </div>
        `;
    });

    resultDiv.innerHTML = `
        <h4>✓ PDOS Layer Indices Ready</h4>
        <p><strong>Interfaces:</strong> ${result.n_interfaces}</p>
        <p><strong>Layers detected:</strong> ${result.n_layers}</p>
        <p><strong>Minimum gap:</strong> ${result.min_gap} Å</p>
        <div class="pdos-actions">
            <button class="btn btn-secondary" data-copy="pdos">Copy CP2K PDOS Block</button>
            <button class="btn btn-secondary" data-copy="ranges">Copy Layer Ranges</button>
        </div>
        <div class="code-block" id="pdos-block">${escapeHtml(pdosBlock)}</div>
        <div class="layers-grid">
            ${layersHTML}
        </div>
    `;

    const pdosButton = resultDiv.querySelector('[data-copy="pdos"]');
    if (pdosButton) {
        pdosButton.addEventListener('click', () => {
            copyToClipboard(pdosBlock);
            showMessage('CP2K PDOS block copied', 'success');
        });
    }

    const rangesButton = resultDiv.querySelector('[data-copy="ranges"]');
    if (rangesButton) {
        rangesButton.addEventListener('click', () => {
            copyToClipboard(rangesText);
            showMessage('Layer ranges copied', 'success');
        });
    }

    const layerButtons = resultDiv.querySelectorAll('[data-copy-layer]');
    layerButtons.forEach(button => {
        button.addEventListener('click', () => {
            const layerNumber = button.getAttribute('data-copy-layer');
            const layer = result.layers.find(l => String(l.layer_number) === String(layerNumber));
            if (layer) {
                copyToClipboard(layer.ranges_1 || '');
                showMessage(`Layer ${layer.layer_number} LIST copied`, 'success');
            }
        });
    });
}

/**
 * Compute density difference
 */
async function computeDensity() {
    showMessage('Density analysis functionality coming soon', 'info');
}

/**
 * Initialize dynamic PDOS file uploads
 */
function initializePdosUploads() {
    const addPdosBtn = document.getElementById('add-pdos-btn');
    if (addPdosBtn) {
        addPdosBtn.addEventListener('click', addPdosUpload);
    }
    
    // Add initial two PDOS uploads by default
    addPdosUpload();
    addPdosUpload();
}

/**
 * Add a new PDOS upload field
 */
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
            <button type="button" class="btn btn-danger btn-sm btn-remove-pdos" data-pdos-id="${pdosId}" title="Remove this PDOS">
                ✕
            </button>
        </div>
        <div id="pdos${pdosId}-info" class="file-info hidden"></div>
    `;
    
    container.appendChild(pdosItem);
    
    // Setup file upload for this new input
    const fileInput = document.getElementById(`pdos${pdosId}-file`);
    const uploadArea = document.getElementById(`pdos${pdosId}-upload`);
    
    fileInput.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (files.length === 0) return;
        await uploadPdosFile(files[0], pdosId);
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
    });
    
    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            await uploadPdosFile(files[0], pdosId);
        }
    });
    
    // Remove button
    const removeBtn = pdosItem.querySelector('.btn-remove-pdos');
    removeBtn.addEventListener('click', () => removePdosUpload(pdosId));
}

/**
 * Upload a PDOS file
 */
async function uploadPdosFile(file, pdosId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', `pdos${pdosId}`);
    
    try {
        showLoading(true);
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Store in pdosFiles array
            const existing = state.pdosFiles.findIndex(p => p.id === pdosId);
            if (existing >= 0) {
                state.pdosFiles[existing] = { id: pdosId, ...data };
            } else {
                state.pdosFiles.push({ id: pdosId, ...data });
            }
            
            // Also store in uploadedFiles for compatibility
            state.uploadedFiles[`pdos${pdosId}`] = data;
            
            // Update display
            displayDosFileInfo(`pdos${pdosId}`, data);
            showMessage(`✓ ${file.name} uploaded successfully`, 'success');
        } else {
            showMessage(`✗ Upload failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage(`✗ Upload error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Remove a PDOS upload field
 */
function removePdosUpload(pdosId) {
    const pdosItem = document.getElementById(`pdos-item-${pdosId}`);
    if (pdosItem) {
        pdosItem.remove();
    }
    
    // Remove from state
    state.pdosFiles = state.pdosFiles.filter(p => p.id !== pdosId);
    delete state.uploadedFiles[`pdos${pdosId}`];
}

/**
 * Plot TDOS + PDOS overlay
 */
async function plotDos() {
    // Check TDOS
    if (!state.uploadedFiles.tdos) {
        showMessage('Please upload a TDOS file', 'error');
        return;
    }
    
    // Check PDOS files
    if (state.pdosFiles.length === 0) {
        showMessage('Please upload at least one PDOS file', 'error');
        return;
    }
    
    // Collect all PDOS file paths
    const pdosFilePaths = state.pdosFiles
        .sort((a, b) => a.id - b.id)
        .map(p => p.filepath);

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

/**
 * Display result in a panel
 */
function displayResult(elementId, result, success) {
    const resultDiv = document.getElementById(elementId);
    if (!resultDiv) return;
    
    resultDiv.classList.remove('hidden');
    
    if (success) {
        resultDiv.classList.remove('error');
        resultDiv.innerHTML = `
            <h4>✓ Success</h4>
            <pre>${JSON.stringify(result, null, 2)}</pre>
        `;
    } else {
        resultDiv.classList.add('error');
        resultDiv.innerHTML = `
            <h4>✗ Error</h4>
            <p>${result.error || result.message || 'Unknown error'}</p>
        `;
    }
}

/**
 * Display DOS plot result
 */
function displayDosResult(result) {
    const resultDiv = document.getElementById('dos-result');
    if (!resultDiv) return;

    resultDiv.classList.remove('hidden', 'error');
    const efText = result.ef !== null && result.ef !== undefined
        ? `Fermi energy: ${result.ef.toFixed(4)} eV`
        : 'Fermi energy: not found in headers';

    const metaLines = [];
    if (result.tdos_source_used) {
        metaLines.push(`<p><strong>TDOS source:</strong> ${result.tdos_source_used}</p>`);
    }
    if (result.tdos_peak !== null && result.tdos_peak !== undefined) {
        metaLines.push(`<p><strong>TDOS peak:</strong> ${result.tdos_peak} states/eV</p>`);
    }
    if (result.pdos_peaks) {
        const pdosPeakStr = Object.entries(result.pdos_peaks)
            .map(([label, peak]) => `${label}: ${peak}`)
            .join(', ');
        metaLines.push(`<p><strong>PDOS peaks:</strong> ${pdosPeakStr}</p>`);
    }
    if (result.tdos_total_atoms_used !== null && result.tdos_total_atoms_used !== undefined) {
        metaLines.push(`<p><strong>TDOS total atoms:</strong> ${result.tdos_total_atoms_used}</p>`);
    }

    resultDiv.innerHTML = `
        <h4>✓ DOS Plot Ready</h4>
        <p>${efText}</p>
        ${metaLines.join('')}
        <p><strong>Download:</strong> <a href="${result.download_url}" download>PNG file</a></p>
        <div class="result-image">
            <img src="${result.image_url}" alt="DOS plot preview">
        </div>
    `;
}

/**
 * Check server health
 */
async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (!data.core_available) {
            console.warn('Core modules not available - some features may be limited');
        }
    } catch (error) {
        console.error('Failed to connect to server:', error);
    }
}

/**
 * Show loading state
 */
function showLoading(show) {
    const body = document.body;
    if (show) {
        body.style.cursor = 'wait';
    } else {
        body.style.cursor = '';
    }
}

/**
 * Show toast message
 */
function showMessage(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6'};
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    if (!text) return;

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).catch(() => {
            fallbackCopy(text);
        });
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

/**
 * Escape HTML for safe rendering in code blocks
 */
function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);


// ============================================
// AI STRUCTURE GENERATION FUNCTIONALITY
// ============================================

/**
 * Initialize AI tab functionality
 */
function initializeAITab() {
    // Check AI module availability
    checkAIAvailability();
    
    // Custom atom count selector
    const numAtomsSelect = document.getElementById('ai-num-atoms');
    const customAtomsContainer = document.getElementById('custom-atoms-container');
    
    if (numAtomsSelect) {
        numAtomsSelect.addEventListener('change', (e) => {
            if (e.target.value === 'custom') {
                customAtomsContainer.style.display = 'block';
            } else {
                customAtomsContainer.style.display = 'none';
            }
        });
    }
    
    // Temperature slider
    const tempSlider = document.getElementById('ai-temperature');
    const tempOutput = document.getElementById('temp-value');
    
    if (tempSlider && tempOutput) {
        tempSlider.addEventListener('input', (e) => {
            tempOutput.textContent = e.target.value;
        });
    }
    
    // Generate button
    const generateBtn = document.getElementById('generate-ai-btn');
    if (generateBtn) {
        generateBtn.addEventListener('click', handleAIGeneration);
    }
    
    // Batch generate button
    const batchBtn = document.getElementById('batch-generate-btn');
    if (batchBtn) {
        batchBtn.addEventListener('click', handleBatchGeneration);
    }
}

/**
 * Check if AI module is available
 */
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
        try {
            data = await response.json();
        } catch (jsonError) {
            data = null;
        }
        
        if (response.ok && data && data.available) {
            state.aiAvailable = true;
            
            // Update status indicator
            if (statusDiv) {
                statusDiv.innerHTML = `
                    <div class="status-indicator available">
                        <span class="status-dot"></span>
                        <span class="status-text">AI Ready</span>
                    </div>
                `;
            }
            
            // Show main content
            if (contentDiv) contentDiv.style.display = 'block';
            if (unavailableDiv) unavailableDiv.style.display = 'none';
            
            console.log('AI Module Info:', data);
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

        // Update status indicator
        if (statusDiv) {
            statusDiv.innerHTML = `
                <div class="status-indicator unavailable">
                    <span class="status-dot"></span>
                    <span class="status-text">Unavailable</span>
                </div>
            `;
        }
        
        // Show unavailable message
        if (contentDiv) contentDiv.style.display = 'none';
        if (unavailableDiv) unavailableDiv.style.display = 'block';

        if (reasonEl) reasonEl.textContent = errorMessage;
        if (checkpointEl && checkpointPath) checkpointEl.textContent = checkpointPath;
        if (basePathEl && basePath) basePathEl.textContent = basePath;
        
        console.error('AI module check failed:', error);
    }
}

/**
 * Handle AI structure generation
 */
async function handleAIGeneration() {
    if (!state.aiAvailable) {
        showMessage('AI module is not available', 'error');
        return;
    }
    
    // Get parameters
    const numAtomsSelect = document.getElementById('ai-num-atoms');
    const customAtoms = document.getElementById('ai-custom-atoms');
    const numSamples = parseInt(document.getElementById('ai-num-samples').value);
    const outputFormat = document.getElementById('ai-output-format').value;
    
    let numAtoms;
    if (numAtomsSelect.value === 'custom') {
        numAtoms = parseInt(customAtoms.value);
    } else {
        numAtoms = parseInt(numAtomsSelect.value);
    }
    
    // Validate
    if (numAtoms < 20 || numAtoms > 240) {
        showMessage('Number of atoms must be between 20 and 240', 'error');
        return;
    }
    if (numAtoms % 2 !== 0) {
        showMessage('Number of atoms must be even', 'error');
        return;
    }
    if (numSamples < 1 || numSamples > 50) {
        showMessage('Number of samples must be between 1 and 50', 'error');
        return;
    }
    
    // Show progress
    const progressDiv = document.getElementById('ai-progress');
    const progressFill = document.getElementById('ai-progress-fill');
    const progressText = document.getElementById('ai-progress-text');
    const resultsDiv = document.getElementById('ai-results');
    
    progressDiv.style.display = 'block';
    resultsDiv.classList.add('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = `Generating ${numSamples} structures with ${numAtoms} carbon atoms...`;
    
    // Animate progress
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
            body: JSON.stringify({
                num_atoms: numAtoms,
                num_samples: numSamples,
                output_format: outputFormat
            })
        });
        
        const data = await response.json();
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        
        if (response.ok && data.status === 'success') {
            progressText.textContent = `✓ ${data.message}`;
            state.currentStructures = data.structures || [];
            state.generationHistory.push(data);
            
            // Display results
            setTimeout(() => {
                progressDiv.style.display = 'none';
                displayGenerationResults(data);
            }, 1000);
            
            showMessage(`Generated ${data.num_generated} structures successfully!`, 'success');
        } else {
            throw new Error(data.error || 'Generation failed');
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressDiv.style.display = 'none';
        showMessage(`Generation failed: ${error.message}`, 'error');
        console.error('Generation error:', error);
    }
}

/**
 * Display generation results
 */
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
                        <div class="result-card-header">
                            Structure ${idx + 1}
                        </div>
                        <div class="result-card-body">
                            <div class="result-metric">
                                <span class="metric-label">Filename:</span>
                                <span class="metric-value">${struct.filename}</span>
                            </div>
                        </div>
                        <div class="result-card-actions">
                            <a href="${struct.download_url}" class="btn btn-secondary btn-sm" download>
                                💾 Download
                            </a>
                            <button class="btn btn-secondary btn-sm" onclick="viewStructure(${idx})">
                                👁️ View
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    resultsDiv.innerHTML = `
        <h3>Generation Results</h3>
        <div class="info-grid" style="margin-bottom: 2rem;">
            <div class="info-item">
                <span class="info-label">Structures Generated:</span>
                <span class="info-value">${data.num_generated}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Success Rate:</span>
                <span class="info-value">${successRate}%</span>
            </div>
            <div class="info-item">
                <span class="info-label">Total Time:</span>
                <span class="info-value">${data.generation_time}s</span>
            </div>
            <div class="info-item">
                <span class="info-label">Avg Time/Structure:</span>
                <span class="info-value">${avgTime}s</span>
            </div>
        </div>
        ${structuresHTML}
        ${data.structures.length < data.num_generated ? `
            <p class="result-note">
                Showing first ${data.structures.length} of ${data.num_generated} structures. 
                All structures have been saved to the output directory.
            </p>
        ` : ''}
    `;
}

/**
 * Handle batch generation
 */
async function handleBatchGeneration() {
    if (!state.aiAvailable) {
        showMessage('AI module is not available', 'error');
        return;
    }
    
    showMessage('Batch generation feature coming soon!', 'info');
}

/**
 * View structure in 3D viewer
 */
function viewStructure(index) {
    if (index >= state.currentStructures.length) return;
    
    const struct = state.currentStructures[index];
    const viewerDiv = document.getElementById('ai-viewer');
    const viewerInfo = document.getElementById('viewer-info');
    
    viewerDiv.classList.remove('hidden');
    viewerInfo.innerHTML = `
        <div class="info-grid">
            <div class="info-item">
                <span class="info-label">Filename:</span>
                <span class="info-value">${struct.filename}</span>
            </div>
        </div>
    `;
    
    // Scroll to viewer
    viewerDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    showMessage('3D visualization requires additional libraries (Three.js)', 'info');
}

/**
 * Toggle advanced settings
 */
function toggleAdvanced(id) {
    const content = document.getElementById(id);
    const header = content.previousElementSibling;
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        header.classList.add('open');
    } else {
        content.style.display = 'none';
        header.classList.remove('open');
    }
}

/**
 * Structure viewer controls
 */
function rotateStructure() {
    showMessage('Auto-rotation enabled', 'info');
}

function resetView() {
    showMessage('View reset to default', 'info');
}

function downloadCurrent() {
    if (state.currentStructures.length > 0) {
        window.location.href = state.currentStructures[0].download_url;
    }
}
