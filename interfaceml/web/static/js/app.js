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
    selectedModel: 'egnn',   // 'egnn' or 'painn_fm'
    availableModels: {},     // keyed by model key
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

    // ===== Model Selector =====
    initializeModelSelector();

    // ===== AI Mode Switcher =====
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

    // ===== Interface base-structure upload =====
    setupFileUpload('ai-interface-base-file', 'ai-interface-base', (fileType, data) => {
        const infoDiv = document.getElementById('ai-interface-base-info');
        if (!infoDiv) return;
        const fi = data.file_info || {};
        infoDiv.classList.remove('hidden');
        infoDiv.innerHTML = `
            <div class="file-info-content">
                <strong>${fi.filename || 'Uploaded'}</strong>
                ${fi.composition ? ` — ${fi.composition}` : ''}
                ${fi.n_atoms ? ` (${fi.n_atoms} atoms)` : ''}
            </div>
        `;
        state.uploadedFiles['ai-interface-base'] = fi;
        // Enable generate button
        const genBtn = document.getElementById('generate-interface-ai-btn');
        if (genBtn) genBtn.disabled = false;
        // Highlight pipeline step 1
        const step1 = document.getElementById('pipe-step-1');
        if (step1) { step1.classList.add('completed'); }
    });

    // ===== Generate interface button =====
    const genIfaceBtn = document.getElementById('generate-interface-ai-btn');
    if (genIfaceBtn) {
        genIfaceBtn.addEventListener('click', handleAIInterfaceGeneration);
    }

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
 * Initialize model selector toggle cards
 */
function initializeModelSelector() {
    const modelOptions = document.querySelectorAll('.model-option');
    modelOptions.forEach(opt => {
        opt.addEventListener('click', () => {
            const modelKey = opt.dataset.model;
            // Don't switch if model is unavailable
            const entry = state.availableModels[modelKey];
            if (entry && !entry.available) {
                showMessage(`Model "${entry.label}" is not available: ${entry.error || 'unknown error'}`, 'error');
                return;
            }
            // Toggle active
            modelOptions.forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            const radio = opt.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
            state.selectedModel = modelKey;
            updateModelInfoPanel(modelKey);
            // Update pipeline label
            const pipeLabel = document.getElementById('pipe-model-label');
            if (pipeLabel) {
                pipeLabel.textContent = entry ? entry.label.split('+')[0].trim() : modelKey;
            }
        });
    });
}

/**
 * Update the About the AI Models info panel when model changes
 */
function updateModelInfoPanel(modelKey) {
    const entry = state.availableModels[modelKey];
    if (!entry) return;

    const elModel = document.getElementById('info-active-model');
    const elArch = document.getElementById('info-architecture');
    const elSampling = document.getElementById('info-sampling');
    const elSpeed = document.getElementById('info-gen-speed');

    if (elModel) elModel.textContent = entry.label;
    if (elArch) elArch.textContent = entry.architecture === 'painn' ? 'PaiNN (angular-aware)' : 'EGNN (distance-based)';
    if (elSampling) elSampling.textContent = entry.diffusion_type === 'flow_matching'
        ? 'Flow Matching (50-step ODE)'
        : 'DDPM / DDIM (200 steps)';
    if (elSpeed) elSpeed.textContent = entry.diffusion_type === 'flow_matching'
        ? '~1-3 seconds/structure'
        : '~2-5 seconds/structure';
}

/**
 * Update model status dots after we know availability
 */
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
    // Update count
    const countEl = document.getElementById('info-models-count');
    const nAvail = modelsData.filter(m => m.available).length;
    if (countEl) countEl.textContent = `${nAvail} of ${modelsData.length}`;
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
            
            // Update GNN status panel
            updateGNNStatus(data);

            // Update model selector statuses
            if (data.available_models) {
                updateModelStatusDots(data.available_models);
                // If current selection is unavailable, auto-switch to default
                const activeEntry = state.availableModels[state.selectedModel];
                if (!activeEntry || !activeEntry.available) {
                    const defaultKey = data.default_model || 'egnn';
                    state.selectedModel = defaultKey;
                    // UI update
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
    progressText.textContent = `Generating ${numSamples} structures with ${numAtoms} carbon atoms (${state.selectedModel})...`;
    
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
                output_format: outputFormat,
                model: state.selectedModel
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
                // Auto-show first structure in 3D viewer
                if (state.currentStructures.length > 0) {
                    viewStructure(0);
                }
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
                <span class="info-label">Model Used:</span>
                <span class="info-value">${data.model_label || data.model || 'EGNN + DDPM'}</span>
            </div>
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
 * Update GNN status badge and info panel
 */
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
            badge.textContent = 'GNN ✓';
        } else if (hasRefiner && !isTrained) {
            badge.classList.add('gnn-untrained');
            badge.textContent = 'GNN (untrained)';
        } else {
            badge.classList.add('gnn-unavailable');
            badge.textContent = 'GNN —';
        }
    }

    if (infoText) {
        if (hasRefiner && isTrained) {
            infoText.textContent = 'Local GNN refiner is loaded with a trained checkpoint. Interface structures will be refined automatically.';
        } else if (hasRefiner && !isTrained) {
            infoText.textContent = 'Local GNN is loaded but no trained checkpoint was found. The GNN defaults to near-zero displacements (safe pass-through). You can still generate interfaces.';
        } else if (refinerError) {
            infoText.textContent = `Local GNN unavailable: ${refinerError}. Interface generation will proceed without local refinement.`;
        } else {
            infoText.textContent = 'Local GNN not configured (no INTERFACEML_LOCAL_GNN_CHECKPOINT set). Interface generation will proceed without local refinement.';
        }
    }
}

/**
 * Handle AI interface (fullerene + perovskite) generation
 */
async function handleAIInterfaceGeneration() {
    if (!state.aiAvailable) {
        showMessage('AI module is not available', 'error');
        return;
    }

    // Check base file uploaded
    const baseInfo = state.uploadedFiles['ai-interface-base'];
    if (!baseInfo || !baseInfo.filename) {
        showMessage('Please upload a perovskite base structure first', 'error');
        return;
    }

    // Parse Miller index
    const millerStr = document.getElementById('ai-iface-miller').value.trim();
    const millerParts = millerStr.split(/[\s,]+/).map(Number);
    if (millerParts.length !== 3 || millerParts.some(isNaN)) {
        showMessage('Miller index must be three integers (e.g. 0 0 1)', 'error');
        return;
    }

    // Parse supercell
    const supercellVal = document.getElementById('ai-iface-supercell').value;
    let supercellXY = null;
    if (supercellVal !== 'auto') {
        const parts = supercellVal.split(',').map(Number);
        if (parts.length === 2) supercellXY = parts;
    }

    // Parse xy_frac
    const xyFracX = parseFloat(document.getElementById('ai-iface-xy-frac-x').value) || 0.5;
    const xyFracY = parseFloat(document.getElementById('ai-iface-xy-frac-y').value) || 0.5;

    // Parse termination
    const terminationVal = document.getElementById('ai-iface-termination').value;
    const termination = terminationVal === 'auto' ? null : terminationVal;

    // Build request payload
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

    // Show progress
    const progressDiv = document.getElementById('ai-iface-progress');
    const progressFill = document.getElementById('ai-iface-progress-fill');
    const progressText = document.getElementById('ai-iface-progress-text');
    const resultsDiv = document.getElementById('ai-iface-results');

    progressDiv.style.display = 'block';
    resultsDiv.classList.add('hidden');
    progressFill.style.width = '0%';
    const modelLabel = (state.availableModels[state.selectedModel] || {}).label || state.selectedModel;
    progressText.textContent = `Generating fullerene via ${modelLabel}...`;

    // Animate pipeline steps
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
                // Update pipeline display
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
            // --- Async path: submit task, poll for completion ---
            const submitResp = await fetch('/api/ai/generate-interface-async', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const submitData = await submitResp.json();

            if (!submitResp.ok || submitData.status !== 'accepted') {
                throw new Error(submitData.error || 'Failed to submit async task');
            }

            const taskId = submitData.task_id;
            const pollUrl = submitData.poll_url;
            progressText.textContent = `Task submitted (${numSamples} samples). Polling for results...`;

            // Poll every 2s until done
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
                        // else PENDING — keep polling
                    } catch (e) {
                        clearInterval(pollTimer);
                        reject(e);
                    }
                }, 2000);
            });

            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (pollResult.status === 'success') {
                progressText.textContent = '✓ Interface structures generated!';
                setPipelineStep(5);
                setTimeout(() => {
                    progressDiv.style.display = 'none';
                    displayInterfaceResults(pollResult);
                }, 1000);
                showMessage(`Interface structure(s) generated (${pollResult.num_samples} sample(s), ${pollResult.n_atoms} atoms each) in ${pollResult.generation_time}s`, 'success');
            } else {
                throw new Error(pollResult.error || 'Generation failed');
            }

        } else {
            // --- Sync path: single sample, direct call ---
            const response = await fetch('/api/ai/generate-interface', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (response.ok && data.status === 'success') {
                progressText.textContent = '✓ Interface generated successfully!';
                setPipelineStep(5); // all complete

                setTimeout(() => {
                    progressDiv.style.display = 'none';
                    displayInterfaceResults(data);
                }, 1000);

                showMessage(`Interface structure(s) generated (${data.num_samples} sample(s), ${data.n_atoms} atoms each) in ${data.generation_time}s`, 'success');
            } else {
                throw new Error(data.error || 'Interface generation failed');
            }
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressDiv.style.display = 'none';
        resetPipelineSteps();
        showMessage(`Interface generation failed: ${error.message}`, 'error');
        console.error('Interface generation error:', error);
    }
}

/**
 * Set pipeline step state: marks steps up to `step` as completed, current step as active
 */
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
    // Re-mark step 1 if base is uploaded
    if (state.uploadedFiles['ai-interface-base']) {
        const step1 = document.getElementById('pipe-step-1');
        if (step1) step1.classList.add('completed');
    }
}

/**
 * Display interface generation results (supports batch samples)
 */
function displayInterfaceResults(data) {
    const resultsDiv = document.getElementById('ai-iface-results');
    resultsDiv.classList.remove('hidden');

    const samples = data.samples || [{
        sample_index: 0,
        download_url: data.download_url,
        n_atoms: data.n_atoms,
        metadata: data.metadata,
    }];

    const numSamples = samples.length;
    const batchHeader = numSamples > 1
        ? `<div class="batch-header"><h3>✅ ${numSamples} Interface Structures Generated</h3>
             <span class="meta-value">${data.generation_time}s total</span></div>`
        : '';

    let cardsHtml = '';
    for (const sample of samples) {
        const meta = sample.metadata || {};
        const supercell = meta.supercell || {};
        const refinerUsed = meta.local_refiner_used ? 'Yes' : 'No';
        const refinerTrained = meta.local_refiner_trained ? 'Yes (trained)' : 'No (default pass-through)';
        const termTop = (meta.termination_top || []).join(', ') || '—';
        const termBot = (meta.termination_bottom || []).join(', ') || '—';
        const sampleLabel = numSamples > 1 ? ` — Sample #${sample.sample_index + 1}` : '';
        const refinerWarn = data.local_refiner_error
            ? `<div class="gnn-info-panel" style="margin-top:0.5rem; border-left-color: #f59e0b;">
                 ⚠️ Local GNN note: ${data.local_refiner_error}
               </div>`
            : '';

        cardsHtml += `
            <div class="interface-result-card">
                <h3>✅ Interface Structure${sampleLabel}</h3>

                <div class="interface-meta-grid">
                    <div class="interface-meta-item">
                        <span class="meta-label">Model</span>
                        <span class="meta-value">${data.model_label || data.model || '—'}</span>
                    </div>
                    <div class="interface-meta-item">
                        <span class="meta-label">Total Atoms</span>
                        <span class="meta-value">${sample.n_atoms}</span>
                    </div>
                    ${numSamples === 1 ? `<div class="interface-meta-item">
                        <span class="meta-label">Generation Time</span>
                        <span class="meta-value">${data.generation_time}s</span>
                    </div>` : ''}
                    <div class="interface-meta-item">
                        <span class="meta-label">Supercell</span>
                        <span class="meta-value">${supercell.nx || '?'}×${supercell.ny || '?'}</span>
                    </div>
                    <div class="interface-meta-item">
                        <span class="meta-label">Termination</span>
                        <span class="meta-value">${meta.termination || 'auto'}</span>
                    </div>
                    <div class="interface-meta-item">
                        <span class="meta-label">Top Species</span>
                        <span class="meta-value">${termTop}</span>
                    </div>
                    <div class="interface-meta-item">
                        <span class="meta-label">Bottom Species</span>
                        <span class="meta-value">${termBot}</span>
                    </div>
                    <div class="interface-meta-item">
                        <span class="meta-label">GNN Refiner Used</span>
                        <span class="meta-value">${refinerUsed}</span>
                    </div>
                    <div class="interface-meta-item">
                        <span class="meta-label">GNN Trained</span>
                        <span class="meta-value">${refinerTrained}</span>
                    </div>
                    <div class="interface-meta-item">
                        <span class="meta-label">Base Structure</span>
                        <span class="meta-value">${data.base_filename || '—'}</span>
                    </div>
                </div>

                ${refinerWarn}

                <div class="interface-download-bar">
                    <a href="${sample.download_url}" class="btn btn-primary btn-sm" download>
                        💾 Download VASP (POSCAR)
                    </a>
                    <span style="color: var(--text-secondary); font-size: 0.85rem;">
                        File: ${sample.download_url.split('/').pop()}
                    </span>
                </div>
            </div>
        `;
    }

    resultsDiv.innerHTML = batchHeader + cardsHtml;
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * ===== Three.js 3D Molecule Viewer =====
 */
let viewer3D = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    animationId: null,
    autoRotate: false,
    moleculeGroup: null,
    initialized: false,
    renderMode: 'ball-stick',   // 'ball-stick' or 'wireframe'
    currentPositions: null,
    currentEdges: null,
};

function initViewer() {
    if (viewer3D.initialized) return;

    const canvas = document.getElementById('structure-canvas');
    if (!canvas) {
        console.error('Structure canvas element not found');
        showMessage('Structure viewer canvas not found', 'error');
        return;
    }
    if (typeof THREE === 'undefined') {
        console.error('Three.js library not loaded');
        showMessage('3D viewer library failed to load. Please check your network connection and refresh.', 'error');
        return;
    }

    try {
    const container = canvas.parentElement;
    const w = container.clientWidth || 800;
    const h = 500;

    // Scene
    viewer3D.scene = new THREE.Scene();
    viewer3D.scene.background = new THREE.Color(0xf0f2f5);

    // Camera
    viewer3D.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    viewer3D.camera.position.set(0, 0, 15);

    // Renderer
    viewer3D.renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    viewer3D.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    viewer3D.renderer.setSize(w, h);

    // OrbitControls
    if (typeof THREE.OrbitControls === 'undefined') {
        console.error('OrbitControls not loaded');
        showMessage('3D viewer controls failed to load. Please refresh.', 'error');
        return;
    }
    viewer3D.controls = new THREE.OrbitControls(viewer3D.camera, canvas);
    viewer3D.controls.enableDamping = true;
    viewer3D.controls.dampingFactor = 0.08;
    viewer3D.controls.rotateSpeed = 0.8;

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    viewer3D.scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7);
    viewer3D.scene.add(dirLight);
    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-5, -5, -5);
    viewer3D.scene.add(backLight);

    viewer3D.moleculeGroup = new THREE.Group();
    viewer3D.scene.add(viewer3D.moleculeGroup);
    viewer3D.initialized = true;

    // Animate loop
    function animate() {
        viewer3D.animationId = requestAnimationFrame(animate);
        if (viewer3D.autoRotate) {
            viewer3D.moleculeGroup.rotation.y += 0.005;
        }
        viewer3D.controls.update();
        viewer3D.renderer.render(viewer3D.scene, viewer3D.camera);
    }
    animate();
    console.log('3D viewer initialized successfully, canvas:', w, 'x', h);
    } catch (err) {
        console.error('Failed to initialize 3D viewer:', err);
        showMessage('Failed to initialize 3D viewer: ' + err.message, 'error');
    }
}

function loadMolecule(positions, edges) {
    if (!viewer3D.initialized) initViewer();
    if (!viewer3D.initialized || !viewer3D.moleculeGroup) {
        console.error('3D viewer not initialized, cannot load molecule');
        showMessage('3D viewer failed to initialize. Please check browser console for details.', 'error');
        return;
    }

    // Store for render-mode switching
    viewer3D.currentPositions = positions;
    viewer3D.currentEdges = edges;
    updateRenderModeButtons();

    try {

    // Clear previous molecule
    while (viewer3D.moleculeGroup.children.length > 0) {
        const child = viewer3D.moleculeGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
        viewer3D.moleculeGroup.remove(child);
    }

    if (!positions || positions.length === 0) return;

    // ---- Resolve bonds (shared by both modes) ----
    const n = positions.length;
    let cx = 0, cy = 0, cz = 0;
    for (const p of positions) { cx += p[0]; cy += p[1]; cz += p[2]; }
    cx /= n; cy /= n; cz /= n;

    let resolvedEdges = edges && edges.length > 0 ? edges : null;
    if (!resolvedEdges) {
        // Auto-detect bonds by distance (C-C bond ~1.4–1.6 Å)
        resolvedEdges = [];
        const maxBondDist2 = 1.85 * 1.85;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const dx = positions[i][0] - positions[j][0];
                const dy = positions[i][1] - positions[j][1];
                const dz = positions[i][2] - positions[j][2];
                if (dx*dx + dy*dy + dz*dz <= maxBondDist2) {
                    resolvedEdges.push([i, j]);
                }
            }
        }
    }

    const mode = viewer3D.renderMode;

    if (mode === 'wireframe') {
        // ===== WIREFRAME / KEYLINE MODE =====
        // Edges as thin bright lines, small vertex dots
        const lineColor = 0x5b8def;
        const vertexColor = 0x667eea;

        // Build line segments geometry
        const lineVerts = [];
        for (const [i, j] of resolvedEdges) {
            if (i >= n || j >= n) continue;
            const p1x = positions[i][0] - cx, p1y = positions[i][1] - cy, p1z = positions[i][2] - cz;
            const p2x = positions[j][0] - cx, p2y = positions[j][1] - cy, p2z = positions[j][2] - cz;
            const len2 = (p1x-p2x)**2 + (p1y-p2y)**2 + (p1z-p2z)**2;
            if (len2 < 0.0001 || len2 > 12.25) continue;
            lineVerts.push(p1x, p1y, p1z, p2x, p2y, p2z);
        }
        if (lineVerts.length > 0) {
            const lineGeo = new THREE.BufferGeometry();
            lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(lineVerts, 3));
            const lineMat = new THREE.LineBasicMaterial({ color: lineColor, linewidth: 2 });
            const lines = new THREE.LineSegments(lineGeo, lineMat);
            viewer3D.moleculeGroup.add(lines);
        }

        // Carbon atom spheres (normal size for clarity)
        const dotGeo = new THREE.SphereGeometry(0.25, 16, 12);
        const dotMat = new THREE.MeshPhongMaterial({ color: vertexColor, shininess: 80, specular: 0x8899cc });
        for (const p of positions) {
            const dot = new THREE.Mesh(dotGeo, dotMat);
            dot.position.set(p[0] - cx, p[1] - cy, p[2] - cz);
            viewer3D.moleculeGroup.add(dot);
        }
    } else {
        // ===== BALL & STICK MODE (default) =====
        const atomMat = new THREE.MeshPhongMaterial({ color: 0x333333, shininess: 80, specular: 0x666666 });
        const atomGeo = new THREE.SphereGeometry(0.25, 16, 12);
        const bondMat = new THREE.MeshPhongMaterial({ color: 0x888888, shininess: 40 });

        for (const p of positions) {
            const mesh = new THREE.Mesh(atomGeo, atomMat);
            mesh.position.set(p[0] - cx, p[1] - cy, p[2] - cz);
            viewer3D.moleculeGroup.add(mesh);
        }

        for (const [i, j] of resolvedEdges) {
            if (i >= n || j >= n) continue;
            const p1 = new THREE.Vector3(positions[i][0] - cx, positions[i][1] - cy, positions[i][2] - cz);
            const p2 = new THREE.Vector3(positions[j][0] - cx, positions[j][1] - cy, positions[j][2] - cz);
            const len = p1.distanceTo(p2);
            if (len < 0.01 || len > 3.5) continue;

            const bondGeo = new THREE.CylinderGeometry(0.06, 0.06, len, 6, 1);
            bondGeo.translate(0, len / 2, 0);
            bondGeo.rotateX(Math.PI / 2);
            const bond = new THREE.Mesh(bondGeo, bondMat);
            bond.position.copy(p1);
            bond.lookAt(p2);
            viewer3D.moleculeGroup.add(bond);
        }
    }

    // Fit camera
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
    console.log('Molecule loaded (' + mode + '):', positions.length, 'atoms, maxR:', maxR.toFixed(2));
    } catch (err) {
        console.error('Error loading molecule:', err);
        showMessage('Error rendering 3D structure: ' + err.message, 'error');
    }
}

/**
 * View structure in 3D viewer
 */
function viewStructure(index) {
    if (index >= state.currentStructures.length) return;
    
    const struct = state.currentStructures[index];
    const viewerDiv = document.getElementById('ai-viewer');
    const viewerInfo = document.getElementById('viewer-info');
    
    // Make viewer visible first
    viewerDiv.classList.remove('hidden');
    viewerInfo.innerHTML = `
        <div class="info-grid">
            <div class="info-item">
                <span class="info-label">Filename:</span>
                <span class="info-value">${struct.filename}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Atoms:</span>
                <span class="info-value">${struct.num_atoms || '?'}</span>
            </div>
        </div>
    `;
    
    // Scroll to viewer
    viewerDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Wait for browser reflow so the viewer div has proper dimensions,
    // then initialize/render the 3D molecule
    requestAnimationFrame(() => {
        if (struct.positions && struct.positions.length > 0) {
            console.log('Loading molecule with', struct.positions.length, 'atoms,', (struct.edges || []).length, 'edges');
            loadMolecule(struct.positions, struct.edges);
        } else {
            showMessage('No position data available for 3D view', 'warning');
        }
    });
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
    if (!viewer3D.initialized) return;
    viewer3D.autoRotate = !viewer3D.autoRotate;
    showMessage(viewer3D.autoRotate ? 'Auto-rotation enabled' : 'Auto-rotation disabled', 'info');
}

function resetView() {
    if (!viewer3D.initialized) return;
    viewer3D.moleculeGroup.rotation.set(0, 0, 0);
    viewer3D.camera.position.set(0, 0, 15);
    viewer3D.camera.lookAt(0, 0, 0);
    viewer3D.controls.target.set(0, 0, 0);
    viewer3D.controls.update();
    viewer3D.autoRotate = false;
    showMessage('View reset', 'info');
}

function downloadCurrent() {
    if (state.currentStructures.length > 0) {
        window.location.href = state.currentStructures[0].download_url;
    }
}

/**
 * Switch render mode between ball-stick and wireframe
 */
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
