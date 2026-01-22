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
    pdosCounter: 0     // Counter for unique PDOS IDs
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeFileUploads();
    initializeButtons();
    initializeModeSwitcher();
    initializePdosUploads();
    checkServerHealth();
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
