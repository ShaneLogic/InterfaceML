/**
 * InterfaceML Web Application
 * Client-side JavaScript for heterojunction modeling interface
 */

// State management
const state = {
    uploadedFiles: {},
    currentTab: 'adsorbate',
    layerMode: 'fix'  // 'fix' or 'split'
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeFileUploads();
    initializeButtons();
    initializeModeSwitcher();
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
            
            if (mode === 'fix') {
                fixParams.classList.remove('hidden');
                splitParams.classList.add('hidden');
                state.layerMode = 'fix';
            } else if (mode === 'split') {
                fixParams.classList.add('hidden');
                splitParams.classList.remove('hidden');
                state.layerMode = 'split';
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
    
    // Compute density button
    const computeDensityBtn = document.getElementById('compute-density-btn');
    if (computeDensityBtn) {
        computeDensityBtn.addEventListener('click', computeDensity);
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
 * Compute density difference
 */
async function computeDensity() {
    showMessage('Density analysis functionality coming soon', 'info');
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
