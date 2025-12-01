// Brain Tumor Detection UI JavaScript

const API_BASE = window.location.origin;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const uploadForm = document.getElementById('uploadForm');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const removeImageBtn = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const newAnalysisBtn = document.getElementById('newAnalysis');
const modelSelect = document.getElementById('modelSelect');

let selectedFile = null;

// Upload Area Click
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File Input Change
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle File Selection
function handleFileSelect(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    const validExtensions = ['.nii', '.dcm'];
    
    const isValidType = validTypes.includes(file.type) || 
                       validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if (!isValidType) {
        showNotification('Invalid file type. Please upload PNG, JPG, JPEG, NII, or DCM files.', 'error');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File too large. Maximum size is 16MB.', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show preview for image files
    if (validTypes.includes(file.type)) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadArea.style.display = 'none';
            previewSection.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        imagePreview.style.display = 'none';
        previewSection.innerHTML = `
            <div style="padding: 40px; text-align: center; background: var(--light); border-radius: 12px;">
                <p style="font-size: 1.2rem; color: var(--dark); margin-bottom: 10px;">
                    📄 ${file.name}
                </p>
                <p style="color: #64748b;">
                    Medical image file ready for analysis
                </p>
                <button type="button" class="btn-secondary" id="removeImage" style="margin-top: 20px;">Remove File</button>
            </div>
        `;
        
        // Re-attach remove handler
        document.getElementById('removeImage').addEventListener('click', removeImage);
    }
    
    analyzeBtn.disabled = false;
}

// Remove Image
function removeImage() {
    selectedFile = null;
    imageInput.value = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    imagePreview.style.display = 'block';
    analyzeBtn.disabled = true;
}

removeImageBtn.addEventListener('click', removeImage);

// Form Submit
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
        showNotification('Please select an image first.', 'error');
        return;
    }
    
    const age = document.getElementById('ageInput').value;
    
    await analyzeImage(selectedFile, age);
});

// Helper: fetch with timeout and robust JSON handling
async function fetchWithTimeout(url, options = {}, timeoutMs = 120000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const resp = await fetch(url, { ...options, signal: controller.signal });
        let data;
        const ct = resp.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
            data = await resp.json();
        } else {
            const text = await resp.text();
            // Try to parse JSON from text fallback
            try { data = JSON.parse(text); } catch { data = { raw: text }; }
        }
        return { response: resp, data };
    } finally {
        clearTimeout(id);
    }
}

// Analyze Image
async function analyzeImage(file, age) {
    loadingOverlay.style.display = 'flex';
    analyzeBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', file);
    if (age) {
        formData.append('age', age);
    }
    if (modelSelect && modelSelect.value) {
        formData.append('model', modelSelect.value);
    }
    
    try {
        const { response, data } = await fetchWithTimeout(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: formData
        }, 180000); // 3 minute timeout for first-time model load
        
        if (!response.ok) {
            const errMsg = data && (data.error || data.message) ? (data.error || data.message) : `HTTP ${response.status}`;
            throw new Error(errMsg);
        }
        
        displayResults(data);
        
    } catch (error) {
        showNotification(`Error: ${error.message || 'Analysis failed or timed out. Ensure the server is running.'}`, 'error');
        console.error('Analysis error:', error);
    } finally {
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display Results
function displayResults(data) {
    console.log('Analysis results:', data);
    
    // Hide upload section, show results
    document.querySelector('.upload-section').style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Status Badge
    const statusBadge = document.getElementById('statusBadge');
    const summary = data.summary || {};
    
    if (summary.tumor_present === true) {
        statusBadge.className = 'status-badge tumor';
        statusBadge.innerHTML = `
            <div style="font-size: 1.5rem; margin-bottom: 5px;">🔴 Tumor Detected</div>
            <div style="font-size: 1rem; font-weight: normal;">${summary.primary_label || 'Tumor Present'}</div>
        `;
    } else if (summary.tumor_present === false) {
        statusBadge.className = 'status-badge no-tumor';
        statusBadge.innerHTML = `
            <div style="font-size: 1.5rem; margin-bottom: 5px;">✅ No Tumor Detected</div>
            <div style="font-size: 1rem; font-weight: normal;">Scan appears normal</div>
        `;
    } else {
        statusBadge.className = 'status-badge uncertain';
        statusBadge.innerHTML = `
            <div style="font-size: 1.5rem; margin-bottom: 5px;">⚠️ Uncertain Result</div>
            <div style="font-size: 1rem; font-weight: normal;">${summary.primary_label || 'Further analysis recommended'}</div>
        `;
    }
    
    // Classification Results
    displayClassification(data.classification);
    
    // Segmentation Results
    displaySegmentation(data.segmentation);
    
    // Features
    displayFeatures(data.features);
    
    // Survival Prediction
    displaySurvival(data.survival);
    
    // Recommendations
    displayRecommendations(summary.recommendations || []);
}

// Display Classification
function displayClassification(classification) {
    const content = document.getElementById('classificationContent');
    
    if (!classification || !classification.tumor_detected === undefined) {
        content.innerHTML = '<p style="color: #94a3b8;">Classification data not available</p>';
        return;
    }
    
    const confidence = (classification.confidence * 100).toFixed(1);
    const tumorType = classification.tumor_type || 'Unknown';
    
    content.innerHTML = `
        <div class="metric">
            <span class="metric-label">Tumor Detected</span>
            <span class="metric-value">${classification.tumor_detected ? 'Yes' : 'No'}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Type</span>
            <span class="tumor-type">${tumorType}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence</span>
            <span class="metric-value">${confidence}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence}%"></div>
        </div>
    `;
    
    // Add probabilities if available
    if (classification.probabilities) {
        const probsHTML = Object.entries(classification.probabilities)
            .map(([label, prob]) => `
                <div class="metric">
                    <span class="metric-label">${label}</span>
                    <span class="metric-value">${(prob * 100).toFixed(1)}%</span>
                </div>
            `).join('');
        content.innerHTML += probsHTML;
    }
}

// Display Segmentation
function displaySegmentation(segmentation) {
    const content = document.getElementById('segmentationContent');
    
    if (!segmentation) {
        content.innerHTML = '<p style="color: #94a3b8;">Segmentation data not available</p>';
        return;
    }
    
    const coverage = (segmentation.coverage * 100).toFixed(2);
    
    content.innerHTML = `
        <div class="metric">
            <span class="metric-label">Status</span>
            <span class="metric-value">${segmentation.success ? 'Success' : 'Failed'}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Tumor Coverage</span>
            <span class="metric-value">${coverage}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Tumor Pixels</span>
            <span class="metric-value">${segmentation.tumor_pixels.toLocaleString()}</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${coverage}%; background: linear-gradient(90deg, #06b6d4, #4f46e5);"></div>
        </div>
    `;
}

// Display Features
function displayFeatures(features) {
    const content = document.getElementById('featuresContent');
    
    if (!features || !features.tumor_present) {
        content.innerHTML = '<p style="color: #94a3b8;">No tumor features detected</p>';
        return;
    }
    
    content.innerHTML = `
        <div class="metric">
            <span class="metric-label">Regions Detected</span>
            <span class="metric-value">${features.num_regions}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Area</span>
            <span class="metric-value">${features.total_area.toLocaleString()} px</span>
        </div>
        <div class="metric">
            <span class="metric-label">Coverage</span>
            <span class="metric-value">${features.coverage_pct.toFixed(2)}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Largest Region Area</span>
            <span class="metric-value">${(features.largest_region_area ?? 0).toLocaleString()} px</span>
        </div>
        <div class="metric">
            <span class="metric-label">Eccentricity</span>
            <span class="metric-value">${Number(features.eccentricity ?? 0).toFixed(3)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Solidity</span>
            <span class="metric-value">${Number(features.solidity ?? 0).toFixed(3)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Mean Intensity</span>
            <span class="metric-value">${Number(features.mean_intensity ?? 0).toFixed(2)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Mask Mean Confidence</span>
            <span class="metric-value">${Number(features.mean_confidence ?? 0).toFixed(3)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Mask Max Confidence</span>
            <span class="metric-value">${Number(features.max_confidence ?? 0).toFixed(3)}</span>
        </div>
    `;
    
    // Additional details if available
    if (features.centroid || features.bounding_box) {
        const centroid = features.centroid ? `(${features.centroid[0].toFixed(1)}, ${features.centroid[1].toFixed(1)})` : 'N/A';
        const bbox = features.bounding_box ? `[#${features.bounding_box.join(', ')}]` : 'N/A';
        const extra = document.createElement('div');
        extra.style.marginTop = '10px';
        extra.innerHTML = `
            <div class="metric">
                <span class="metric-label">Centroid (x, y)</span>
                <span class="metric-value">${centroid}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Bounding Box (min_row, min_col, max_row, max_col)</span>
                <span class="metric-value">${bbox}</span>
            </div>
        `;
        content.appendChild(extra);
    }
}

// Display Survival
function displaySurvival(survival) {
    const content = document.getElementById('survivalContent');
    
    if (!survival || !survival.prediction) {
        content.innerHTML = '<p style="color: #94a3b8;">Survival prediction not available</p>';
        return;
    }
    
    const confidence = (survival.confidence * 100).toFixed(1);
    
    content.innerHTML = `
        <div class="metric">
            <span class="metric-label">Prediction</span>
            <span class="metric-value">${survival.prediction}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence</span>
            <span class="metric-value">${confidence}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence}%; background: linear-gradient(90deg, #f59e0b, #ef4444);"></div>
        </div>
        <p style="margin-top: 15px; font-size: 0.9rem; color: #64748b;">
            ⚠️ This is a prediction. Consult with medical professionals for accurate prognosis.
        </p>
    `;
}

// Display Recommendations
function displayRecommendations(recommendations) {
    const content = document.getElementById('recommendationsContent');
    
    if (!recommendations || recommendations.length === 0) {
        content.innerHTML = '<p style="color: #94a3b8;">No recommendations available</p>';
        return;
    }
    
    content.innerHTML = recommendations
        .map(rec => `<div class="recommendation-item">${rec}</div>`)
        .join('');
}

// New Analysis
newAnalysisBtn.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    removeImage();
});

// Show Notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#4f46e5'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Check API Health on Load
window.addEventListener('load', async () => {
    try {
        const { response, data } = await fetchWithTimeout(`${API_BASE}/api/health`, {}, 5000);
        console.log('API Health:', data);
        if (!response.ok) throw new Error('Server unhealthy');
    } catch (error) {
        console.error('API health check failed:', error);
        analyzeBtn.disabled = true;
        showNotification('Server is not reachable. Start the backend and refresh this page.', 'error');
        // Re-enable the button after a short delay in case the server comes up
        setTimeout(() => { analyzeBtn.disabled = false; }, 8000);
    }
});

// Add CSS animations
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
