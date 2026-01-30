// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const uploadSection = document.getElementById('upload-section');
const filePreview = document.getElementById('file-preview');
const previewImage = document.getElementById('preview-image');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const analyzeBtn = document.getElementById('analyze-btn');
const cancelBtn = document.getElementById('cancel-btn');
const loadingSection = document.getElementById('loading-section');
const resultSection = document.getElementById('result-section');
const resultBadge = document.getElementById('result-badge');
const realBar = document.getElementById('real-bar');
const fakeBar = document.getElementById('fake-bar');
const realPercent = document.getElementById('real-percent');
const fakePercent = document.getElementById('fake-percent');
const confidenceScore = document.getElementById('confidence-score');
const resetBtn = document.getElementById('reset-btn');

let selectedFile = null;

// Browse button
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop handlers
dropZone.addEventListener('dragenter', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('active');
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Handle file selection
function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'video/mp4', 'video/avi', 'video/quicktime'];
    
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image (JPG, PNG) or video (MP4, AVI, MOV) file');
        return;
    }

    selectedFile = file;
    
    // Show file info
    fileName.textContent = `> ${file.name}`;
    fileSize.textContent = `[ ${(file.size / 1024 / 1024).toFixed(2)} MB ]`;
    
    // Preview image
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        previewImage.style.display = 'none';
    }
    
    // Show preview section
    dropZone.style.display = 'none';
    filePreview.style.display = 'block';
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Show loading
    filePreview.style.display = 'none';
    loadingSection.style.display = 'block';
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Analysis failed');
        }
        
        const result = await response.json();
        
        // Hide loading
        loadingSection.style.display = 'none';
        uploadSection.style.display = 'none';
        
        // Show result
        displayResult(result);
        
    } catch (error) {
        alert(`Error: ${error.message}`);
        reset();
    }
});

// Display result
function displayResult(result) {
    // Set badge
    if (result.isFake) {
        resultBadge.className = 'result-badge fake';
    } else {
        resultBadge.className = 'result-badge real';
    }
    
    // Set percentages
    realPercent.textContent = `${result.realConfidence.toFixed(2)}%`;
    fakePercent.textContent = `${result.fakeConfidence.toFixed(2)}%`;
    
    // Set progress bars
    realBar.style.width = `${result.realConfidence}%`;
    fakeBar.style.width = `${result.fakeConfidence}%`;
    
    // Set confidence score
    const maxConfidence = Math.max(result.realConfidence, result.fakeConfidence);
    confidenceScore.textContent = `> CONFIDENCE SCORE: ${maxConfidence.toFixed(1)}%`;
    
    // Show result section
    resultSection.style.display = 'block';
}

// Cancel button
cancelBtn.addEventListener('click', () => {
    reset();
});

// Reset button
resetBtn.addEventListener('click', () => {
    reset();
});

// Reset function
function reset() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    
    // Reset display
    dropZone.style.display = 'block';
    filePreview.style.display = 'none';
    loadingSection.style.display = 'none';
    uploadSection.style.display = 'block';
    resultSection.style.display = 'none';
}
