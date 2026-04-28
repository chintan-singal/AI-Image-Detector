document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    // Views
    const viewUpload = document.getElementById('view-upload');
    const viewLoading = document.getElementById('view-loading');
    const viewResult = document.getElementById('view-result');
    const viewError = document.getElementById('view-error');
    const viewHistory = document.getElementById('view-history');

    // Nav
    const navHistory = document.getElementById('nav-history');

    // Upload Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const dropzoneContent = document.getElementById('dropzone-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const actionSection = document.getElementById('action-section');
    const analyzeBtn = document.getElementById('analyze-btn');

    // Loading Elements
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    // Result Elements
    const backBtn = document.getElementById('back-btn');
    const resultImage = document.getElementById('result-image');
    const predictionText = document.getElementById('prediction-text');
    const confidenceScore = document.getElementById('confidence-score');
    const meterFill = document.getElementById('meter-fill');
    const resultMessage = document.getElementById('result-message');
    const reasonsList = document.getElementById('reasons-list');
    const heatmapOverlay = document.getElementById('heatmap-overlay');
    const toggleExplanationBtn = document.getElementById('toggle-explanation');
    
    // Error Elements
    const retryBtn = document.getElementById('retry-btn');
    const errorMessage = document.getElementById('error-message');

    // History Elements
    const historyList = document.getElementById('history-list');
    const emptyHistory = document.getElementById('empty-history');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // --- State ---
    let currentFile = null;
    let currentHeatmap = null;
    let explanationVisible = false;

    // --- View Switching ---
    function switchView(viewElement) {
        [viewUpload, viewLoading, viewResult, viewError, viewHistory].forEach(v => {
            if (v) {
                v.classList.remove('active');
                v.classList.add('hidden');
            }
        });
        viewElement.classList.remove('hidden');
        // trigger reflow
        void viewElement.offsetWidth;
        viewElement.classList.add('active');
    }

    // --- Upload Logic ---
    dropzone.addEventListener('click', (e) => {
        // Prevent click if clicking buttons inside preview
        if (e.target.closest('button')) return;
        if (!currentFile) {
            fileInput.click();
        }
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (JPG, PNG, WEBP).');
            return;
        }

        currentFile = file;
        showPreview(file);
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            dropzone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            actionSection.classList.remove('hidden');
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropzone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        actionSection.classList.add('hidden');
    });

    analyzeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!currentFile) return;
        startAnalysis();
    });

    // --- Loading & API Logic ---
    function startAnalysis() {
        switchView(viewLoading);
        
        let progress = 0;
        updateProgress(0);

        // Fake progress up to 90% while waiting
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 5;
                if (progress > 90) progress = 90;
                updateProgress(progress);
            }
        }, 300);

        const formData = new FormData();
        formData.append('image', currentFile);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            clearInterval(progressInterval);
            updateProgress(100);
            
            setTimeout(() => {
                showResult(data);
            }, 600); // Wait a bit to show 100%
        })
        .catch(err => {
            console.error('API Error:', err);
            clearInterval(progressInterval);
            showError("Failed to connect to the server. Make sure it is running.");
        });
    }

    function updateProgress(percent) {
        percent = Math.max(0, Math.min(100, percent));
        progressFill.style.width = `${percent}%`;
        progressText.innerText = `${Math.floor(percent)}%`;
    }

    // --- Result Logic ---
    function showResult(data) {
        if (!data || !data.success) {
            showError(data.error || "Unknown error occurred during analysis.");
            return;
        }

        // Set Image
        resultImage.src = imagePreview.src;

        // Set Prediction & Badge colors
        const prediction = data.prediction || 'Uncertain';
        predictionText.innerText = prediction;
        
        predictionText.className = 'prediction-value'; // reset
        if (prediction === 'AI Generated') {
            predictionText.classList.add('ai');
        } else if (prediction === 'Real') {
            predictionText.classList.add('real');
        } else {
            predictionText.classList.add('uncertain');
        }

        // Set Confidence
        const conf = data.confidence || 0;
        confidenceScore.innerText = `${conf}%`;
        
        // Timeout to allow DOM update before css transition
        setTimeout(() => {
            meterFill.style.width = `${conf}%`;
        }, 100);

        // Set Message
        resultMessage.innerText = data.message || '';

        // Set Reasons
        reasonsList.innerHTML = '';
        if (data.reasons && data.reasons.length > 0) {
            data.reasons.forEach(r => {
                const li = document.createElement('li');
                li.innerText = r;
                reasonsList.appendChild(li);
            });
        }

        // Set Heatmap
        if (data.heatmap_b64) {
            currentHeatmap = data.heatmap_b64;
            heatmapOverlay.src = data.heatmap_b64;
            heatmapOverlay.classList.remove('hidden');
            toggleExplanationBtn.parentElement.style.display = 'block';
        } else {
            currentHeatmap = null;
            heatmapOverlay.classList.add('hidden');
            toggleExplanationBtn.parentElement.style.display = 'none';
        }
        
        // Reset toggle state
        explanationVisible = false;
        toggleExplanationBtn.classList.remove('active');
        toggleExplanationBtn.querySelector('.btn-text').innerText = 'Show AI Explanation';
        heatmapOverlay.classList.remove('visible');

        switchView(viewResult);
        
        // Save to History
        saveToHistory(data, resultImage.src);
    }

    backBtn.addEventListener('click', () => {
        // Reset state
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropzone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        actionSection.classList.add('hidden');
        meterFill.style.width = '0%';
        switchView(viewUpload);
    });

    // --- Error Logic ---
    function showError(msg) {
        errorMessage.innerText = msg;
        switchView(viewError);
    }

    retryBtn.addEventListener('click', () => {
        if (currentFile) {
            startAnalysis();
        } else {
            switchView(viewUpload);
        }
    });

    // --- History Logic ---
    navHistory.addEventListener('click', (e) => {
        e.preventDefault();
        renderHistory();
        switchView(viewHistory);
    });

    clearHistoryBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear your prediction history?')) {
            localStorage.removeItem('detection_history');
            renderHistory();
        }
    });

    function saveToHistory(data, imageSrc) {
        const history = JSON.parse(localStorage.getItem('detection_history') || '[]');
        
        const newItem = {
            id: Date.now(),
            timestamp: new Date().toLocaleString(),
            prediction: data.prediction,
            confidence: data.confidence,
            image: imageSrc, // Note: This stores DataURL in localStorage. Might hit limits if too many images.
            message: data.message
        };

        // Keep last 20 items to avoid storage limits
        history.unshift(newItem);
        if (history.length > 20) history.pop();
        
        localStorage.setItem('detection_history', JSON.stringify(history));
    }

    function renderHistory() {
        const history = JSON.parse(localStorage.getItem('detection_history') || '[]');
        
        // Clear previous list except empty message
        historyList.innerHTML = '';
        historyList.appendChild(emptyHistory);

        if (history.length === 0) {
            emptyHistory.classList.remove('hidden');
            return;
        }

        emptyHistory.classList.add('hidden');

        history.forEach(item => {
            const card = document.createElement('div');
            card.className = 'history-card';
            
            const badgeClass = item.prediction === 'AI Generated' ? 'badge-ai' : 
                              (item.prediction === 'Real' ? 'badge-real' : 'badge-uncertain');
            
            card.innerHTML = `
                <div class="history-thumb">
                    <img src="${item.image}" alt="Detection">
                    <span class="history-badge ${badgeClass}">${item.prediction}</span>
                </div>
                <div class="history-info">
                    <span class="history-date">${item.timestamp}</span>
                    <h4 class="history-result-text">${item.prediction}</h4>
                    <p class="history-conf">Confidence: ${item.confidence}%</p>
                </div>
            `;

            card.addEventListener('click', () => {
                // Restore result view for this item
                resultImage.src = item.image;
                predictionText.innerText = item.prediction;
                predictionText.className = 'prediction-value ' + 
                    (item.prediction === 'AI Generated' ? 'ai' : 
                    (item.prediction === 'Real' ? 'real' : 'uncertain'));
                
                confidenceScore.innerText = `${item.confidence}%`;
                resultMessage.innerText = item.message || '';
                reasonsList.innerHTML = '<li>Details loaded from history.</li>';
                
                // Hide explanation for history items (unless we start saving heatmaps too)
                toggleExplanationBtn.parentElement.style.display = 'none';
                heatmapOverlay.classList.add('hidden');
                
                meterFill.style.width = '0%';
                switchView(viewResult);
                setTimeout(() => {
                    meterFill.style.width = `${item.confidence}%`;
                }, 100);
            });

            historyList.appendChild(card);
        });
    }

    // --- Explainability Logic ---
    toggleExplanationBtn.addEventListener('click', () => {
        explanationVisible = !explanationVisible;
        
        if (explanationVisible) {
            toggleExplanationBtn.classList.add('active');
            toggleExplanationBtn.querySelector('.btn-text').innerText = 'Hide AI Explanation';
            heatmapOverlay.classList.add('visible');
        } else {
            toggleExplanationBtn.classList.remove('active');
            toggleExplanationBtn.querySelector('.btn-text').innerText = 'Show AI Explanation';
            heatmapOverlay.classList.remove('visible');
        }
    });

});
