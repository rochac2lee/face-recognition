// Elementos DOM
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const clearBtn = document.getElementById('clearBtn');
const unifiedSection = document.getElementById('unifiedSection');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const resultImage = document.getElementById('resultImage');
const matchesList = document.getElementById('matchesList');
const matchesCount = document.getElementById('matchesCount');
const resultsStats = document.getElementById('resultsStats');

let selectedFile = null;

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
uploadBtn.addEventListener('click', uploadFile);
clearBtn.addEventListener('click', clearAll);

// Drag and Drop
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File Selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!isValidImage(file)) {
        showError('Por favor, selecione um arquivo de imagem válido (JPG, PNG, GIF)');
        return;
    }
    
    selectedFile = file;
    uploadBtn.disabled = false;
    
    // Preview da imagem
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadArea.innerHTML = `
            <div class="upload-content">
                <div class="image-preview">
                    <img src="${e.target.result}" alt="Preview" style="max-width: 100px; max-height: 100px; border-radius: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <p style="margin-top: 1rem; color: var(--gray-600);">${file.name}</p>
                </div>
            </div>
        `;
    };
    reader.readAsDataURL(file);
}

function isValidImage(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    return validTypes.includes(file.type);
}

// Upload
async function uploadFile() {
    if (!selectedFile) {
        showError('Por favor, selecione uma imagem primeiro');
        return;
    }
    
    showLoading();
    hideError();
    hideResults();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        hideLoading();
        
        if (data.success) {
            if (data.matches && data.matches.length > 0) {
                showResults(data);
            } else {
                showError(data.message || 'Nenhuma correspondência encontrada');
            }
        } else {
            showError(data.error || 'Erro no processamento');
        }
    } catch (error) {
        hideLoading();
        showError('Erro na comunicação com o servidor: ' + error.message);
    }
}

// Display Results
function showResults(data) {
    // Mostrar imagem enviada
    if (data.image_with_boxes) {
        resultImage.src = 'data:image/jpeg;base64,' + data.image_with_boxes;
    }
    
    // Atualizar estatísticas
    const stats = `${data.faces_detected} face(s) detectada(s) • ${data.matches_found} correspondência(s) encontrada(s)`;
    resultsStats.textContent = stats;
    matchesCount.textContent = `${data.matches_found} fotos`;
    
    // Limpar matches anteriores
    matchesList.innerHTML = '';
    
    // Criar cards para cada match
    data.matches.forEach((match, index) => {
        const matchCard = createMatchCard(match, index);
        matchesList.appendChild(matchCard);
    });
    
    // Mostrar seção de resultados
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createMatchCard(match, index) {
    const card = document.createElement('div');
    card.className = 'match-card';
    
                       const accuracy = typeof match.accuracy === 'number' ? match.accuracy.toFixed(1) : 
                                        (typeof match.confidence === 'number' ? match.confidence.toFixed(1) : 'N/A');
    
    card.innerHTML = `
        <img src="/album/${match.filename}" alt="Match ${index + 1}" class="match-image" loading="lazy">
        <div class="match-overlay">
            <div class="match-stats">
                <span class="match-stat accuracy">${accuracy}%</span>
            </div>
        </div>
    `;
    
    // Adicionar evento de clique para ampliar
    card.addEventListener('click', () => {
        openImageModal(`/album/${match.filename}`, match);
    });
    
    return card;
}

function openImageModal(imageSrc, matchData) {
    // Criar modal simples
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        cursor: pointer;
    `;
    
    const img = document.createElement('img');
    img.src = imageSrc;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        border-radius: 8px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    `;
    
    modal.appendChild(img);
    document.body.appendChild(modal);
    
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
}

// UI Helpers
function showLoading() {
    loading.style.display = 'block';
    uploadBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    uploadBtn.disabled = false;
}

function showError(message) {
    errorMessage.textContent = message;
    error.style.display = 'flex';
}

function hideError() {
    error.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showUploadOnly() {
    resultsSection.style.display = 'none';
    unifiedSection.scrollIntoView({ behavior: 'smooth' });
}

function clearAll() {
    selectedFile = null;
    fileInput.value = '';
    uploadBtn.disabled = true;
    hideError();
    showUploadOnly();
    
    // Restaurar upload area
    uploadArea.innerHTML = `
        <div class="upload-content">
            <div class="upload-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21,15 16,10 5,21"/>
                </svg>
                <span>Clique aqui ou arraste uma imagem</span>
            </div>
        </div>
    `;
}

// Inicialização
document.addEventListener('DOMContentLoaded', () => {
    console.log('Face Recognition AI - Interface carregada');
});