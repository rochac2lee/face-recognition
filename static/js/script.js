document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const clearBtn = document.getElementById('clearBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');
    const resultImage = document.getElementById('resultImage');
    const matchesList = document.getElementById('matchesList');

    let selectedFile = null;

    // Event listeners para drag and drop
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    fileInput.addEventListener('change', handleFileSelect);
    uploadBtn.addEventListener('click', uploadFile);
    clearBtn.addEventListener('click', clearAll);

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

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        // Validar tipo de arquivo
        if (!file.type.startsWith('image/')) {
            showError('Por favor, selecione apenas arquivos de imagem.');
            return;
        }

        // Validar tamanho (16MB)
        if (file.size > 16 * 1024 * 1024) {
            showError('O arquivo é muito grande. Tamanho máximo: 16MB.');
            return;
        }

        selectedFile = file;
        uploadBtn.disabled = false;
        
        // Mostrar preview da imagem
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadArea.innerHTML = `
                <div class="upload-content">
                    <img src="${e.target.result}" style="max-width: 200px; max-height: 200px; border-radius: 10px; margin-bottom: 15px;">
                    <h3>Imagem selecionada: ${file.name}</h3>
                    <p>Clique em "Buscar no Álbum" para processar</p>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    }

    async function uploadFile() {
        if (!selectedFile) {
            showError('Por favor, selecione uma imagem primeiro.');
            return;
        }

        hideError();
        showLoading();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                showResults(data);
            } else {
                showError(data.error || 'Erro desconhecido ocorreu.');
            }
        } catch (err) {
            showError('Erro de conexão. Tente novamente.');
            console.error('Erro:', err);
        } finally {
            hideLoading();
        }
    }

    function showResults(data) {
        // Mostrar imagem com detecção
        resultImage.src = 'data:image/jpeg;base64,' + data.image_with_boxes;
        
        // Mostrar correspondências
        matchesList.innerHTML = '';
        
        if (data.album_images_with_boxes && data.album_images_with_boxes.length > 0) {
            data.album_images_with_boxes.forEach((albumImage, index) => {
                const imgElement = document.createElement('img');
                imgElement.src = `data:image/jpeg;base64,${albumImage.image_base64}`;
                imgElement.alt = `Correspondência ${index + 1}`;
                imgElement.className = 'album-match-image';
                imgElement.title = `Confiança: ${(albumImage.confidence * 100).toFixed(1)}% - Clique para ampliar`;
                
                matchesList.appendChild(imgElement);
                
                // Adicionar evento de clique na imagem
                const title = `Correspondência ${index + 1}`;
                const details = `
                    <strong>Arquivo:</strong> ${albumImage.filename}<br>
                    <strong>Confiança:</strong> ${(albumImage.confidence * 100).toFixed(1)}%<br>
                    <strong>Distância:</strong> ${albumImage.distance.toFixed(3)}
                `;
                makeImageClickable(imgElement, title, details);
            });
        } else {
            matchesList.innerHTML = '<p style="text-align: center; color: #666; font-style: italic;">Nenhuma correspondência encontrada no álbum.</p>';
        }

        // Mostrar estatísticas simples
        const statsElement = document.createElement('p');
        statsElement.className = 'stats-info';
        statsElement.innerHTML = `
            <strong>Faces detectadas:</strong> ${data.faces_detected} | 
            <strong>Correspondências:</strong> ${data.album_images_with_boxes ? data.album_images_with_boxes.length : 0}
        `;
        matchesList.insertBefore(statsElement, matchesList.firstChild);

        // Adicionar evento de clique na imagem enviada
        const uploadedImageTitle = 'Imagem Enviada';
        const uploadedImageDetails = `
            <strong>Faces detectadas:</strong> ${data.faces_detected}<br>
            <strong>Correspondências:</strong> ${data.album_images_with_boxes ? data.album_images_with_boxes.length : 0}<br>
            <em>💡 Clique na imagem para ver em tela cheia</em>
        `;
        makeImageClickable(resultImage, uploadedImageTitle, uploadedImageDetails);

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function clearAll() {
        selectedFile = null;
        uploadBtn.disabled = true;
        resultsSection.style.display = 'none';
        hideError();
        
        // Restaurar upload area original
        uploadArea.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon">📷</div>
                <h3>Arraste e solte uma imagem aqui</h3>
                <p>ou clique para selecionar</p>
            </div>
        `;
        
        fileInput.value = '';
    }

    function showLoading() {
        loading.style.display = 'block';
        uploadBtn.disabled = true;
    }

    function hideLoading() {
        loading.style.display = 'none';
        uploadBtn.disabled = !selectedFile;
    }

    function showError(message) {
        errorMessage.textContent = message;
        error.style.display = 'block';
        error.scrollIntoView({ behavior: 'smooth' });
    }

    function hideError() {
        error.style.display = 'none';
    }

    // Funcionalidade do modal de tela cheia
    function setupFullscreenModal() {
        const modal = document.getElementById('fullscreenModal');
        const modalImg = document.getElementById('fullscreenImage');
        const modalTitle = document.getElementById('modalTitle');
        const modalDetails = document.getElementById('modalDetails');
        const closeBtn = document.querySelector('.close');

        // Função para abrir modal
        function openModal(imgSrc, title, details) {
            modalImg.src = imgSrc;
            modalTitle.textContent = title;
            modalDetails.innerHTML = details;
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden'; // Impede scroll do body
        }

        // Função para fechar modal
        function closeModal() {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto'; // Reabilita scroll do body
        }

        // Event listeners
        closeBtn.addEventListener('click', closeModal);
        
        // Fechar com clique no fundo
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeModal();
            }
        });

        // Fechar com ESC
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && modal.style.display === 'block') {
                closeModal();
            }
        });

        return { openModal, closeModal };
    }

    // Configurar modal na inicialização
    const modalController = setupFullscreenModal();

    // Adicionar funcionalidade para imagens clicáveis
    function makeImageClickable(imgElement, title, details) {
        imgElement.addEventListener('click', function() {
            modalController.openModal(imgElement.src, title, details);
        });
    }

    // Expor função globalmente para uso nas outras funções
    window.makeImageClickable = makeImageClickable;
});