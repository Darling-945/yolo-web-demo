/**
 * Multi-file upload handler with video support
 * Maintains compatibility with existing interface
 */

// Global variables
let uploadedFiles = [];
let selectedFilesInfo = [];

// Initialize multi-file upload
document.addEventListener('DOMContentLoaded', function() {
    initializeMultiFileUpload();
});

function initializeMultiFileUpload() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput) return;

    // Modify the file input to handle multiple files
    fileInput.addEventListener('change', handleMultipleFileSelect);

    // Update form submission to handle multiple files
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleMultipleFileSubmit);
    }
}

function handleMultipleFileSelect(event) {
    const files = event.target.files;
    if (!files || files.length === 0) {
        clearPreviews();
        return;
    }

    console.log(`选择了 ${files.length} 个文件`);
    uploadedFiles = [];
    selectedFilesInfo = [];

    const previewContainer = document.getElementById('previewContainer');
    const previewGrid = document.getElementById('previewGrid');
    const previewTitle = document.getElementById('previewTitle');
    const fileCount = document.getElementById('fileCount');

    if (previewContainer) {
        previewContainer.style.display = 'block';
    }

    if (previewTitle) {
        previewTitle.textContent = files.length === 1 ? '文件预览' : '文件预览';
    }

    if (fileCount) {
        fileCount.textContent = files.length;
    }

    if (previewGrid) {
        previewGrid.innerHTML = ''; // Clear previous previews
    }

    // Process each file
    Array.from(files).forEach((file, index) => {
        const fileInfo = processSingleFile(file, index);
        if (fileInfo) {
            selectedFilesInfo.push(fileInfo);
            createPreviewItem(fileInfo, previewGrid);
        }
    });

    // Enable upload button if we have valid files
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.disabled = selectedFilesInfo.length === 0;
    }

    // Show success message
    const validCount = selectedFilesInfo.length;
    if (validCount > 0) {
        showToast('success', `已成功添加 ${validCount} 个文件`, 'success');
    }

    // Show error message for failed files
    const failedCount = files.length - validCount;
    if (failedCount > 0) {
        showToast('error', `${failedCount} 个文件无效或格式不支持`, 'error');
    }
}

function processSingleFile(file, index) {
    // More robust file extension extraction
    const fileName = file.name;
    const lastDotIndex = fileName.lastIndexOf('.');
    const fileExtension = lastDotIndex > -1 ? fileName.substring(lastDotIndex).toLowerCase() : '';

    // Also check MIME type as backup
    const mimeType = file.type || '';
    const isVideoMime = mimeType.startsWith('video/');
    const isImageMime = mimeType.startsWith('image/');

    // Validate file type (case-insensitive)
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'];
    const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'];

    // Check both extension and MIME type
    const hasValidExtension = imageExtensions.includes(fileExtension) || videoExtensions.includes(fileExtension);
    const hasValidMime = isVideoMime || isImageMime;

    console.debug(`Processing file: ${fileName}, Extension: ${fileExtension}, MIME: ${mimeType}`);

    if (!hasValidExtension && !hasValidMime) {
        console.error(`不支持的文件类型: Extension=${fileExtension}, MIME=${mimeType}`);
        showToast('error', `不支持的文件类型: ${fileName}`, 'error');
        return null;
    }

    // Determine file type (prefer extension over MIME)
    let fileType, isImage;
    if (imageExtensions.includes(fileExtension)) {
        fileType = 'image';
        isImage = true;
    } else if (videoExtensions.includes(fileExtension)) {
        fileType = 'video';
        isImage = false;
    } else if (isVideoMime) {
        fileType = 'video';
        isImage = false;
    } else if (isImageMime) {
        fileType = 'image';
        isImage = true;
    } else {
        console.error(`无法确定文件类型: ${fileName}`);
        showToast('error', `无法确定文件类型: ${fileName}`, 'error');
        return null;
    }

    // Validate file size
    const maxSize = isImage ? 50 * 1024 * 1024 : 500 * 1024 * 1024; // 50MB for images, 500MB for videos

    if (file.size > maxSize) {
        const maxSizeMB = maxSize / (1024 * 1024);
        console.error(`文件过大: ${formatFileSize(file.size)}, 最大允许: ${maxSizeMB}MB`);
        showToast('error', `文件过大: ${fileName} (${formatFileSize(file.size)} > ${maxSizeMB}MB)`, 'error');
        return null;
    }

    console.debug(`File accepted: ${fileName}, Type: ${fileType}, Size: ${formatFileSize(file.size)}`);

    return {
        file: file,
        name: file.name,
        type: fileType,
        size: file.size,
        sizeFormatted: formatFileSize(file.size),
        extension: fileExtension,
        index: index
    };
}

function createPreviewItem(fileInfo, container) {
    const col = document.createElement('div');
    col.className = 'col-12 col-md-6 col-lg-4';

    const card = document.createElement('div');
    card.className = 'card preview-item fade-in';
    card.style.position = 'relative';

    // Create preview content based on file type
    if (fileInfo.type === 'image') {
        createImagePreview(fileInfo, card);
    } else {
        createVideoPreview(fileInfo, card);
    }

    // Add file info
    const infoDiv = document.createElement('div');
    infoDiv.className = 'card-body p-2';
    infoDiv.innerHTML = `
        <h6 class="card-title small mb-1 text-truncate">${fileInfo.name}</h6>
        <div class="d-flex justify-content-between align-items-center">
            <span class="badge bg-${fileInfo.type === 'image' ? 'primary' : 'success'} badge-sm">
                ${fileInfo.type === 'image' ? '图片' : '视频'}
            </span>
            <small class="text-muted">${fileInfo.sizeFormatted}</small>
        </div>
    `;

    card.appendChild(infoDiv);
    col.appendChild(card);
    container.appendChild(col);
}

function createImagePreview(fileInfo, card) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.createElement('img');
        img.className = 'card-img-top';
        img.style.height = '150px';
        img.style.objectFit = 'cover';
        img.src = e.target.result;
        img.alt = fileInfo.name;
        card.insertBefore(img, card.firstChild);
    };
    reader.readAsDataURL(fileInfo.file);
}

function createVideoPreview(fileInfo, card) {
    // For videos, we create a placeholder with file icon
    const placeholder = document.createElement('div');
    placeholder.className = 'card-img-top d-flex align-items-center justify-content-center bg-light';
    placeholder.style.height = '150px';
    placeholder.innerHTML = `
        <div class="text-center">
            <i class="fas fa-video fa-3x text-secondary mb-2"></i>
            <div class="small text-muted">${fileInfo.extension.toUpperCase()}</div>
        </div>
    `;
    card.insertBefore(placeholder, card.firstChild);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function handleMultipleFileSubmit(event) {
    event.preventDefault();

    console.log('=== Starting file submission ===');
    console.log('Selected files:', selectedFilesInfo.length);

    if (selectedFilesInfo.length === 0) {
        showToast('error', '请先选择要上传的文件', 'error');
        return;
    }

    // Log all selected files before processing
    console.log('Files to be uploaded:');
    selectedFilesInfo.forEach((fileInfo, index) => {
        console.log(`${index + 1}. Name: ${fileInfo.name}, Type: ${fileInfo.type}, Size: ${fileInfo.sizeFormatted}, Extension: ${fileInfo.extension}`);
    });

    const uploadBtn = document.getElementById('uploadBtn');
    const uploadBtnText = document.getElementById('uploadBtnText');
    const uploadSpinner = document.getElementById('uploadSpinner');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    // Show loading state
    if (uploadBtn) uploadBtn.disabled = true;
    if (uploadBtnText) uploadBtnText.style.display = 'none';
    if (uploadSpinner) uploadSpinner.style.display = 'inline-block';
    if (progressContainer) progressContainer.style.display = 'block';

    showToast('info', `正在处理 ${selectedFilesInfo.length} 个文件...`, 'info');

    // Get configuration parameters
    const modelSelect = document.getElementById('modelSelect');
    const confidenceInput = document.getElementById('confidenceInput');
    const iouInput = document.getElementById('iouInput');

    const formData = new FormData();

    // Add all files to FormData with the correct key name
    selectedFilesInfo.forEach((fileInfo, index) => {
        formData.append(`files`, fileInfo.file);
    });

    // Add configuration parameters
    formData.append('model', modelSelect ? modelSelect.value : 'yolo11n.pt');
    formData.append('confidence', confidenceInput ? confidenceInput.value : '0.25');
    formData.append('iou', iouInput ? iouInput.value : '0.45');

    // Send request to the existing /infer endpoint
    fetch('/infer', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('=== RESPONSE ANALYSIS ===');
        console.log('Response status:', response.status);
        console.log('Response headers:', [...response.headers.entries()]);
        console.log('Response ok:', response.ok);
        console.log('Response URL:', response.url);

        if (!response.ok) {
            return response.text().then(text => {
                console.error('Error response body:', text);
                throw new Error(`HTTP error! status: ${response.status}, body: ${text}`);
            });
        }
        return response.text();
    })
    .then(html => {
        updateProgress(100, '处理完成！');
        showToast('success', '批量推理完成！', 'success');

        // Let the server handle the response and redirect
        document.body.innerHTML = html;
    })
    .catch(error => {
        console.error('=== ERROR DETAILS ===');
        console.error('Full error object:', error);
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
        console.error('=== END ERROR DETAILS ===');
        showToast('error', '处理失败: ' + error.message, 'error');
        resetUploadState();
    });
}

function updateProgress(percentage, message) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    if (progressBar) {
        progressBar.style.width = percentage + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
    }

    if (progressText) {
        progressText.textContent = message || `${percentage}%`;
    }
}

function resetUploadState() {
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadBtnText = document.getElementById('uploadBtnText');
    const uploadSpinner = document.getElementById('uploadSpinner');
    const progressContainer = document.getElementById('progressContainer');

    if (uploadBtn) uploadBtn.disabled = false;
    if (uploadBtnText) uploadBtnText.style.display = 'inline';
    if (uploadSpinner) uploadSpinner.style.display = 'none';
    if (progressContainer) progressContainer.style.display = 'none';
}

function clearPreviews() {
    const previewContainer = document.getElementById('previewContainer');
    const previewGrid = document.getElementById('previewGrid');
    const fileCount = document.getElementById('fileCount');

    if (previewContainer) {
        previewContainer.style.display = 'none';
    }

    if (previewGrid) {
        previewGrid.innerHTML = '';
    }

    if (fileCount) {
        fileCount.textContent = '';
    }

    selectedFilesInfo = [];
    uploadedFiles = [];
}

// Override drag and drop handlers for multiple files
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();

    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length === 0) {
        showToast('error', '没有检测到文件', 'error');
        return;
    }

    // Update file input with dropped files
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        // Create a new DataTransfer object to set files
        const dataTransfer = new DataTransfer();
        Array.from(files).forEach(file => {
            dataTransfer.items.add(file);
        });
        fileInput.files = dataTransfer.files;

        // Trigger the change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }

    const fileNames = Array.from(files).map(f => f.name).join(', ');
    showToast('success', `已添加文件: ${fileNames}`, 'success');
}

// Extend drag and drop functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');

    if (uploadArea) {
        // Update dragover text
        const uploadHelp = document.getElementById('upload-help');
        if (uploadHelp) {
            uploadHelp.innerHTML = '支持图片格式：JPG, JPEG, PNG, WEBP, BMP, GIF, TIFF<br>支持视频格式：MP4, AVI, MOV, MKV, WMV, FLV, WEBM, M4V<br>图片最大 50MB，视频最大 500MB<br>可同时选择多个文件';
        }

        // Replace existing drop handler
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        uploadArea.addEventListener('drop', handleDrop, false);
    }
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function unhighlight(e) {
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
}

// Toast notification function (if not already defined)
if (typeof showToast === 'undefined') {
    function showToast(type, message, title = '') {
        // Create toast container if it doesn't exist
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.style.position = 'fixed';
            toastContainer.style.top = '20px';
            toastContainer.style.right = '20px';
            toastContainer.style.zIndex = '9999';
            document.body.appendChild(toastContainer);
        }

        // Create toast element
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} alert-dismissible fade show mb-2`;
        toast.style.minWidth = '300px';
        toast.innerHTML = `
            ${title ? `<strong>${title}</strong><br>` : ''}
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        toastContainer.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }
}