/**
 * 摄像头实时检测功能
 * 使用 getUserMedia API 获取摄像头，定期捕获帧并发送到后端进行推理
 */

// 全局变量
let videoElement = null;
let canvasElement = null;
let ctx = null;
let stream = null;
let isDetecting = false;
let detectionInterval = null;
let lastFrameTime = 0;
const TARGET_FPS = 10; // 目标FPS，避免过载
const FRAME_INTERVAL = 1000 / TARGET_FPS;

// 统计信息
let frameCount = 0;
let lastFpsUpdate = Date.now();
let currentFps = 0;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeCamera();
    setupControls();
});

/**
 * 初始化摄像头相关元素
 */
function initializeCamera() {
    videoElement = document.getElementById('cameraVideo');
    canvasElement = document.getElementById('detectionCanvas');

    if (videoElement && canvasElement) {
        ctx = canvasElement.getContext('2d');

        // 监听视频元数据加载完成
        videoElement.addEventListener('loadedmetadata', function() {
            // 设置canvas尺寸与视频一致
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
        });
    }
}

/**
 * 设置控制按钮和参数调节
 */
function setupControls() {
    // 开始/停止按钮
    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');
    const captureBtn = document.getElementById('captureBtn');

    if (startBtn) {
        startBtn.addEventListener('click', startCamera);
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', stopCamera);
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', captureScreenshot);
    }

    // 置信度控制
    setupConfidenceControls();

    // IOU控制
    setupIOUControls();

    // 模型选择变更提示
    const modelSelect = document.getElementById('cameraModelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', function(e) {
            const modelText = e.target.options[e.target.selectedIndex].text;
            showToast('info', `模型已切换为: ${modelText}`, 'info');
        });
    }
}

/**
 * 置信度双模式控制
 */
function setupConfidenceControls() {
    const slider = document.getElementById('cameraConfidenceSlider');
    const input = document.getElementById('cameraConfidenceInput');
    const badge = document.getElementById('cameraConfidenceValue');

    if (!slider || !input || !badge) return;

    // 滑块变化时更新输入框和标签
    slider.addEventListener('input', function(e) {
        const value = parseFloat(e.target.value);
        input.value = value;
        badge.textContent = value;
    });

    // 输入框变化时更新滑块和标签
    input.addEventListener('input', function(e) {
        let value = parseFloat(e.target.value);

        // 验证输入值范围
        if (isNaN(value)) value = 0.25;
        if (value < 0.1) value = 0.1;
        if (value > 0.9) value = 0.9;

        // 四舍五入到步长的精度
        value = Math.round(value / 0.05) * 0.05;

        input.value = value;
        slider.value = value;
        badge.textContent = value;
    });

    // 输入框失去焦点时验证格式
    input.addEventListener('blur', function(e) {
        let value = parseFloat(e.target.value);
        if (isNaN(value) || value < 0.1 || value > 0.9) {
            value = 0.25;
            input.value = value;
            slider.value = value;
            badge.textContent = value;
        }
    });
}

/**
 * IOU双模式控制
 */
function setupIOUControls() {
    const slider = document.getElementById('cameraIouSlider');
    const input = document.getElementById('cameraIouInput');
    const badge = document.getElementById('cameraIouValue');

    if (!slider || !input || !badge) return;

    // 滑块变化时更新输入框和标签
    slider.addEventListener('input', function(e) {
        const value = parseFloat(e.target.value);
        input.value = value;
        badge.textContent = value;
    });

    // 输入框变化时更新滑块和标签
    input.addEventListener('input', function(e) {
        let value = parseFloat(e.target.value);

        // 验证输入值范围
        if (isNaN(value)) value = 0.45;
        if (value < 0.1) value = 0.1;
        if (value > 0.9) value = 0.9;

        // 四舍五入到步长的精度
        value = Math.round(value / 0.05) * 0.05;

        input.value = value;
        slider.value = value;
        badge.textContent = value;
    });

    // 输入框失去焦点时验证格式
    input.addEventListener('blur', function(e) {
        let value = parseFloat(e.target.value);
        if (isNaN(value) || value < 0.1 || value > 0.9) {
            value = 0.45;
            input.value = value;
            slider.value = value;
            badge.textContent = value;
        }
    });
}

/**
 * 开启摄像头并开始检测
 */
async function startCamera() {
    try {
        // 检查浏览器是否支持 getUserMedia
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showToast('error', '您的浏览器不支持摄像头访问，请使用 Chrome/Edge/Firefox 等现代浏览器', 'error');
            console.error('getUserMedia not supported');
            return;
        }

        // 检查当前访问地址是否安全
        const currentProtocol = window.location.protocol;
        const currentHost = window.location.hostname;
        if (currentProtocol !== 'https:' && currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
            showToast('error', '浏览器安全限制：请使用 https:// 或 localhost 访问', 'error');
            console.error('Insecure context detected');
            return;
        }

        // 请求摄像头权限
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            },
            audio: false
        });

        // 设置视频源
        videoElement.srcObject = stream;

        // 等待视频开始播放
        videoElement.onloadedmetadata = function() {
            videoElement.play();
            isDetecting = true;

            // 隐藏占位符，显示控制按钮
            const placeholder = document.getElementById('cameraPlaceholder');
            const startBtn = document.getElementById('startCameraBtn');
            const stopBtn = document.getElementById('stopCameraBtn');
            const captureBtn = document.getElementById('captureBtn');

            if (placeholder) placeholder.style.display = 'none';
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'inline-block';
            if (captureBtn) captureBtn.style.display = 'inline-block';

            // 启用截图按钮
            if (captureBtn) captureBtn.disabled = false;

            // 开始检测循环
            startDetectionLoop();

            showToast('success', '摄像头已启动，开始实时检测', 'success');
        };

    } catch (error) {
        console.error('摄像头启动失败:', error);
        let errorMsg = '无法访问摄像头';
        let detailMsg = '';

        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMsg = '摄像头权限被拒绝';
            detailMsg = '请在浏览器地址栏左侧点击锁图标，允许摄像头访问权限';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            errorMsg = '未找到摄像头设备';
            detailMsg = '请检查摄像头是否正确连接';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            errorMsg = '摄像头无法访问';
            detailMsg = '摄像头可能被其他应用占用，请关闭其他使用摄像头的程序';
        } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
            errorMsg = '摄像头不支持请求的配置';
            detailMsg = '尝试使用其他摄像头或降低分辨率要求';
        } else if (error.name === 'TypeError' || error.name === 'TypeErrorError') {
            errorMsg = '摄像头类型不支持';
            detailMsg = '请检查摄像头设备是否正常';
        } else if (error.name === 'SecurityError') {
            errorMsg = '安全限制';
            detailMsg = '请使用 https:// 或 localhost 访问此页面';
        }

        showToast('error', errorMsg + (detailMsg ? ' - ' + detailMsg : ''), 'error');

        // 在控制台输出详细信息供调试
        console.error('摄像头错误详情:', {
            name: error.name,
            message: error.message,
            toString: error.toString()
        });
    }
}

/**
 * 停止摄像头和检测
 */
function stopCamera() {
    isDetecting = false;

    // 停止检测循环
    if (detectionInterval) {
        cancelAnimationFrame(detectionInterval);
        detectionInterval = null;
    }

    // 停止摄像头流
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    // 清空canvas
    if (ctx && canvasElement) {
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }

    // 显示占位符，隐藏停止按钮
    const placeholder = document.getElementById('cameraPlaceholder');
    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');
    const captureBtn = document.getElementById('captureBtn');

    if (placeholder) placeholder.style.display = 'flex';
    if (startBtn) startBtn.style.display = 'inline-block';
    if (stopBtn) stopBtn.style.display = 'none';
    if (captureBtn) captureBtn.style.display = 'none';

    // 清空统计信息
    updateStatistics(0, 0, 0);

    // 隐藏检测结果
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) resultsSection.style.display = 'none';

    showToast('info', '已停止检测', 'info');
}

/**
 * 开始检测循环
 */
function startDetectionLoop() {
    if (!isDetecting) return;

    const now = Date.now();
    const elapsed = now - lastFrameTime;

    if (elapsed >= FRAME_INTERVAL) {
        lastFrameTime = now - (elapsed % FRAME_INTERVAL);

        // 捕获帧并发送到后端
        captureAndDetect();
    }

    // 计算FPS
    frameCount++;
    const fpsElapsed = now - lastFpsUpdate;
    if (fpsElapsed >= 1000) {
        currentFps = Math.round((frameCount * 1000) / fpsElapsed);
        frameCount = 0;
        lastFpsUpdate = now;
    }

    // 继续循环
    detectionInterval = requestAnimationFrame(startDetectionLoop);
}

/**
 * 捕获帧并发送到后端进行检测
 */
async function captureAndDetect() {
    if (!videoElement || !canvasElement || !ctx) return;

    try {
        // 设置canvas尺寸
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;

        // 绘制当前帧到canvas
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        // 获取图像数据
        canvasElement.toBlob(async function(blob) {
            if (!blob) return;

            const startTime = performance.now();

            // 准备FormData
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            // 获取参数
            const modelSelect = document.getElementById('cameraModelSelect');
            const confidenceInput = document.getElementById('cameraConfidenceInput');
            const iouInput = document.getElementById('cameraIouInput');

            formData.append('model', modelSelect ? modelSelect.value : 'yolo11n.pt');
            formData.append('confidence', confidenceInput ? confidenceInput.value : '0.25');
            formData.append('iou', iouInput ? iouInput.value : '0.45');

            try {
                // 发送到后端
                const response = await fetch('/api/camera_detect', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                const endTime = performance.now();
                const inferenceTime = Math.round(endTime - startTime);

                if (result.success) {
                    // 清空canvas并重新绘制视频帧
                    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

                    // 绘制检测框
                    drawDetections(result.detections);

                    // 更新统计信息
                    updateStatistics(result.count, currentFps, inferenceTime);

                    // 更新检测结果列表
                    updateDetectionTable(result.detections);
                } else {
                    console.error('检测失败:', result.error);
                }

            } catch (error) {
                console.error('发送检测请求失败:', error);
            }
        }, 'image/jpeg', 0.8); // 0.8质量，平衡性能和质量

    } catch (error) {
        console.error('捕获帧失败:', error);
    }
}

/**
 * 在canvas上绘制检测结果
 */
function drawDetections(detections) {
    if (!detections || detections.length === 0) return;

    // 定义颜色映射
    const colors = [
        '#FF5733', '#33FF57', '#3357FF', '#FF33F6', '#33FFF6',
        '#F6FF33', '#FF8C33', '#8C33FF', '#FF338C', '#33FF8C'
    ];

    detections.forEach((det, index) => {
        const bbox = det.bbox; // [x1, y1, x2, y2]
        const class_name = det.class;
        const confidence = det.confidence;

        // 选择颜色
        const color = colors[index % colors.length];

        // 计算框的尺寸
        const x = bbox[0];
        const y = bbox[1];
        const width = bbox[2] - bbox[0];
        const height = bbox[3] - bbox[1];

        // 绘制检测框
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // 绘制标签背景
        const label = `${class_name} ${(confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 16px Arial';
        const textWidth = ctx.measureText(label).width;
        const textHeight = 20;

        ctx.fillStyle = color;
        ctx.fillRect(x, y - textHeight - 8, textWidth + 16, textHeight + 8);

        // 绘制标签文字
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(label, x + 8, y - 8);
    });
}

/**
 * 更新统计信息
 */
function updateStatistics(detectionCount, fps, inferenceTime) {
    const countElement = document.getElementById('detectionCount');
    const fpsElement = document.getElementById('fpsCount');
    const timeElement = document.getElementById('inferenceTime');

    if (countElement) countElement.textContent = detectionCount;
    if (fpsElement) fpsElement.textContent = fps;
    if (timeElement) timeElement.textContent = inferenceTime;
}

/**
 * 更新检测结果表格
 */
function updateDetectionTable(detections) {
    const tableBody = document.getElementById('detectionTableBody');
    const resultsSection = document.getElementById('resultsSection');

    if (!tableBody) return;

    // 显示结果区域
    if (detections && detections.length > 0) {
        if (resultsSection) resultsSection.style.display = 'block';

        // 清空表格
        tableBody.innerHTML = '';

        // 添加检测结果行
        detections.forEach(det => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${det.class}</strong></td>
                <td>
                    <div class="d-flex align-items-center gap-2">
                        <div class="progress" style="width: 100px; height: 8px;">
                            <div class="progress-bar bg-success" style="width: ${det.confidence * 100}%"></div>
                        </div>
                        <span>${(det.confidence * 100).toFixed(1)}%</span>
                    </div>
                </td>
                <td><code>[${det.bbox[0].toFixed(0)}, ${det.bbox[1].toFixed(0)}, ${det.bbox[2].toFixed(0)}, ${det.bbox[3].toFixed(0)}]</code></td>
            `;
            tableBody.appendChild(row);
        });
    } else {
        if (resultsSection) resultsSection.style.display = 'none';
        tableBody.innerHTML = '';
    }
}

/**
 * 拍照截图
 */
function captureScreenshot() {
    if (!videoElement || !canvasElement) return;

    try {
        // 创建临时canvas用于保存截图
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoElement.videoWidth;
        tempCanvas.height = videoElement.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // 绘制当前帧
        tempCtx.drawImage(videoElement, 0, 0);

        // 如果有检测结果，也绘制到截图中
        if (ctx && canvasElement) {
            tempCtx.drawImage(canvasElement, 0, 0);
        }

        // 转换为图像并下载
        tempCanvas.toBlob(function(blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `camera_detection_${Date.now()}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('success', '截图已保存', 'success');
        }, 'image/jpeg', 0.95);

    } catch (error) {
        console.error('截图失败:', error);
        showToast('error', '截图失败', 'error');
    }
}

/**
 * 页面卸载时清理资源
 */
window.addEventListener('beforeunload', function() {
    stopCamera();
});
