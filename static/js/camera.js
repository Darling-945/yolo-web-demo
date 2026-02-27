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

// 帧率配置（保守设置，避免触发速率限制）
const TARGET_FPS = 3;       // 目标FPS：每秒3次检测
const MIN_FPS = 1;          // 最小FPS：每秒1次
const MAX_FPS = 5;          // 最大FPS：每秒5次
let currentTargetFps = TARGET_FPS;
let FRAME_INTERVAL = 1000 / currentTargetFps;

// 统计信息
let frameCount = 0;
let lastFpsUpdate = Date.now();
let currentFps = 0;

// 请求控制（严格限制，同一时间只允许一个请求）
let pendingRequest = null;
let isRequestInProgress = false;  // 是否有请求正在进行
let lastRequestTime = 0;          // 上次请求时间
const MIN_REQUEST_INTERVAL = 100; // 最小请求间隔(ms)

// 图片压缩配置（YOLO模型标准输入分辨率）
const MODEL_INPUT_SIZE = 640;     // YOLO标准输入尺寸640x640
const JPEG_QUALITY = 0.4;         // JPEG压缩质量0.4（更低以减小传输大小）

// 速率限制控制
let rateLimitBackoff = false;
let rateLimitBackoffUntil = 0;
let consecutive429Errors = 0;
let consecutiveSuccessCount = 0;  // 连续成功计数
const MAX_429_ERRORS = 2;         // 降低阈值：2次429即触发退避
const BACKOFF_MULTIPLIER = 2;
const SUCCESS_TO_RECOVER = 5;     // 连续成功5次后恢复帧率

// 性能监控
let inferenceTimes = [];
const MAX_INFERENCE_SAMPLES = 10;

// 摄像头设备管理
let currentDeviceId = null;
let availableDevices = [];

// 重连控制
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;
let reconnectTimeout = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeCamera();
    setupControls();
    setupNetworkMonitoring();
});

/**
 * 初始化摄像头相关元素
 */
function initializeCamera() {
    videoElement = document.getElementById('cameraVideo');
    canvasElement = document.getElementById('detectionCanvas');

    console.log('初始化摄像头元素:', {
        videoElement: videoElement ? 'found' : 'not found',
        canvasElement: canvasElement ? 'found' : 'not found'
    });

    if (videoElement && canvasElement) {
        ctx = canvasElement.getContext('2d');
        console.log('Canvas context 创建:', ctx ? 'success' : 'failed');

        // 更新canvas显示尺寸以匹配视频
        function updateCanvasDisplaySize() {
            if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
                const rect = videoElement.getBoundingClientRect();
                canvasElement.style.width = rect.width + 'px';
                canvasElement.style.height = rect.height + 'px';
                console.log(`Canvas CSS尺寸更新: ${rect.width}x${rect.height}`);
            }
        }

        // 监听视频元数据加载完成
        videoElement.addEventListener('loadedmetadata', function() {
            console.log(`视频元数据加载: videoWidth=${videoElement.videoWidth}, videoHeight=${videoElement.videoHeight}`);
            if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                console.log(`Canvas像素尺寸: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
            }
        });

        // 监听视频播放事件，更新canvas显示尺寸
        videoElement.addEventListener('playing', function() {
            console.log(`视频开始播放: videoWidth=${videoElement.videoWidth}, videoHeight=${videoElement.videoHeight}`);
            if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
                if (canvasElement.width !== videoElement.videoWidth ||
                    canvasElement.height !== videoElement.videoHeight) {
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                    console.log(`Canvas像素尺寸更新: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                }
            }
            updateCanvasDisplaySize();
        });

        // 监听窗口大小变化，更新canvas显示尺寸
        window.addEventListener('resize', updateCanvasDisplaySize);
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

    // 加载摄像头设备列表
    loadCameraDevices();

    // 监听设备变化
    navigator.mediaDevices?.addEventListener('devicechange', handleDeviceChange);
}

/**
 * 加载可用的摄像头设备
 */
async function loadCameraDevices() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            console.log('设备枚举不支持');
            return;
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        availableDevices = devices.filter(device => device.kind === 'videoinput');

        console.log('检测到摄像头设备:', availableDevices.map(d => ({
            id: d.deviceId,
            label: d.label || `摄像头 ${availableDevices.indexOf(d) + 1}`
        })));

        // 如果有多个设备，添加设备选择器
        if (availableDevices.length > 1) {
            addDeviceSelector();
        }
    } catch (error) {
        console.error('获取设备列表失败:', error);
    }
}

/**
 * 添加摄像头设备选择器
 */
function addDeviceSelector() {
    const modelSelect = document.getElementById('cameraModelSelect');
    if (!modelSelect) return;

    const configSection = modelSelect.closest('.card-body');
    if (!configSection) return;

    // 检查是否已存在选择器
    if (document.getElementById('cameraDeviceSelect')) return;

    // 创建设备选择器容器
    const deviceDiv = document.createElement('div');
    deviceDiv.className = 'col-12 mb-3';
    deviceDiv.innerHTML = `
        <div class="text-center">
            <label for="cameraDeviceSelect" class="form-label fw-semibold d-block mb-3">
                <i class="fas fa-video me-2"></i>选择摄像头
            </label>
            <select class="form-select mx-auto" id="cameraDeviceSelect" style="max-width: 450px;">
                ${availableDevices.map((device, index) => `
                    <option value="${device.deviceId}">
                        ${device.label || `摄像头 ${index + 1}`}
                    </option>
                `).join('')}
            </select>
        </div>
    `;

    // 插入到模型选择之前
    const modelCol = modelSelect.closest('.col-12');
    modelCol.parentNode.insertBefore(deviceDiv, modelCol);

    // 添加事件监听
    const deviceSelect = document.getElementById('cameraDeviceSelect');
    deviceSelect.addEventListener('change', async function(e) {
        if (isDetecting) {
            // 如果正在检测，切换摄像头需要重启
            const confirmSwitch = confirm('切换摄像头需要重新启动检测，是否继续？');
            if (confirmSwitch) {
                stopCamera();
                currentDeviceId = e.target.value;
                await startCamera();
            } else {
                // 恢复原选择
                e.target.value = currentDeviceId || availableDevices[0].deviceId;
            }
        } else {
            currentDeviceId = e.target.value;
        }
    });
}

/**
 * 处理设备变化事件
 */
async function handleDeviceChange() {
    console.log('检测到设备变化');
    if (!isDetecting) {
        await loadCameraDevices();
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
        // 重置重连计数
        reconnectAttempts = 0;

        // 详细的浏览器支持检测
        const diagnostics = checkCameraSupport();

        if (!diagnostics.mediaDevicesSupported) {
            showCameraErrorHTML('browser_not_supported');
            return;
        }

        if (!diagnostics.isSecureContext) {
            showCameraErrorHTML('not_secure_context', diagnostics);
            return;
        }

        // 显示加载状态
        const placeholder = document.getElementById('cameraPlaceholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <div class="text-center p-4">
                    <div class="loading-ring mb-3"></div>
                    <p class="mb-0">正在启动摄像头...</p>
                </div>
            `;
        }

        // 准备视频约束
        const videoConstraints = {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            audio: false
        };

        // 如果选择了特定设备
        if (currentDeviceId) {
            videoConstraints.deviceId = { exact: currentDeviceId };
        } else {
            videoConstraints.facingMode = 'environment';
        }

        // 请求摄像头权限
        stream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints
        });

        // 重新加载设备列表（获取标签）
        await loadCameraDevices();

        // 设置视频源
        videoElement.srcObject = stream;

        // 等待视频开始播放
        videoElement.onloadedmetadata = function() {
            videoElement.play().then(() => {
                isDetecting = true;
                updateUIState('detecting');

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
            }).catch(error => {
                console.error('视频播放失败:', error);
                showToast('error', '视频播放失败', 'error');
                updateUIState('error');
            });
        };

        // 处理视频错误
        videoElement.onerror = function(error) {
            console.error('视频元素错误:', error);
            handleCameraError(new Error('视频加载失败'));
        };

    } catch (error) {
        console.error('摄像头启动失败:', error);
        handleCameraError(error);
    }
}

/**
 * 检查摄像头支持状态
 */
function checkCameraSupport() {
    const info = {
        mediaDevicesSupported: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
        isSecureContext: window.isSecureContext,
        protocol: window.location.protocol,
        hostname: window.location.hostname,
        href: window.location.href,
        userAgent: navigator.userAgent,
        // 检测浏览器类型
        browser: detectBrowser(),
        // 检测是否在本地环境
        isLocalEnvironment: isLocalEnvironment()
    };

    console.log('摄像头支持检测:', info);
    return info;
}

/**
 * 检测浏览器类型
 */
function detectBrowser() {
    const ua = navigator.userAgent;
    if (ua.indexOf('Chrome') > -1 && ua.indexOf('Edg') === -1) return 'Chrome';
    if (ua.indexOf('Edg') > -1) return 'Edge';
    if (ua.indexOf('Firefox') > -1) return 'Firefox';
    if (ua.indexOf('Safari') > -1 && ua.indexOf('Chrome') === -1) return 'Safari';
    if (ua.indexOf('MSIE') > -1 || ua.indexOf('Trident/') > -1) return 'IE';
    return 'Unknown';
}

/**
 * 检测是否在本地环境
 */
function isLocalEnvironment() {
    const hostname = window.location.hostname;
    return hostname === 'localhost' ||
           hostname === '127.0.0.1' ||
           hostname === '[::1]' ||
           hostname.startsWith('192.168.') ||
           hostname.startsWith('10.') ||
           hostname.startsWith('172.');
}

/**
 * 显示详细的HTML错误信息
 */
function showCameraErrorHTML(errorType, diagnostics = null) {
    const placeholder = document.getElementById('cameraPlaceholder');
    if (!placeholder) return;

    let errorHTML = '';

    if (errorType === 'browser_not_supported') {
        errorHTML = `
            <div class="text-center p-4">
                <i class="fas fa-exclamation-triangle fa-4x mb-3 text-warning"></i>
                <h5 class="mb-3">浏览器不支持摄像头访问</h5>
                <p class="text-muted mb-3">您的浏览器可能版本过旧，不支持 getUserMedia API</p>

                <div class="alert alert-info text-start">
                    <strong>检测结果：</strong><br>
                    <code>navigator.mediaDevices:</code> ${!!(navigator.mediaDevices)}<br>
                    <code>navigator.mediaDevices.getUserMedia:</code> ${!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)}<br>
                    <code>当前浏览器:</code> ${diagnostics?.browser || detectBrowser()}<br>
                    <code>UserAgent:</code> ${diagnostics?.userAgent || navigator.userAgent.substring(0, 50)}...
                </div>

                <p class="text-muted mb-3">请使用以下现代浏览器之一：</p>
                <div class="d-flex justify-content-center gap-3 flex-wrap">
                    <span class="badge bg-primary">Chrome 53+</span>
                    <span class="badge bg-primary">Edge 79+</span>
                    <span class="badge bg-primary">Firefox 36+</span>
                    <span class="badge bg-primary">Safari 11+</span>
                </div>

                <div class="mt-4">
                    <a href="https://whatwebcandoi.appspot.com/static/camera.html" target="_blank" class="btn btn-outline-primary btn-sm me-2">
                        <i class="fas fa-external-link-alt me-1"></i>测试浏览器摄像头支持
                    </a>
                    <button class="btn btn-outline-secondary btn-sm" onclick="copyDiagnosticInfo()">
                        <i class="fas fa-copy me-1"></i>复制诊断信息
                    </button>
                </div>
            </div>
        `;
    } else if (errorType === 'not_secure_context') {
        const isLocalhost = diagnostics.hostname === 'localhost' || diagnostics.hostname === '127.0.0.1';
        const isLocalIP = /^\d+\.\d+\.\d+\.\d+$/.test(diagnostics.hostname) &&
                         (diagnostics.hostname.startsWith('192.168.') ||
                          diagnostics.hostname.startsWith('10.') ||
                          diagnostics.hostname.startsWith('172.'));
        const isPublicIP = /^\d+\.\d+\.\d+\.\d+$/.test(diagnostics.hostname) && !isLocalIP;

        errorHTML = `
            <div class="text-center p-4">
                <i class="fas fa-lock fa-4x mb-3 text-danger"></i>
                <h5 class="mb-3">浏览器安全限制</h5>
                <p class="text-muted mb-3">摄像头访问需要安全的上下文（HTTPS 或 localhost）</p>

                <div class="alert alert-warning text-start">
                    <strong>当前环境信息：</strong><br>
                    <strong>访问地址：</strong><code>${diagnostics.href}</code><br>
                    <strong>当前协议：</strong><code>${diagnostics.protocol}</code><br>
                    <strong>主机名：</strong><code>${diagnostics.hostname}</code><br>
                    <strong>安全上下文：</strong><span class="badge ${diagnostics.isSecureContext ? 'bg-success' : 'bg-danger'}">${diagnostics.isSecureContext ? '是' : '否'}</span>
                </div>

                <div class="text-start">
                    <h6 class="fw-semibold mb-3"><i class="fas fa-lightbulb text-warning me-2"></i>解决方案：</h6>

                    ${isPublicIP ? `
                        <div class="alert alert-danger">
                            <strong><i class="fas fa-exclamation-triangle me-2"></i>检测到使用公网 IP 地址访问！</strong><br>
                            这是不安全的，浏览器会阻止摄像头访问。
                        </div>
                        <p class="mb-2">请选择以下方式之一：</p>
                        <div class="accordion" id="solutionAccordion">
                            <div class="accordion-item">
                                <h2 class="accordion-header">
                                    <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#solution1">
                                        <strong>方案1（推荐）：</strong>使用 localhost 访问
                                    </button>
                                </h2>
                                <div id="solution1" class="accordion-collapse collapse show" data-bs-parent="#solutionAccordion">
                                    <div class="accordion-body">
                                        <ol>
                                            <li>确保服务器监听在 <code>127.0.0.1</code> 或 <code>0.0.0.0</code></li>
                                            <li>在浏览器中访问：<code class="user-select-all">http://localhost:5000</code></li>
                                        </ol>
                                        <button class="btn btn-sm btn-primary mt-2" onclick="window.location.href='http://localhost:5000'">
                                            <i class="fas fa-external-link-alt me-1"></i>跳转到 localhost
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="accordion-item">
                                <h2 class="accordion-header">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#solution2">
                                        <strong>方案2：</strong>配置 HTTPS 证书
                                    </button>
                                </h2>
                                <div id="solution2" class="accordion-collapse collapse" data-bs-parent="#solutionAccordion">
                                    <div class="accordion-body">
                                        <p>生成自签名证书：</p>
                                        <pre class="bg-light p-2 rounded"><code>openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes</code></pre>
                                        <p>修改Flask应用使用HTTPS：</p>
                                        <pre class="bg-light p-2 rounded"><code>app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))</code></pre>
                                    </div>
                                </div>
                            </div>
                            <div class="accordion-item">
                                <h2 class="accordion-header">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#solution3">
                                        <strong>方案3：</strong>Chrome 添加安全源例外（仅开发用）
                                    </button>
                                </h2>
                                <div id="solution3" class="accordion-collapse collapse" data-bs-parent="#solutionAccordion">
                                    <div class="accordion-body">
                                        <ol>
                                            <li>在 Chrome 地址栏输入：<code class="user-select-all">chrome://flags/#unsafely-treat-insecure-origin-as-secure</code></li>
                                            <li>启用该选项</li>
                                            <li>在输入框中填入：<code class="user-select-all">http://${diagnostics.hostname}:5000</code></li>
                                            <li>重启浏览器</li>
                                        </ol>
                                        <div class="alert alert-warning mt-2 mb-0">
                                            <small><i class="fas fa-info-circle me-1"></i>此方法仅适用于开发环境，不要在生产环境使用</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ` : isLocalIP ? `
                        <div class="alert alert-warning">
                            <strong><i class="fas fa-network-wired me-2"></i>检测到使用内网 IP 地址访问</strong><br>
                            内网IP也需要使用 HTTPS 或改用 localhost 访问
                        </div>
                        <p class="mb-2">请选择以下方式之一：</p>
                        <ol class="mb-3">
                            <li><strong>推荐：</strong>使用 <code class="user-select-all">http://localhost:5000</code> 访问
                                <button class="btn btn-sm btn-outline-primary ms-2" onclick="window.location.href='http://localhost:5000'">
                                    <i class="fas fa-external-link-alt me-1"></i>跳转
                                </button>
                            </li>
                            <li><strong>或者：</strong>配置 SSL 证书使用 HTTPS 访问</li>
                            <li><strong>临时方案：</strong>Chrome 添加安全源例外（见上方方案3）</li>
                        </ol>
                    ` : isLocalhost ? `
                        <div class="alert alert-info">
                            <strong><i class="fas fa-info-circle me-2"></i>检测到 localhost 访问</strong><br>
                            但浏览器报告这不是安全上下文，可能是浏览器配置问题。
                        </div>
                        <p>请尝试以下步骤：</p>
                        <ul>
                            <li>确保使用 <code>http://localhost:5000</code> 而不是其他变体（如 127.0.0.1）</li>
                            <li>检查浏览器控制台（F12）是否有其他错误信息</li>
                            <li>尝试使用不同的端口（如 8080）</li>
                            <li>清除浏览器缓存后重试</li>
                        </ul>
                    ` : `
                        <div class="alert alert-danger">
                            <strong><i class="fas fa-exclamation-triangle me-2"></i>当前访问方式不支持摄像头</strong>
                        </div>
                        <p class="mb-2">请使用以下方式之一：</p>
                        <ul class="mb-3">
                            <li><strong>本地开发：</strong><code class="user-select-all">http://localhost:5000</code></li>
                            <li><strong>生产环境：</strong>配置 HTTPS 证书</li>
                        </ul>
                    `}
                </div>

                <div class="mt-4">
                    <button class="btn btn-outline-secondary btn-sm" onclick="copyDiagnosticInfo()">
                        <i class="fas fa-copy me-1"></i>复制诊断信息
                    </button>
                    <button class="btn btn-outline-primary btn-sm ms-2" onclick="testCameraAccess()">
                        <i class="fas fa-vial me-1"></i>测试摄像头权限
                    </button>
                </div>
            </div>
        `;
    }

    placeholder.innerHTML = errorHTML;
}

/**
 * 复制诊断信息到剪贴板
 */
function copyDiagnosticInfo() {
    const diagnostics = checkCameraSupport();
    const info = `
摄像头访问诊断信息
====================
浏览器: ${diagnostics.browser}
UserAgent: ${diagnostics.userAgent}
访问地址: ${diagnostics.href}
协议: ${diagnostics.protocol}
主机名: ${diagnostics.hostname}
安全上下文: ${diagnostics.isSecureContext ? '是' : '否'}
媒体设备支持: ${diagnostics.mediaDevicesSupported ? '是' : '否'}

时间: ${new Date().toLocaleString('zh-CN')}
    `.trim();

    navigator.clipboard.writeText(info).then(() => {
        showToast('success', '诊断信息已复制到剪贴板', 'success');
    }).catch(() => {
        // 降级方案
        const textarea = document.createElement('textarea');
        textarea.value = info;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast('success', '诊断信息已复制到剪贴板', 'success');
    });
}

/**
 * 测试摄像头权限
 */
async function testCameraAccess() {
    showToast('info', '正在请求摄像头权限...', 'info');

    try {
        // 先请求权限
        await navigator.mediaDevices.getUserMedia({ video: true });

        // 如果成功，显示详细信息
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        let message = `摄像头权限已获取！\n\n检测到 ${videoDevices.length} 个摄像头设备：\n`;
        videoDevices.forEach((device, index) => {
            message += `\n${index + 1}. ${device.label || '未命名设备'}`;
        });

        showToast('success', '摄像头权限已获取', 'success');

        // 显示设备列表
        const placeholder = document.getElementById('cameraPlaceholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <div class="text-center p-4">
                    <i class="fas fa-check-circle fa-4x mb-3 text-success"></i>
                    <h5 class="mb-3">摄像头权限已获取</h5>
                    <p class="text-muted">检测到 ${videoDevices.length} 个摄像头设备</p>
                    <div class="text-start">
                        ${videoDevices.map((device, index) => `
                            <div class="alert alert-info">
                                <strong>摄像头 ${index + 1}:</strong> ${device.label || '未命名设备'}
                                <br><small class="text-muted">ID: ${device.deviceId.substring(0, 20)}...</small>
                            </div>
                        `).join('')}
                    </div>
                    <button class="btn btn-primary mt-3" onclick="location.reload()">
                        <i class="fas fa-redo me-2"></i>刷新页面重新开始
                    </button>
                </div>
            `;
        }
    } catch (error) {
        console.error('摄像头权限测试失败:', error);
        handleCameraError(error);
    }
}

/**
 * 处理摄像头错误
 */
function handleCameraError(error) {
    let errorMsg = '无法访问摄像头';
    let detailMsg = '';

    switch (error.name) {
        case 'NotAllowedError':
        case 'PermissionDeniedError':
            errorMsg = '摄像头权限被拒绝';
            detailMsg = '请在浏览器地址栏左侧点击图标，允许摄像头访问权限';
            break;
        case 'NotFoundError':
        case 'DevicesNotFoundError':
            errorMsg = '未找到摄像头设备';
            detailMsg = '请检查摄像头是否正确连接';
            break;
        case 'NotReadableError':
        case 'TrackStartError':
            errorMsg = '摄像头无法访问';
            detailMsg = '摄像头可能被其他应用占用';
            break;
        case 'OverconstrainedError':
        case 'ConstraintNotSatisfiedError':
            errorMsg = '摄像头不支持请求的配置';
            detailMsg = '尝试使用其他摄像头或降低分辨率';
            break;
        case 'SecurityError':
            errorMsg = '安全限制';
            detailMsg = '请使用 https:// 或 localhost 访问';
            break;
        default:
            errorMsg = '摄像头启动失败';
            detailMsg = error.message || '未知错误';
    }

    showToast('error', errorMsg, 'error');

    // 显示详细的错误信息
    const placeholder = document.getElementById('cameraPlaceholder');
    if (placeholder) {
        placeholder.innerHTML = `
            <div class="text-center p-4">
                <i class="fas fa-exclamation-circle fa-4x mb-3 text-danger"></i>
                <h5 class="mb-3">${errorMsg}</h5>
                <p class="text-muted">${detailMsg}</p>
                <div class="alert alert-info text-start mt-3">
                    <strong>错误详情：</strong><br>
                    <code>${error.name}: ${error.message}</code>
                </div>
                <button class="btn btn-primary mt-3" onclick="location.reload()">
                    <i class="fas fa-redo me-2"></i>刷新页面重试
                </button>
            </div>
        `;
    }

    console.error('摄像头错误详情:', {
        name: error.name,
        message: error.message,
        stack: error.stack
    });
}

/**
 * 停止摄像头和检测
 */
function stopCamera() {
    isDetecting = false;

    // 取消待处理的请求
    if (pendingRequest) {
        pendingRequest.abort();
        pendingRequest = null;
    }

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

    if (placeholder) {
        placeholder.style.display = 'flex';
        placeholder.innerHTML = `
            <div class="text-center">
                <i class="fas fa-camera fa-4x mb-3" style="color: var(--primary-color);"></i>
                <p class="mb-0">点击"开启摄像头"开始实时检测</p>
            </div>
        `;
    }
    if (startBtn) startBtn.style.display = 'inline-block';
    if (stopBtn) stopBtn.style.display = 'none';
    if (captureBtn) captureBtn.style.display = 'none';

    // 清空统计信息
    updateStatistics(0, 0, 0);

    // 隐藏检测结果
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) resultsSection.style.display = 'none';

    // 重置性能数据
    inferenceTimes = [];
    frameCount = 0;
    currentFps = 0;

    // 清除重连定时器
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }

    // 重置请求控制状态
    isRequestInProgress = false;
    lastRequestTime = 0;

    // 重置速率限制状态
    rateLimitBackoff = false;
    rateLimitBackoffUntil = 0;
    consecutive429Errors = 0;
    consecutiveSuccessCount = 0;
    currentTargetFps = TARGET_FPS;
    FRAME_INTERVAL = 1000 / currentTargetFps;
    hideRateLimitWarning();

    // 更新UI状态
    updateUIState('idle');

    showToast('info', '已停止检测', 'info');
}

/**
 * 开始检测循环
 */
function startDetectionLoop() {
    if (!isDetecting) return;

    const now = Date.now();

    // 检查是否在退避期
    if (rateLimitBackoff && now < rateLimitBackoffUntil) {
        const waitTime = Math.ceil((rateLimitBackoffUntil - now) / 1000);
        console.log(`速率限制退避中，等待 ${waitTime} 秒...`);
        detectionInterval = requestAnimationFrame(startDetectionLoop);
        return;
    }

    // 如果退避期结束，恢复正常
    if (rateLimitBackoff && now >= rateLimitBackoffUntil) {
        rateLimitBackoff = false;
        consecutive429Errors = 0;
        console.log('速率限制退避结束，恢复正常检测');

        // 逐步恢复帧率（先恢复到较低水平）
        currentTargetFps = Math.min(TARGET_FPS, Math.max(MIN_FPS, currentTargetFps + 1));
        FRAME_INTERVAL = 1000 / currentTargetFps;
        console.log(`帧率恢复到 ${currentTargetFps} FPS`);

        hideRateLimitWarning();
    }

    // 检查是否有请求正在进行（严格单请求模式）
    if (isRequestInProgress) {
        // 有请求在进行，跳过本次，继续等待
        detectionInterval = requestAnimationFrame(startDetectionLoop);
        return;
    }

    // 检查最小请求间隔
    const timeSinceLastRequest = now - lastRequestTime;
    if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
        // 距离上次请求太近，等待
        detectionInterval = requestAnimationFrame(startDetectionLoop);
        return;
    }

    const elapsed = now - lastFrameTime;
    const currentInterval = 1000 / currentTargetFps;

    if (elapsed >= currentInterval) {
        lastFrameTime = now - (elapsed % currentInterval);
        lastRequestTime = now;

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
    if (!videoElement || !canvasElement || !ctx) {
        return;
    }

    // 检查视频是否就绪
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
        return;
    }

    // 严格单请求模式：如果已有请求在进行，直接返回
    if (isRequestInProgress) {
        return;
    }

    // 取消待处理的旧请求
    if (pendingRequest) {
        pendingRequest.abort();
        pendingRequest = null;
    }

    // 标记请求开始
    isRequestInProgress = true;

    try {
        const videoWidth = videoElement.videoWidth;
        const videoHeight = videoElement.videoHeight;

        // 更新显示canvas尺寸（用于绘制检测框）
        if (canvasElement.width !== videoWidth || canvasElement.height !== videoHeight) {
            canvasElement.width = videoWidth;
            canvasElement.height = videoHeight;
        }

        // 更新canvas显示尺寸
        const videoRect = videoElement.getBoundingClientRect();
        canvasElement.style.width = videoRect.width + 'px';
        canvasElement.style.height = videoRect.height + 'px';

        // 绘制当前帧到显示canvas
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        // 保存原始帧数据（用于后续绘制检测框）
        const frameImageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);
        const savedWidth = canvasElement.width;
        const savedHeight = canvasElement.height;

        // 计算压缩后的尺寸（保持宽高比，最长边为MODEL_INPUT_SIZE）
        const scale = Math.min(MODEL_INPUT_SIZE / videoWidth, MODEL_INPUT_SIZE / videoHeight);
        const compressedWidth = Math.round(videoWidth * scale);
        const compressedHeight = Math.round(videoHeight * scale);

        // 创建压缩用的离屏canvas
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = compressedWidth;
        offscreenCanvas.height = compressedHeight;
        const offscreenCtx = offscreenCanvas.getContext('2d');

        // 绘制压缩后的图像
        offscreenCtx.drawImage(videoElement, 0, 0, compressedWidth, compressedHeight);

        // 从压缩canvas获取blob
        offscreenCanvas.toBlob(async function(blob) {
            if (!blob) {
                isRequestInProgress = false;
                return;
            }

            const startTime = performance.now();

            // 准备FormData
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            // 添加压缩信息，让后端知道原始比例
            formData.append('scale', scale.toFixed(4));

            // 获取参数
            const modelSelect = document.getElementById('cameraModelSelect');
            const confidenceInput = document.getElementById('cameraConfidenceInput');
            const iouInput = document.getElementById('cameraIouInput');

            const modelName = modelSelect ? modelSelect.value : 'yolo11n.pt';
            const confidenceValue = confidenceInput ? confidenceInput.value : '0.25';
            const iouValue = iouInput ? iouInput.value : '0.45';

            formData.append('model', modelName);
            formData.append('confidence', confidenceValue);
            formData.append('iou', iouValue);

            console.log(`发送检测请求: model=${modelName}, conf=${confidenceValue}, iou=${iouValue}`);

            // 创建可取消的请求
            const controller = new AbortController();
            pendingRequest = controller;
            const timeoutId = setTimeout(() => controller.abort(), 15000); // 增加超时到15秒

            try {
                // 发送到后端
                const response = await fetch('/api/camera_detect', {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                // 处理429速率限制错误
                if (response.status === 429) {
                    consecutive429Errors++;
                    consecutiveSuccessCount = 0; // 重置成功计数
                    console.warn(`收到429速率限制错误 (${consecutive429Errors}/${MAX_429_ERRORS})`);

                    if (consecutive429Errors >= MAX_429_ERRORS) {
                        // 触发退避机制
                        rateLimitBackoff = true;
                        const backoffSeconds = Math.pow(BACKOFF_MULTIPLIER, Math.min(consecutive429Errors - MAX_429_ERRORS + 1, 4)) * 10;
                        rateLimitBackoffUntil = Date.now() + backoffSeconds * 1000;

                        // 降低帧率
                        const previousFps = currentTargetFps;
                        currentTargetFps = MIN_FPS; // 直接降到最低
                        FRAME_INTERVAL = 1000 / currentTargetFps;

                        console.warn(`触发速率限制退避:`);
                        console.warn(`  - 退避时间: ${backoffSeconds} 秒`);
                        console.warn(`  - 帧率降低: ${previousFps} -> ${currentTargetFps} FPS`);
                        console.warn(`  - 恢复时间: ${new Date(rateLimitBackoffUntil).toLocaleTimeString()}`);

                        // 显示用户提示
                        showRateLimitWarning(backoffSeconds);
                    }
                    return;
                }

                const result = await response.json();

                const endTime = performance.now();
                const inferenceTime = Math.round(endTime - startTime);

                console.log(`检测完成: success=${result.success}, count=${result.count}, time=${inferenceTime}ms`);

                if (result.success) {
                    // 确保canvas尺寸正确
                    if (canvasElement.width !== savedWidth || canvasElement.height !== savedHeight) {
                        canvasElement.width = savedWidth;
                        canvasElement.height = savedHeight;
                    }

                    // 恢复保存的帧图像
                    ctx.putImageData(frameImageData, 0, 0);

                    // 绘制检测框
                    if (result.detections && result.detections.length > 0) {
                        drawDetections(result.detections);
                    }

                    // 更新统计信息
                    updateStatistics(result.count, currentFps, inferenceTime);

                    // 更新检测结果列表
                    updateDetectionTable(result.detections);

                    // 更新性能数据
                    updatePerformanceMetrics(inferenceTime);

                    // 跟踪连续成功次数
                    consecutiveSuccessCount++;

                    // 重置速率限制计数器（成功请求）
                    if (consecutive429Errors > 0) {
                        console.log('检测成功，重置429错误计数');
                        consecutive429Errors = 0;
                    }

                    // 连续成功多次后，逐步恢复帧率
                    if (consecutiveSuccessCount >= SUCCESS_TO_RECOVER && currentTargetFps < TARGET_FPS) {
                        const previousFps = currentTargetFps;
                        currentTargetFps = Math.min(TARGET_FPS, currentTargetFps + 1);
                        FRAME_INTERVAL = 1000 / currentTargetFps;
                        consecutiveSuccessCount = 0; // 重置计数
                        console.log(`帧率恢复: ${previousFps} -> ${currentTargetFps} FPS`);
                    }
                } else {
                    console.error('检测失败:', result.error);
                    consecutiveSuccessCount = 0; // 失败时重置成功计数
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    console.log('请求已取消');
                } else {
                    console.error('发送检测请求失败:', error);
                }
            } finally {
                pendingRequest = null;
                isRequestInProgress = false;
            }
        }, 'image/jpeg', JPEG_QUALITY);

    } catch (error) {
        console.error('捕获帧失败:', error);
        isRequestInProgress = false;
    }
}

/**
 * 更新性能指标
 */
function updatePerformanceMetrics(inferenceTime) {
    inferenceTimes.push(inferenceTime);
    if (inferenceTimes.length > MAX_INFERENCE_SAMPLES) {
        inferenceTimes.shift();
    }

    // 计算平均推理时间
    const avgInferenceTime = inferenceTimes.reduce((a, b) => a + b, 0) / inferenceTimes.length;

    // 自适应调整帧率
    adjustFrameRate(avgInferenceTime);
}

/**
 * 自适应调整帧率（保守策略）
 */
function adjustFrameRate(avgInferenceTime) {
    const targetInferenceTime = 200; // 目标推理时间200ms（更宽松）

    let newFps = currentTargetFps;

    // 推理时间过长（>400ms），立即降低帧率
    if (avgInferenceTime > targetInferenceTime * 2) {
        newFps = Math.max(MIN_FPS, currentTargetFps - 1);
        console.log(`推理时间过长(${avgInferenceTime.toFixed(0)}ms)，降低帧率`);
    }
    // 推理时间很短（<100ms）且样本充足，可以缓慢提升帧率
    else if (avgInferenceTime < targetInferenceTime / 2 &&
             inferenceTimes.length >= MAX_INFERENCE_SAMPLES &&
             currentTargetFps < TARGET_FPS) {
        // 只有在帧率低于目标时才提升
        newFps = Math.min(TARGET_FPS, currentTargetFps + 1);
        console.log(`推理时间稳定(${avgInferenceTime.toFixed(0)}ms)，可提升帧率`);
    }

    if (newFps !== currentTargetFps) {
        const previousFps = currentTargetFps;
        currentTargetFps = newFps;
        FRAME_INTERVAL = 1000 / currentTargetFps;
        console.log(`帧率调整: ${previousFps} -> ${currentTargetFps} FPS`);
    }
}

/**
 * 在canvas上绘制检测结果
 * 注意：检测框坐标是基于canvas像素尺寸的，与发送到后端的图像尺寸一致
 */
function drawDetections(detections) {
    if (!detections || detections.length === 0) {
        return;
    }
    if (!ctx || !canvasElement) {
        return;
    }

    // 定义颜色映射
    const colors = [
        '#FF5733', '#33FF57', '#3357FF', '#FF33F6', '#33FFF6',
        '#F6FF33', '#FF8C33', '#8C33FF', '#FF338C', '#33FF8C'
    ];

    // 计算合适的线条宽度和字体大小
    const lineWidth = Math.max(2, Math.min(4, canvasElement.width / 400));
    const fontSize = Math.max(14, Math.min(20, canvasElement.width / 35));
    const padding = 8;

    detections.forEach((det, index) => {
        const bbox = det.bbox; // [x1, y1, x2, y2] - 已还原到原始尺寸
        const className = det.class;
        const confidence = det.confidence;

        // 选择颜色
        const color = colors[index % colors.length];

        // 计算框的位置和尺寸（坐标已由后端还原到原始图像尺寸）
        const x = Math.max(0, Math.round(bbox[0]));
        const y = Math.max(0, Math.round(bbox[1]));
        const width = Math.round(bbox[2] - bbox[0]);
        const height = Math.round(bbox[3] - bbox[1]);

        // 跳过无效的框
        if (width <= 0 || height <= 0) {
            return;
        }

        // 绘制检测框
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, width, height);

        // 绘制标签
        const label = `${className} ${(confidence * 100).toFixed(0)}%`;
        ctx.font = `bold ${fontSize}px Arial`;
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = fontSize;

        // 计算标签位置（默认在框上方）
        let labelX = x;
        let labelY = y - textHeight - padding;

        // 如果框太靠上，标签放在框内上方
        if (labelY < padding) {
            labelY = y + padding;
        }

        // 确保标签不超出右边界
        if (labelX + textWidth + padding * 2 > canvasElement.width) {
            labelX = canvasElement.width - textWidth - padding * 2;
        }

        // 确保标签不超出左边界
        if (labelX < padding) {
            labelX = padding;
        }

        // 绘制标签背景
        ctx.fillStyle = color;
        ctx.fillRect(labelX - padding/2, labelY - textHeight/2, textWidth + padding, textHeight + 4);

        // 绘制标签文字
        ctx.fillStyle = '#FFFFFF';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, labelX, labelY + 2);
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
        // 禁用截图按钮防止重复点击
        const captureBtn = document.getElementById('captureBtn');
        if (captureBtn) {
            captureBtn.disabled = true;
            const originalText = captureBtn.innerHTML;
            captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>保存中...';
        }

        // 创建临时canvas用于保存截图
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoElement.videoWidth;
        tempCanvas.height = videoElement.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // 绘制视频帧
        tempCtx.drawImage(videoElement, 0, 0);

        // 绘制canvas上的检测框
        tempCtx.drawImage(canvasElement, 0, 0);

        // 添加时间戳水印
        const timestamp = new Date().toLocaleString('zh-CN');
        tempCtx.font = '14px Arial';
        tempCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        tempCtx.fillRect(10, tempCanvas.height - 30, tempCtx.measureText(timestamp).width + 20, 24);
        tempCtx.fillStyle = '#000';
        tempCtx.fillText(timestamp, 20, tempCanvas.height - 14);

        // 转换为图像并下载
        tempCanvas.toBlob(function(blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `detection_${Date.now()}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('success', '截图已保存', 'success');

            // 恢复按钮状态
            if (captureBtn) {
                captureBtn.disabled = false;
                captureBtn.innerHTML = originalText;
            }
        }, 'image/jpeg', 0.95);

    } catch (error) {
        console.error('截图失败:', error);
        showToast('error', '截图失败', 'error');

        // 恢复按钮状态
        const captureBtn = document.getElementById('captureBtn');
        if (captureBtn) {
            captureBtn.disabled = false;
            captureBtn.innerHTML = '<i class="fas fa-camera me-2"></i>拍照截图';
        }
    }
}

/**
 * 尝试重新连接摄像头
 */
async function tryReconnect() {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.error('超过最大重连次数');
        showToast('error', '摄像头连接失败，请刷新页面重试', 'error');
        return;
    }

    reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 5000); // 指数退避

    console.log(`尝试重新连接摄像头 (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})，${delay}ms后重试...`);

    reconnectTimeout = setTimeout(async () => {
        try {
            await startCamera();
            showToast('success', '摄像头已重新连接', 'success');
        } catch (error) {
            console.error('重连失败:', error);
            tryReconnect();
        }
    }, delay);
}

/**
 * 设置网络状态监控
 */
function setupNetworkMonitoring() {
    const networkIndicator = document.getElementById('networkIndicator');
    const networkStatusBadge = document.getElementById('networkStatusBadge');
    const networkStatusText = document.getElementById('networkStatusText');

    if (!networkIndicator) return;

    // 更新网络状态显示
    function updateNetworkStatus() {
        const isOnline = navigator.onLine;
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;

        if (networkIndicator) {
            networkIndicator.style.display = 'block';
        }

        if (networkStatusBadge && networkStatusText) {
            if (isOnline) {
                networkStatusBadge.className = 'badge bg-success';
                networkStatusText.textContent = connection ? connection.effectiveType.toUpperCase() : '在线';

                if (connection) {
                    // 监听网络变化
                    connection.addEventListener('change', updateNetworkStatus);
                }
            } else {
                networkStatusBadge.className = 'badge bg-danger';
                networkStatusText.textContent = '离线';
            }
        }
    }

    // 初始更新
    updateNetworkStatus();

    // 监听在线/离线事件
    window.addEventListener('online', function() {
        updateNetworkStatus();
        showToast('success', '网络已连接', 'success');
    });

    window.addEventListener('offline', function() {
        updateNetworkStatus();
        showToast('warning', '网络已断开', 'warning');
    });
}

/**
 * 更新UI状态
 */
function updateUIState(state) {
    const videoContainer = document.querySelector('.video-container');
    if (!videoContainer) return;

    switch (state) {
        case 'detecting':
            videoContainer.classList.add('detection-active');
            break;
        case 'idle':
            videoContainer.classList.remove('detection-active');
            break;
        case 'error':
            videoContainer.classList.remove('detection-active');
            break;
    }
}

/**
 * 显示速率限制警告
 */
function showRateLimitWarning(backoffSeconds) {
    // 查找或创建警告元素
    let warningEl = document.getElementById('rateLimitWarning');
    if (!warningEl) {
        warningEl = document.createElement('div');
        warningEl.id = 'rateLimitWarning';
        warningEl.style.cssText = `
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideDown 0.3s ease;
        `;
        document.body.appendChild(warningEl);
    }

    warningEl.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>请求过于频繁，暂停检测 <span id="rateLimitCountdown">${backoffSeconds}</span> 秒后恢复...</span>
    `;
    warningEl.style.display = 'flex';

    // 启动倒计时
    let remainingSeconds = backoffSeconds;
    const countdownEl = document.getElementById('rateLimitCountdown');

    if (window.rateLimitCountdownInterval) {
        clearInterval(window.rateLimitCountdownInterval);
    }

    window.rateLimitCountdownInterval = setInterval(() => {
        remainingSeconds--;
        if (countdownEl) {
            countdownEl.textContent = remainingSeconds;
        }
        if (remainingSeconds <= 0) {
            clearInterval(window.rateLimitCountdownInterval);
            hideRateLimitWarning();
        }
    }, 1000);
}

/**
 * 隐藏速率限制警告
 */
function hideRateLimitWarning() {
    const warningEl = document.getElementById('rateLimitWarning');
    if (warningEl) {
        warningEl.style.display = 'none';
    }
    if (window.rateLimitCountdownInterval) {
        clearInterval(window.rateLimitCountdownInterval);
        window.rateLimitCountdownInterval = null;
    }
}

/**
 * 页面卸载时清理资源
 */
window.addEventListener('beforeunload', function() {
    stopCamera();
});
