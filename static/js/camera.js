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

// 请求控制
let pendingRequest = null;
let requestQueue = [];
const MAX_QUEUE_SIZE = 2; // 最大队列长度，避免堆积

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

    // 清空请求队列
    requestQueue = [];

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

    // 检查队列大小，避免堆积
    if (requestQueue.length >= MAX_QUEUE_SIZE) {
        console.log('请求队列已满，跳过此帧');
        return;
    }

    // 取消待处理的旧请求
    if (pendingRequest) {
        pendingRequest.abort();
        pendingRequest = null;
    }

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

            // 创建可取消的请求
            const controller = new AbortController();
            pendingRequest = controller;
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10秒超时

            try {
                // 添加到队列
                const requestId = Date.now();
                requestQueue.push(requestId);

                // 发送到后端
                const response = await fetch('/api/camera_detect', {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                // 从队列移除
                const queueIndex = requestQueue.indexOf(requestId);
                if (queueIndex > -1) {
                    requestQueue.splice(queueIndex, 1);
                }

                // 检查是否是最新的请求
                if (requestQueue.length > 0 || requestId < Math.max(...requestQueue, requestId)) {
                    console.log('忽略过期响应');
                    return;
                }

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

                    // 更新性能数据
                    updatePerformanceMetrics(inferenceTime);
                } else {
                    console.error('检测失败:', result.error);
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    console.log('请求已取消');
                } else {
                    console.error('发送检测请求失败:', error);
                }
            } finally {
                pendingRequest = null;
            }
        }, 'image/jpeg', 0.8); // 0.8质量，平衡性能和质量

    } catch (error) {
        console.error('捕获帧失败:', error);
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
 * 自适应调整帧率
 */
function adjustFrameRate(avgInferenceTime) {
    const targetInferenceTime = 100; // 目标推理时间100ms
    const minFps = 5;
    const maxFps = 15;

    let newFps = TARGET_FPS;

    if (avgInferenceTime > targetInferenceTime * 2) {
        // 推理时间过长，降低帧率
        newFps = Math.max(minFps, TARGET_FPS - 3);
    } else if (avgInferenceTime < targetInferenceTime / 2 && inferenceTimes.length >= MAX_INFERENCE_SAMPLES) {
        // 推理时间较短，可以提高帧率
        newFps = Math.min(maxFps, TARGET_FPS + 2);
    }

    if (newFps !== TARGET_FPS) {
        console.log(`自适应调整帧率: ${TARGET_FPS} -> ${newFps} FPS (平均推理时间: ${avgInferenceTime.toFixed(0)}ms)`);
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
 * 页面卸载时清理资源
 */
window.addEventListener('beforeunload', function() {
    stopCamera();
});
