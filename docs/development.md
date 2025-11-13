# 开发指南

## 开发环境设置

### 1. 环境准备

#### 克隆项目
```bash
git clone https://github.com/your-repo/yolo-web-demo.git
cd yolo-web-demo
```

#### 创建虚拟环境
```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 或使用 conda
conda create -n yolo-demo python=3.9
conda activate yolo-demo
```

#### 安装开发依赖
```bash
pip install -r requirements.txt

# 开发工具
pip install pytest pytest-cov black flake8 mypy pre-commit
```

### 2. 代码格式化和检查

#### 安装 pre-commit
```bash
pre-commit install
```

#### pre-commit 配置
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### 3. 开发服务器

#### 启动开发服务器
```bash
python run.py --debug
```

#### 热重载配置
```python
# 在 app.py 中添加
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

## 代码结构

### 项目目录结构

```
yolo-web-demo/
├── app.py                    # Flask 主应用
├── run.py                    # 应用启动脚本
├── model_inference.py        # YOLO 推理模块
├── convert_models.py         # 模型转换工具
├── requirements.txt          # 依赖列表
├── requirements-dev.txt      # 开发依赖
├── .env                      # 环境配置
├── .gitignore               # Git 忽略文件
├── pyproject.toml           # 项目配置
├── README.md                # 项目说明
├── docker-compose.yml       # Docker 编排
├── Dockerfile              # Docker 镜像
├── tests/                   # 测试目录
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_inference.py
│   └── test_utils.py
├── docs/                    # 文档目录
│   ├── architecture.md
│   ├── api-reference.md
│   ├── deployment.md
│   └── development.md
├── utils/                   # 工具模块
│   ├── __init__.py
│   └── utils.py
├── static/                  # 静态文件
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── outputs/
├── templates/               # HTML 模板
│   ├── base.html
│   ├── index.html
│   ├── inference.html
│   └── about.html
├── models/                  # 自定义模型
└── logs/                    # 日志文件
```

### 核心模块详解

#### 1. Flask 应用 (`app.py`)

```python
from flask import Flask, render_template, request, redirect, url_for
from model_inference import yolo_inference
from utils import secure_file_upload, process_inference_parameters

app = Flask(__name__)

@app.route('/')
def home():
    """首页路由"""
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    """推理接口"""
    # 文件上传处理
    file = request.files.get('file')
    upload_result = secure_file_upload(file, config.UPLOAD_FOLDER)

    # 参数处理
    params = process_inference_parameters(request, config)

    # 执行推理
    result = yolo_inference.detect(
        image_path=upload_result['file_path'],
        output_path=output_path
    )

    return render_template('inference.html', result=result)
```

#### 2. YOLO 推理 (`model_inference.py`)

```python
from ultralytics import YOLO
import cv2
import time

class YOLOInference:
    def __init__(self, model_path: str = 'yolo11n.pt'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """加载模型"""
        self.model = YOLO(self.model_path)

    def detect(self, image_path: str, output_path: str) -> dict:
        """执行检测"""
        # 读取图片
        img = cv2.imread(image_path)

        # 推理计时
        start_time = time.time()
        results = self.model(img)
        inference_time = time.time() - start_time

        # 处理结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = self._process_box(box)
                    detections.append(detection)

        # 保存结果图片
        annotated_img = results[0].plot()
        cv2.imwrite(output_path, annotated_img)

        return {
            'detections': detections,
            'inference_time': inference_time,
            'total_detections': len(detections)
        }

    def _process_box(self, box):
        """处理检测框"""
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()
        class_name = self.model.names[class_id]

        return {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'bbox': bbox.tolist(),
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        }
```

#### 3. 工具函数 (`utils/utils.py`)

```python
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image

def secure_file_upload(file, upload_folder: str) -> dict:
    """安全文件上传"""
    try:
        if not file or file.filename == '':
            return {'success': False, 'error': 'No file provided'}

        filename = secure_filename(file.filename)
        if not filename:
            return {'success': False, 'error': 'Invalid filename'}

        # 验证文件类型
        if not _allowed_file(filename):
            return {'success': False, 'error': 'File type not allowed'}

        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(upload_folder, unique_filename)

        # 保存文件
        file.save(file_path)

        # 验证图片
        if not _validate_image(file_path):
            os.remove(file_path)
            return {'success': False, 'error': 'Invalid image file'}

        return {
            'success': True,
            'filename': unique_filename,
            'file_path': file_path
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

def _allowed_file(filename: str) -> bool:
    """检查文件扩展名"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def _validate_image(file_path: str) -> bool:
    """验证图片文件"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False
```

## API 开发

### 1. RESTful API 设计

#### 版本控制
```python
# API 版本控制
@app.route('/api/v1/detect', methods=['POST'])
def api_v1_detect():
    pass

@app.route('/api/v2/detect', methods=['POST'])
def api_v2_detect():
    pass
```

#### 请求验证
```python
from functools import wraps
from flask import request, jsonify

def validate_file_upload(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/detect', methods=['POST'])
@validate_file_upload
def api_detect():
    # 检测逻辑
    pass
```

### 2. 响应格式标准化

```python
def api_response(success=True, data=None, error=None, status_code=200):
    """标准化 API 响应"""
    response = {'success': success}

    if success:
        response['data'] = data
    else:
        response['error'] = error

    return jsonify(response), status_code

# 使用示例
@app.route('/api/models')
def api_models():
    try:
        models = get_available_models()
        return api_response(success=True, data=models)
    except Exception as e:
        return api_response(success=False, error=str(e), status_code=500)
```

## 前端开发

### 1. 静态资源管理

#### CSS 结构
```css
/* static/css/style.css */
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --success-color: #28a745;
    --danger-color: #dc3545;
    --warning-color: #ffc107;
    --info-color: #17a2b8;
}

/* 基础样式 */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

/* 上传区域 */
.upload-area {
    border: 2px dashed #dee2e6;
    border-radius: 0.375rem;
    padding: 3rem 1rem;
    text-align: center;
    transition: border-color 0.15s ease-in-out;
}

.upload-area.dragover {
    border-color: var(--primary-color);
    background-color: rgba(0, 123, 255, 0.1);
}

/* 结果展示 */
.detection-result {
    position: relative;
    display: inline-block;
}

.detection-box {
    position: absolute;
    border: 2px solid #00ff00;
    background-color: rgba(0, 255, 0, 0.1);
}

.detection-label {
    position: absolute;
    top: -25px;
    left: 0;
    background-color: #00ff00;
    color: #000;
    padding: 2px 6px;
    font-size: 12px;
    border-radius: 3px;
}
```

#### JavaScript 模块
```javascript
// static/js/main.js
class YOLOWebDemo {
    constructor() {
        this.initElements();
        this.bindEvents();
    }

    initElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.submitBtn = document.getElementById('submitBtn');
        this.resultsContainer = document.getElementById('resultsContainer');
    }

    bindEvents() {
        // 拖拽事件
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));

        // 文件选择事件
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // 表单提交事件
        this.submitBtn.addEventListener('click', this.handleSubmit.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.fileInput.files = files;
            this.previewImage(files[0]);
        }
    }

    async handleSubmit() {
        const formData = new FormData();
        formData.append('file', this.fileInput.files[0]);
        formData.append('model', document.getElementById('modelSelect').value);
        formData.append('confidence', document.getElementById('confidenceSlider').value);

        try {
            this.showLoading();
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.displayResults(result);
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        if (result.success) {
            this.renderDetections(result.data);
        } else {
            this.showError(result.error);
        }
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new YOLOWebDemo();
});
```

### 2. 模板系统

#### 基础模板
```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}YOLO Web Demo{% endblock %}</title>

    <!-- CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">

    <!-- JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block head %}{% endblock %}
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('home') }}">YOLO Web Demo</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('home') }}">首页</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('about') }}">关于</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- 主要内容 -->
    <main class="container my-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </main>

    <!-- 页脚 -->
    <footer class="bg-light text-center py-3 mt-5">
        <div class="container">
            <p class="mb-0">&copy; 2023 YOLO Web Demo. All rights reserved.</p>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

## 测试

### 1. 单元测试

#### 测试配置
```python
# conftest.py
import pytest
import tempfile
import os
from app import app
from run import get_config

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture
def sample_image():
    """创建测试用图片"""
    from PIL import Image
    import io

    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes
```

#### API 测试
```python
# tests/test_api.py
import json
import pytest

class TestAPI:
    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_get_models(self, client):
        """测试获取模型列表"""
        response = client.get('/api/models')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predefined_models' in data
        assert 'yolo11n.pt' in data['predefined_models']['pytorch']

    def test_detect_api(self, client, sample_image):
        """测试检测接口"""
        response = client.post('/api/detect',
                             data={'file': (sample_image, 'test.jpg'),
                                   'model': 'yolo11n.pt'},
                             content_type='multipart/form-data')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'result' in data

    def test_detect_no_file(self, client):
        """测试无文件上传"""
        response = client.post('/api/detect')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data

    def test_detect_invalid_file(self, client):
        """测试无效文件"""
        response = client.post('/api/detect',
                             data={'file': (b'invalid', 'test.txt')},
                             content_type='multipart/form-data')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
```

#### 工具函数测试
```python
# tests/test_utils.py
import pytest
import tempfile
import os
from utils.utils import secure_file_upload, is_valid_image_file
from PIL import Image

class TestUtils:
    def test_secure_file_upload_success(self, sample_image):
        """测试成功上传"""
        with tempfile.TemporaryDirectory() as temp_dir:
            from werkzeug.datastructures import FileStorage

            file = FileStorage(
                stream=sample_image,
                filename='test.jpg',
                content_type='image/jpeg'
            )

            result = secure_file_upload(file, temp_dir)

            assert result['success'] is True
            assert 'filename' in result
            assert 'file_path' in result
            assert os.path.exists(result['file_path'])

    def test_secure_file_upload_no_file(self):
        """测试无文件"""
        result = secure_file_upload(None, '/tmp')
        assert result['success'] is False
        assert 'No file provided' in result['error']

    def test_is_valid_image_file_valid(self):
        """测试有效图片"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp.name, 'JPEG')

            assert is_valid_image_file(tmp.name) is True
            os.unlink(tmp.name)

    def test_is_valid_image_file_invalid(self):
        """测试无效文件"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'invalid image data')
            tmp.flush()

            assert is_valid_image_file(tmp.name) is False
            os.unlink(tmp.name)
```

### 2. 集成测试

```python
# tests/test_integration.py
import pytest
import requests
import json
import time

class TestIntegration:
    @pytest.fixture(scope="class")
    def api_url(self):
        return "http://localhost:5000/api"

    def test_full_workflow(self, api_url):
        """测试完整工作流程"""
        # 1. 获取模型列表
        response = requests.get(f"{api_url}/models")
        assert response.status_code == 200
        models = response.json()
        assert models['success'] is True

        # 2. 上传并检测图片
        with open('tests/fixtures/test_image.jpg', 'rb') as f:
            files = {'file': f}
            data = {
                'model': 'yolo11n.pt',
                'confidence': 0.25
            }

            response = requests.post(f"{api_url}/detect", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'detections' in result['result']

        # 3. 验证输出文件
        if 'output_image' in result:
            output_response = requests.get(f"http://localhost:5000/{result['output_image']}")
            assert output_response.status_code == 200
```

### 3. 性能测试

```python
# tests/test_performance.py
import pytest
import time
import requests
import concurrent.futures

class TestPerformance:
    def test_response_time(self):
        """测试响应时间"""
        start_time = time.time()

        response = requests.get("http://localhost:5000/api/models")

        response_time = time.time() - start_time
        assert response.status_code == 200
        assert response_time < 1.0  # 响应时间应小于1秒

    def test_concurrent_requests(self):
        """测试并发请求"""
        url = "http://localhost:5000/api/health"

        def make_request():
            response = requests.get(url)
            return response.status_code == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in futures]

        # 所有请求都应该成功
        assert all(results) is True

    @pytest.mark.slow
    def test_inference_performance(self):
        """测试推理性能"""
        with open('tests/fixtures/test_image.jpg', 'rb') as f:
            files = {'file': f}
            data = {'model': 'yolo11n.pt'}

            start_time = time.time()
            response = requests.post(
                "http://localhost:5000/api/detect",
                files=files,
                data=data
            )
            inference_time = time.time() - start_time

        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True

        # 推理时间应该合理（取决于硬件）
        assert inference_time < 10.0  # 应该在10秒内完成
```

### 4. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_api.py

# 运行特定测试类
pytest tests/test_api.py::TestAPI

# 运行特定测试方法
pytest tests/test_api.py::TestAPI::test_health_check

# 生成覆盖率报告
pytest --cov=. --cov-report=html

# 运行性能测试
pytest -m slow
```

## 调试

### 1. 日志配置

```python
import logging
import os

# 开发环境日志配置
def setup_dev_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug.log')
        ]
    )

# 设置详细日志
logging.getLogger('ultralytics').setLevel(logging.DEBUG)
```

### 2. 调试工具

#### Flask 调试器
```python
# 开启调试模式
app.run(debug=True, use_debugger=True, use_reloader=True)
```

#### Python 调试器
```python
import pdb; pdb.set_trace()

# 或使用更现代的调试器
import ipdb; ipdb.set_trace()
```

#### 性能分析
```python
import cProfile
import pstats

def profile_inference():
    profiler = cProfile.Profile()
    profiler.enable()

    # 执行推理代码
    result = yolo_inference.detect('test.jpg', 'output.jpg')

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

## 贡献指南

### 1. 代码贡献流程

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建 Pull Request

### 2. 代码规范

#### Python 代码规范
- 使用 Black 进行代码格式化
- 使用 Flake8 进行代码检查
- 使用 MyPy 进行类型检查
- 遵循 PEP 8 编码规范

#### 提交信息规范
```
type(scope): description

[optional body]

[optional footer]
```

类型说明：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建或辅助工具的变动

### 3. 发布流程

1. 更新版本号
2. 更新 CHANGELOG.md
3. 创建 Git tag
4. 构建和发布 Docker 镜像
5. 发布到 PyPI（如适用）

```bash
# 版本更新
git tag v1.0.0
git push origin v1.0.0

# Docker 构建
docker build -t yolo-web-demo:v1.0.0 .
docker push your-registry/yolo-web-demo:v1.0.0
```