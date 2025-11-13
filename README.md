# YOLO 目标检测 Web 工具

🚀 一个基于 Flask 的现代化 YOLO 目标检测 Web 应用，支持 YOLOv5/v8/v11 系列模型，提供直观的 Web 界面和强大的 API。

![YOLO Web Demo](https://img.shields.io/badge/YOLO-v5%2Fv8%2F11-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

## ✨ 功能特性

### 🎯 智能检测
- **多模型支持**: YOLOv5、YOLOv8、YOLOv11 全系列模型
- **多格式兼容**: PyTorch(.pt)、ONNX(.onnx)、TensorRT(.engine)
- **实时推理**: 基于 Ultralytics 框架的高性能推理
- **灵活配置**: 可调节置信度阈值和 IOU 阈值

### 🌐 Web 界面
- **现代化设计**: 响应式布局，支持桌面和移动设备
- **拖拽上传**: 支持拖拽文件上传，实时图片预览
- **实时预览**: 上传前预览图片，确认后开始检测
- **结果展示**: 直观的检测结果可视化，包含检测框和标签

### 🔧 开发友好
- **RESTful API**: 完整的 API 接口，支持第三方集成
- **模块化设计**: 清晰的代码结构，易于扩展和维护
- **配置管理**: 灵活的配置系统，支持开发和生产环境
- **自动部署**: 支持 Docker 和传统部署方式

### 🛡️ 安全可靠
- **文件验证**: 多层级文件安全验证，防止恶意文件上传
- **速率限制**: API 访问频率控制，防止滥用
- **自动清理**: 智能文件生命周期管理，防止磁盘空间溢出
- **错误处理**: 完善的错误处理机制，提供友好的错误信息

## 🚀 快速开始

### 📋 环境要求
- Python 3.8+
- 4GB+ 内存
- 2GB+ 磁盘空间

### 🛠️ 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-repo/yolo-web-demo.git
cd yolo-web-demo
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动服务**
```bash
python run.py
```

4. **访问应用**
- 本地访问: http://localhost:5000
- 局域网访问: http://your-ip:5000

### 🐳 Docker 部署

```bash
# 快速启动
docker-compose up -d

# 或者使用单个容器
docker run -p 5000:5000 yolo-web-demo
```

## 🎮 使用指南

### Web 界面使用

1. **上传图片**
   - 点击"选择文件"按钮
   - 或直接拖拽图片到上传区域
   - 支持 JPG、PNG、WEBP、GIF 等格式

2. **选择模型**
   - 从下拉菜单选择合适的模型
   - Nano 模型速度最快，Extra-Large 模型精度最高

3. **调整参数**
   - **置信度阈值**: 0.1-0.9，控制检测的严格程度
   - **IOU 阈值**: 0.1-0.9，控制重叠框的处理

4. **开始检测**
   - 点击"检测目标"按钮
   - 等待处理完成，查看结果

### API 使用示例

#### Python SDK
```python
import requests

# 单张图片检测
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'model': 'yolo11n.pt',
        'confidence': 0.3
    }
    response = requests.post(
        'http://localhost:5000/api/detect',
        files=files,
        data=data
    )

result = response.json()
print(f"检测到 {result['result']['summary']['total_detections']} 个对象")
```

#### JavaScript SDK
```javascript
// 使用 Fetch API
async function detectObjects(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', 'yolo11n.pt');
    formData.append('confidence', '0.3');

    const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    return result;
}

// 使用示例
document.getElementById('upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const result = await detectObjects(file);
    console.log('检测结果:', result);
});
```

#### cURL 示例
```bash
# 基本检测
curl -X POST \
  http://localhost:5000/api/detect \
  -F "file=@image.jpg" \
  -F "model=yolo11n.pt" \
  -F "confidence=0.25"

# 获取可用模型
curl http://localhost:5000/api/models

# 健康检查
curl http://localhost:5000/api/health
```

## 🔧 配置选项

### 基础配置
```bash
# 显示当前配置
python run.py --manage show

# 开发模式
python run.py --debug

# 生产模式
python run.py --production

# 指定端口
python run.py --port 8080
```

### 环境变量配置
```bash
# .env 文件示例
FLASK_ENV=production
SECRET_KEY=your-super-secret-key
HOST=0.0.0.0
PORT=5000
MAX_CONTENT_LENGTH=50485760
DEFAULT_MODEL=yolo11n.pt
DEFAULT_CONFIDENCE=0.25
DEFAULT_IOU=0.45
LOG_LEVEL=INFO
```

## 🎨 模型支持

### 预定义模型
| 模型系列 | Nano | Small | Medium | Large | Extra-Large |
|---------|------|-------|--------|-------|-------------|
| **YOLOv11** | ⚡ 最快 | 🚀 快速 | ⭐ 平衡 | 💪 强大 | 🔥 最强 |
| **YOLOv8**  | ⚡ 最快 | 🚀 快速 | ⭐ 平衡 | 💪 强大 | 🔥 最强 |
| **YOLOv5**  | ⚡ 最快 | 🚀 快速 | ⭐ 平衡 | 💪 强大 | 🔥 最强 |

### 模型特点
- **Nano (n)**: 速度最快，适合实时应用
- **Small (s)**: 速度和精度平衡
- **Medium (m)**: 适合一般用途
- **Large (l)**: 高精度，适合离线处理
- **Extra-Large (x)**: 最高精度，处理速度较慢

### 自定义模型
支持添加自定义模型：
1. 将 `.pt`、`.onnx` 或 `.engine` 文件放入 `models/` 目录
2. 重启应用即可在模型列表中看到自定义模型

## 🔄 模型转换

内置 PyTorch 到 ONNX 的模型转换工具：

```bash
# 转换单个模型
python convert_models.py --input yolo11n.pt --output-dir ./onnx_models

# 自定义输入尺寸
python convert_models.py --input custom_model.pt --input-size 1024

# 禁用模型简化
python convert_models.py --input yolo11n.pt --no-simplify
```

转换优势：
- **ONNX 格式**: CPU 推理性能优化
- **跨平台兼容**: 支持多种操作系统
- **部署友好**: 适合生产环境部署

## 📊 API 参考

### 核心接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/detect` | POST | 图片目标检测 |
| `/api/models` | GET | 获取可用模型列表 |
| `/api/health` | GET | 系统健康检查 |
| `/api/files` | GET | 获取文件列表 |

### 检测接口参数
- `file` (required): 图片文件
- `model` (optional): 模型名称，默认 `yolo11n.pt`
- `confidence` (optional): 置信度阈值，0.0-1.0，默认 0.25
- `iou` (optional): IOU 阈值，0.0-1.0，默认 0.45

### 响应格式
```json
{
  "success": true,
  "result": {
    "summary": {
      "total_detections": 3,
      "detection_summary": {
        "person": 2,
        "car": 1
      },
      "model_used": "yolo11n.pt",
      "confidence_threshold": 0.25,
      "inference_time": 0.125
    },
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [100, 200, 150, 300]
      }
    ]
  }
}
```

## 🏗️ 项目架构

```
yolo-web-demo/
├── 📄 app.py                 # Flask 主应用
├── 🚀 run.py                 # 启动脚本
├── 🤖 model_inference.py     # YOLO 推理引擎
├── 🔄 convert_models.py      # 模型转换工具
├── 🛠️ utils.py               # 工具函数
├── 📋 requirements.txt       # 依赖列表
├── 📁 models/               # 自定义模型目录
├── 📁 static/               # 静态资源
│   ├── css/                 # 样式文件
│   ├── js/                  # JavaScript 文件
│   ├── uploads/             # 上传文件
│   └── outputs/             # 输出文件
├── 📁 templates/            # HTML 模板
├── 📁 docs/                 # 详细文档
│   ├── architecture.md      # 系统架构
│   ├── api-reference.md     # API 参考
│   ├── deployment.md        # 部署指南
│   └── development.md       # 开发指南
└── 📁 tests/                # 测试文件
```

## 🐳 Docker 部署

### 快速部署
```bash
# 使用 docker-compose
git clone https://github.com/your-repo/yolo-web-demo.git
cd yolo-web-demo
docker-compose up -d
```

### 单容器部署
```bash
# 构建镜像
docker build -t yolo-web-demo .

# 运行容器
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/uploads:/app/static/uploads \
  yolo-web-demo
```

### 环境变量配置
```yaml
version: '3.8'
services:
  web:
    image: yolo-web-demo:latest
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-secret-key
      - DEFAULT_MODEL=yolo11n.pt
    volumes:
      - ./models:/app/models
      - ./uploads:/app/static/uploads
```

## 🚀 性能优化

### 推理性能
- **GPU 加速**: 自动检测并使用 GPU（如果可用）
- **模型缓存**: 避免重复加载模型
- **批量处理**: 支持批量图片检测

### 系统性能
- **异步处理**: 文件清理和日志记录异步执行
- **内存管理**: 智能内存释放，避免内存泄漏
- **连接池**: 数据库连接复用（如适用）

### 建议配置
- **开发环境**: YOLOv11n，CPU 推理
- **生产环境**: YOLOv11s，GPU 推理
- **高精度需求**: YOLOv11x，GPU 推理

## 🔒 安全特性

### 文件安全
- **类型验证**: 严格的文件类型检查
- **大小限制**: 可配置的文件大小限制
- **路径安全**: 防止路径遍历攻击
- **病毒扫描**: 可选的文件病毒扫描

### API 安全
- **速率限制**: 防止 API 滥用
- **输入验证**: 严格的参数验证
- **CORS 支持**: 跨域资源共享配置
- **HTTPS 支持**: SSL/TLS 加密传输

### 运行时安全
- **最小权限**: 应用运行在最小权限用户下
- **沙箱隔离**: Docker 容器化部署
- **日志监控**: 完整的安全事件日志
- **自动更新**: 安全补丁自动更新

## 🌍 部署选项

### 云服务部署
- **AWS**: 使用 EC2 + Elastic Beanstalk
- **Google Cloud**: 使用 Cloud Run + GCS
- **Azure**: 使用 App Service + Blob Storage
- **阿里云**: 使用 ECS + OSS

### 本地部署
- **传统服务器**: 使用 Nginx + Gunicorn
- **容器化**: 使用 Docker + Docker Compose
- **Kubernetes**: 使用 K8s 集群部署

### 边缘设备
- **Jetson 系列**: NVIDIA Jetson Nano/Xavier
- **树莓派**: Raspberry Pi 4+ (使用轻量模型)
- **工业 PC**: 支持工业环境部署

## 🤝 贡献指南

### 如何贡献
1. Fork 项目仓库
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建 Pull Request

### 开发环境
```bash
# 克隆仓库
git clone https://github.com/your-repo/yolo-web-demo.git
cd yolo-web-demo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 运行测试
pytest

# 代码格式化
black .
flake8 .
```

### 代码规范
- 遵循 PEP 8 编码规范
- 使用 Black 进行代码格式化
- 编写单元测试
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 模型实现
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [Bootstrap](https://getbootstrap.com/) - UI 框架
- 所有贡献者和用户的支持

## 📞 支持与反馈

- 🐛 [报告 Bug](https://github.com/your-repo/yolo-web-demo/issues)
- 💡 [功能建议](https://github.com/your-repo/yolo-web-demo/issues)
- 📧 [邮件联系](mailto:support@example.com)
- 💬 [讨论区](https://github.com/your-repo/yolo-web-demo/discussions)

---

⭐ 如果这个项目对您有帮助，请给我们一个 Star！