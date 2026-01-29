# YOLO 目标检测 Web 工具

一个基于 Flask 的 YOLO 目标检测 Web 应用，支持 YOLOv5/v8/v11 系列模型。

![YOLO](https://img.shields.io/badge/YOLO-v5%2Fv8%2F11-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)

## 功能特性

- **多模型支持**: YOLOv5、YOLOv8、YOLOv11 全系列
- **多种格式**: PyTorch(.pt)、ONNX(.onnx)、TensorRT(.engine)
- **图片检测**: 单张/批量图片检测
- **视频检测**: 支持视频目标检测
- **Web 界面**: 现代化响应式设计
- **RESTful API**: 完整的 API 接口

## 快速开始

### 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
# 启动服务
python run.py

# 访问
http://localhost:5000
```

## 使用方法

### Web 界面

1. 上传图片或视频
2. 选择模型（可选）
3. 调整参数（可选）
4. 点击检测

### API 示例

```python
import requests

# 检测图片
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/detect',
        files={'file': f}
    )
result = response.json()
```

```bash
# cURL 示例
curl -X POST http://localhost:5000/api/detect \
  -F "file=@image.jpg"
```

## 模型支持

### 预定义模型

| 系列 | Nano | Small | Medium | Large | X-Large |
|------|------|-------|--------|-------|---------|
| YOLOv11 | yolo11n.pt | yolo11s.pt | yolo11m.pt | yolo11l.pt | yolo11x.pt |
| YOLOv8  | yolo8n.pt  | yolo8s.pt  | yolo8m.pt  | yolo8l.pt  | yolo8x.pt  |
| YOLOv5  | yolo5n.pt  | yolo5s.pt  | yolo5m.pt  | yolo5l.pt  | yolo5x.pt  |

### 自定义模型

将模型文件放入 `models/` 目录，重启应用即可使用。

## 配置

### 命令行参数

```bash
# 开发模式
python run.py --debug

# 生产模式
python run.py --production

# 指定端口
python run.py --port 8080

# 查看配置
python run.py --manage show
```

### 环境变量

在 `.env` 文件中配置：

```ini
HOST=0.0.0.0
PORT=5000
DEFAULT_MODEL=yolo11n.pt
DEFAULT_CONFIDENCE=0.25
DEFAULT_IOU=0.45
MAX_CONTENT_LENGTH=104857600
```

## API 参考

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/detect` | POST | 图片/视频检测 |
| `/api/models` | GET | 获取模型列表 |
| `/api/batch_upload` | POST | 批量上传 |

## 项目结构

```
yolo-web-demo/
├── app.py                  # Flask 主应用
├── run.py                  # 启动脚本
├── model_inference.py      # YOLO 推理引擎
├── utils/                  # 工具函数
├── models/                 # 自定义模型目录
├── static/                 # 静态资源
├── templates/              # HTML 模板
└── requirements.txt        # 依赖列表
```

## Docker 部署

```bash
# 构建并运行
docker-compose up -d

# 或单独构建
docker build -t yolo-web-demo .
docker run -p 5000:5000 yolo-web-demo
```

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 模型实现
- [Flask](https://flask.palletsprojects.com/) - Web 框架
