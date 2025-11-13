# YOLO 目标检测 Web 工具

基于 Flask 的 YOLO 目标检测推理工具，支持 YOLOv5/v8/v11 模型。

## 功能特点

- 🚀 实时目标检测 - 基于 Ultralytics 框架
- 🔧 多模型支持 - 支持 YOLOv5/v8/v11 系列模型
- 🔄 多格式支持 - 支持 PyTorch(.pt)、ONNX(.onnx)、TensorRT(.engine)
- ⚙️ 可配置参数 - 可调节置信度和IOU阈值
- 📱 移动端友好 - 响应式设计
- 🖱️ 拖拽上传 - 支持拖拽文件上传
- 🌐 局域网访问 - 支持多设备访问

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行应用

```bash
python run.py
```

### 3. 访问应用

- 本地访问：http://127.0.0.1:5000
- 局域网访问：http://192.168.0.200:5000 （你的实际IP）

## 配置系统

配置集中在 `run.py` 中管理，首次运行时自动创建 `.env` 文件。

```bash
# 显示当前配置
python run.py --manage show
```

## 使用说明

1. **上传图片**：点击"选择文件"或拖拽图片到上传区域
2. **选择模型**：从下拉菜单中选择 YOLO 模型
3. **调整置信度**：使用滑块调整检测置信度阈值
4. **开始检测**：点击"检测目标"按钮
5. **查看结果**：在结果页面查看原始图片和检测结果

## 模型转换

```bash
python convert_models.py --input yolo11n.pt --output-dir ./onnx_models
```

## 支持的模型

- **预定义模型**：YOLOv11/v8/v5 系列 (nano、small、medium、large、extra-large)
- **自定义模型**：PyTorch(.pt)、ONNX(.onnx)、TensorRT(.engine) 格式

## 置信度阈值

- **低阈值 (0.1-0.3)**：检测更多目标，可能包含误检
- **中阈值 (0.3-0.6)**：平衡检测数量和准确性
- **高阈值 (0.6-0.9)**：只检测高置信度目标，减少误检

## API 接口

### 检测接口

```bash
POST /api/detect
```

参数：
- `file`：图片文件
- `model`：模型名称（可选）
- `confidence`：置信度阈值（可选）

### 获取可用模型

```bash
GET /api/models
```

## 项目结构

```
yolo-web-demo/
├── run.py                 # 运行脚本
├── app.py                 # Flask 主应用
├── model_inference.py     # YOLO 推理模块
├── convert_models.py      # 模型转换工具
├── utils.py               # 工具函数
├── requirements.txt       # Python 依赖
├── .env                   # 配置文件（自动生成）
├── models/               # 自定义模型目录
├── static/               # 静态文件
└── templates/            # HTML 模板
```

## 技术栈

- **后端**：Flask、Ultralytics YOLO、OpenCV、PyTorch
- **前端**：HTML5、CSS3、JavaScript、Bootstrap 5
- **AI框架**：Ultralytics YOLO (支持 v5/v8/v11)

## 安全建议

1. 生产环境设置自定义SECRET_KEY
2. 限制上传文件夹的访问权限
3. 使用防火墙保护服务器

## 故障排除

1. **模型下载失败**：检查网络连接
2. **内存不足**：使用较小的模型（nano 或 small）
3. **检测速度慢**：使用 GPU 或较小模型
4. **局域网无法访问**：检查防火墙设置

## 性能优化

- 使用 YOLOv11n 获得最快速度
- 使用 YOLOv11x 获得最佳精度
- 调整置信度阈值平衡速度和准确性
