# YOLO 目标检测 Web 工具

一个基于 Flask + HTML 前端的 YOLO 目标检测推理工具，支持 YOLOv5、YOLOv8 和 YOLOv11 模型。集成配置管理系统，支持局域网部署和多设备同步。  
该工具使用Claude Code + Dify进行优化

## 功能特点

- 🚀 **实时目标检测** - 基于 Ultralytics 框架的快速推理
- 🎨 **美观界面** - 现代化的响应式设计
- 📱 **移动端友好** - 完美适配各种屏幕尺寸
- 🔧 **多模型支持** - 支持 YOLOv5/v8/v11 系列模型
- ⚙️ **可配置参数** - 可调节置信度阈值
- 📊 **详细结果** - 显示检测统计和详细信息
- 🖱️ **拖拽上传** - 支持拖拽文件上传
- 🔄 **实时预览** - 上传前预览图片
- 🌐 **局域网访问** - 支持局域网内多设备访问
- 📋 **配置管理** - 集成配置系统，便于部署和管理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行应用

```bash
# 默认启动（开发模式，局域网可访问）
python run.py

# 生产模式启动
python run.py --production

# 指定端口启动
python run.py --port 8080

# 只显示配置信息，不启动服务
python run.py --config
```

### 3. 访问应用

- **本地访问**：http://127.0.0.1:5000
- **局域网访问**：http://192.168.0.200:5000 （你的实际IP）

## 配置管理系统

### 概述

本项目集成了完整的配置管理系统，所有配置都集中在 `run.py` 中管理，不再依赖外部 `.env` 文件。系统会在首次运行时自动创建配置文件。

### 配置管理命令

```bash
# 显示当前配置
python run.py --manage show

# 显示.env文件内容
python run.py --manage env

# 导出JSON格式配置
python run.py --manage json

# 创建配置模板（用于其他设备）
python run.py --manage template

# 同步配置到.env文件
python run.py --manage sync
```

### 启动选项

```bash
# 基本启动
python run.py

# 开发模式（默认）
python run.py --debug

# 生产模式
python run.py --production

# 指定服务器地址和端口
python run.py --host 0.0.0.0 --port 8080

# 强制重新创建.env文件
python run.py --create-env

# 显示配置信息
python run.py --config
```

### 配置项说明

#### 服务器配置
- **HOST**: 服务器监听地址 (默认: 0.0.0.0，允许局域网访问)
- **PORT**: 服务器端口 (默认: 5000)

#### 文件配置
- **MAX_CONTENT_LENGTH**: 最大文件大小 (默认: 16MB)
- **UPLOAD_FOLDER**: 上传文件存储目录 (默认: static/uploads)
- **OUTPUT_FOLDER**: 处理结果存储目录 (默认: static/outputs)
- **MAX_FILE_AGE**: 文件保存时间 (默认: 3600秒 = 1小时)
- **CLEANUP_INTERVAL**: 清理间隔 (默认: 300秒 = 5分钟)

#### 模型配置
- **DEFAULT_MODEL**: 默认使用的YOLO模型 (默认: yolo11n.pt)
- **DEFAULT_CONFIDENCE**: 默认置信度阈值 (默认: 0.25)
- **DEFAULT_IOU**: 默认IOU阈值 (默认: 0.45)

#### 安全配置
- **SECRET_KEY**: Flask应用密钥 (自动生成32位十六进制)

### 多设备部署

#### 1. 导出当前配置

```bash
# 导出为JSON格式
python run.py --manage json > config.json
```

#### 2. 创建配置模板

```bash
# 创建模板文件
python run.py --manage template
```

生成的 `config_template.env` 文件包含所有配置项，可以复制到其他设备使用。

#### 3. 在其他设备上使用

1. 复制 `config_template.env` 文件到目标设备
2. 重命名为 `.env`
3. 修改其中的IP地址和其他设置
4. 运行 `python run.py`

### 配置示例输出

```
============================================================
🚀 YOLO Web Demo - 目标检测服务
============================================================
📊 运行环境: DevelopmentConfig
🔧 调试模式: 开启

🌐 访问地址:
   本地访问: http://127.0.0.1:5000
   局域网访问: http://192.168.0.200:5000
   服务器监听: 0.0.0.0:5000 (所有网络接口)

⚙️  配置信息:
   📁 上传文件夹: static/uploads
   📁 输出文件夹: static/outputs
   📄 最大文件大小: 16.0MB
   🤖 默认模型: yolo11n.pt
   🎯 默认置信度: 0.25
   📐 默认IOU阈值: 0.45
   📝 日志级别: DEBUG
   ⏱️  文件清理间隔: 300秒
============================================================
```

## 使用说明

### 基本使用

1. **上传图片**：点击"选择文件"或拖拽图片到上传区域
2. **选择模型**：从下拉菜单中选择 YOLO 模型
3. **调整置信度**：使用滑块调整检测置信度阈值
4. **开始检测**：点击"检测目标"按钮
5. **查看结果**：在结果页面查看原始图片和检测结果

### 支持的模型

- **YOLOv11 系列**：nano、small、medium、large、extra-large
- **YOLOv8 系列**：nano、small、medium、large、extra-large
- **YOLOv5 系列**：nano、small、medium、large、extra-large
- **自定义模型**：用户训练的YOLO模型（.pt格式）

### 置信度阈值

- **低阈值 (0.1-0.3)**：检测更多目标，可能包含误检
- **中阈值 (0.3-0.6)**：平衡检测数量和准确性
- **高阈值 (0.6-0.9)**：只检测高置信度目标，减少误检

### 使用自定义模型

1. **放置模型文件**：将您的YOLO模型文件（.pt格式）复制到 `models/` 目录
2. **重启应用**：重启Flask应用以加载新模型
3. **选择模型**：在Web界面中选择您的自定义模型
4. **开始检测**：上传图片进行目标检测

**注意事项**：
- 确保模型文件为有效的PyTorch格式
- 模型应与Ultralytics YOLO框架兼容
- 首次使用自定义模型时可能需要额外加载时间

## API 接口

### 检测接口

```bash
POST /api/detect
```

**参数：**
- `file`：图片文件
- `model`：模型名称（可选，默认：yolo11n.pt）
- `confidence`：置信度阈值（可选，默认：0.25）

**响应：**
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
      "confidence_threshold": 0.25
    },
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [100, 200, 150, 300],
        "bbox_normalized": [0.1, 0.2, 0.15, 0.3],
        "area": 5000
      }
    ],
    "output_image_path": "static/outputs/unique_filename.jpg"
  },
  "original_image": "static/uploads/filename.jpg",
  "output_image": "static/outputs/unique_filename.jpg"
}
```

### 获取可用模型

```bash
GET /api/models
```

**响应：**
```json
{
  "predefined_models": [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov5n.pt",
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt"
  ],
  "custom_models": [],
  "default_model": "yolo11n.pt"
}
```

## 项目结构

```
yolov5-web-demo/
├── run.py                 # 统一运行脚本（包含配置系统）
├── app.py                 # Flask 主应用
├── model_inference.py     # YOLO 推理模块
├── utils.py               # 工具函数
├── requirements.txt       # Python 依赖
├── .env                   # 环境配置文件（自动生成）
├── config_template.env    # 配置模板（多设备部署用）
├── models/               # 自定义模型目录
│   └── README.md         # 自定义模型说明
├── static/               # 静态文件
│   ├── uploads/          # 上传文件目录
│   └── outputs/          # 输出文件目录
└── templates/            # HTML 模板
    ├── base.html         # 基础模板
    ├── index.html        # 首页
    ├── inference.html    # 结果页面
    └── about.html        # 关于页面
```

## 技术栈

- **后端**：Flask、Ultralytics YOLO、OpenCV、PyTorch
- **前端**：HTML5、CSS3、JavaScript、Bootstrap 5、Font Awesome
- **AI框架**：Ultralytics YOLO (支持 v5/v8/v11)
- **配置管理**：集成配置系统，支持环境变量和文件管理

## 开发 vs 生产环境

### 开发环境（默认）
- 调试模式开启
- 详细日志输出
- 自动重载功能
- 启动命令：`python run.py` 或 `python run.py --debug`

### 生产环境
```bash
python run.py --production
```
- 调试模式关闭
- 性能优化
- 需要设置自定义SECRET_KEY

## 安全建议

1. **生产环境**: 设置自定义的SECRET_KEY
2. **文件权限**: 限制上传文件夹的访问权限
3. **网络配置**: 在生产环境中使用防火墙
4. **定期清理**: 系统会自动清理过期文件

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 尝试手动下载模型文件

2. **内存不足**
   - 使用较小的模型（如 nano 或 small）
   - 减小图片尺寸

3. **检测速度慢**
   - 确保使用 GPU（如果可用）
   - 使用较小的模型

4. **文件选择功能失效**
   - 确保安装了 `python-magic-bin`：`pip install python-magic-bin`

5. **局域网无法访问**
   - 检查防火墙设置
   - 确保所有设备在同一网络下

### 性能优化

- 使用 YOLOv11n 或 YOLOv8n 获得最快速度
- 使用 YOLOv11x 或 YOLOv8x 获得最佳精度
- 调整置信度阈值平衡速度和准确性

## 配置文件位置

- **主配置**: `run.py` 中的 `DEFAULT_CONFIG`
- **环境文件**: `.env` (自动生成)
- **模板文件**: `config_template.env` (手动创建)
- **应用配置**: 通过 `get_config()` 函数获取

## 许可证
