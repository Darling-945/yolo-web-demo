# 自定义模型使用说明

## 概述

本目录用于存放用户自定义的YOLO模型文件。支持多种格式，系统会自动检测并加载这些模型。

## 支持的模型格式

### 1. PyTorch模型 (`.pt`)
- Ultralytics YOLOv5/v8/v11 原生格式
- 示例: `my_custom_model.pt`
- 兼容性最好，支持所有功能

### 2. ONNX模型 (`.onnx`)
- 跨平台优化推理格式，CPU性能优异
- 示例: `optimized_model.onnx`
- 适合部署和边缘计算

### 3. TensorRT模型 (`.engine`)
- NVIDIA GPU专用优化格式，推理速度最快
- 示例: `gpu_model.engine`
- 需要NVIDIA GPU和TensorRT环境

## 使用方法

### 1. 模型转换（从PyTorch到ONNX）

使用内置的转换脚本将PyTorch模型(.pt)转换为ONNX格式：

#### 基本转换命令

```bash
# 转换单个模型（使用默认参数）
python convert_models.py --input yolo11n.pt

# 转换并指定输出目录
python convert_models.py --input yolo11n.pt --output-dir models

# 转换后删除原始文件
python convert_models.py --input yolo11n.pt --no-keep-original
```

#### 批量转换

```bash
# 批量转换目录中的所有.pt文件
python convert_models.py --input-dir pytorch_models --output-dir onnx_models

# 批量转换特定模式的文件
python convert_models.py --input-dir models --pattern "yolo*.pt" --output-dir yolo_onnx

# 批量转换并保存转换日志
python convert_models.py --input-dir models --output-dir onnx_models --save-log
```

#### 高级参数调优

```bash
# 高性能转换（用于GPU部署）
python convert_models.py \
    --input models/yolo11n.pt \
    --output-dir models/optimized \
    --input-size 640 \
    --batch-size 1 \
    --opset-version 12 \
    --simplify \
    --dynamic

# 嵌入式设备转换（CPU优化）
python convert_models.py \
    --input models/yolo11n.pt \
    --output-dir models/embedded \
    --input-size 416 \
    --opset-version 11

# 完整转换流水线
python convert_models.py \
    --input-dir models/pytorch \
    --output-dir models/onnx \
    --pattern "*.pt" \
    --save-log
```

#### 转换参数说明

- `--input, -i`: 输入模型文件路径(.pt)
- `--input-dir`: 输入目录路径（批量转换）
- `--output-dir, -o`: 输出目录路径
- `--no-keep-original`: 转换后删除原始模型文件
- `--input-size`: 模型输入尺寸 (默认: 640)
- `--batch-size`: 批处理大小 (默认: 1)
- `--opset-version`: ONNX opset版本 (默认: 12)
- `--no-simplify`: 不简化ONNX模型
- `--dynamic`: 使用动态输入尺寸
- `--pattern`: 批量转换文件模式 (默认: "*.pt")
- `--save-log`: 保存转换历史记录

#### 查看帮助

```bash
python convert_models.py --help
```

#### 在Python代码中使用转换器

```python
# 导入转换器类
from convert_models import ModelConverter

# 创建转换器实例
converter = ModelConverter(
    input_size=640,
    batch_size=1,
    opset_version=12,
    simplify=True,
    dynamic=False
)

# 转换单个模型
result = converter.convert_single_model(
    model_path="models/yolo11n.pt",
    output_dir="models/onnx",
    keep_original=True
)

if result['status'] == 'success':
    print(f"转换成功: {result['output_path']}")
    print(f"文件大小: {result['output_size_mb']} MB")
else:
    print(f"转换失败: {result['error']}")

# 批量转换
batch_results = converter.convert_batch_models(
    input_dir="models/pytorch",
    output_dir="models/onnx",
    pattern="*.pt",
    keep_original=True
)

# 保存转换历史
converter.save_conversion_log("conversion_history.json")
```

#### 验证ONNX模型

```python
import onnx
import onnxruntime as ort

def validate_onnx_model(onnx_path):
    """验证ONNX模型的有效性"""
    try:
        # 加载ONNX模型
        model = onnx.load(onnx_path)

        # 检查模型结构
        onnx.checker.check_model(model)
        print("✅ ONNX模型结构验证通过")

        # 创建推理会话
        session = ort.InferenceSession(onnx_path)

        # 获取输入输出信息
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        print(f"输入名称: {input_info.name}")
        print(f"输入形状: {input_info.shape}")
        print(f"输入类型: {input_info.type}")
        print(f"输出名称: {output_info.name}")
        print(f"输出形状: {output_info.shape}")
        print(f"输出类型: {output_info.type}")

        return True
    except Exception as e:
        print(f"❌ ONNX模型验证失败: {e}")
        return False

# 使用示例
validate_onnx_model("models/yolo11n.onnx")
```

### 2. 放置模型文件

将您的模型文件复制到本目录中：

```
yolo-web-demo/
├── models/
│   ├── README.md
│   ├── my_custom_model.pt        # PyTorch格式
│   ├── optimized_model.onnx      # ONNX格式
│   └── gpu_model.engine          # TensorRT格式
```

### 3. 刷新界面

无需重启应用，直接刷新Web界面即可看到新模型。

### 3. 选择模型

在Web界面中，您将看到按格式组织的模型：
- **YOLOv11 系列 (PyTorch)** - 预定义模型
- **YOLOv8 系列 (PyTorch)** - 预定义模型
- **YOLOv5 系列 (PyTorch)** - 预定义模型
- **自定义 PyTorch 模型** - 您的.pt文件
- **自定义 ONNX 模型** - 您的.onnx文件
- **自定义 TensorRT 模型** - 您的.engine文件

## 模型要求

- **PyTorch格式**：标准的Ultralytics YOLO模型
- **ONNX格式**：从PyTorch转换的ONNX模型
- **TensorRT格式**：从ONNX转换的TensorRT引擎
- **命名**：建议使用有意义的文件名
- **兼容性**：确保与Ultralytics YOLO框架兼容

## 注意事项

1. **格式特定优化**：
   - ONNX模型：自动启用CPU优化
   - TensorRT模型：自动启用GPU优化
   - PyTorch模型：使用默认配置

2. **环境要求**：
   - ONNX：需要 `onnxruntime` 库
   - TensorRT：需要NVIDIA GPU和TensorRT
   - PyTorch：需要 `torch` 和 `ultralytics`

3. **性能考虑**：
   - TensorRT模型推理最快但需要GPU
   - ONNX模型跨平台兼容性好
   - PyTorch模型兼容性最佳

## 故障排除

### 模型无法加载
- 检查模型文件是否为有效的.pt格式
- 确认模型与当前YOLO版本兼容
- 查看应用日志获取详细错误信息

### 检测结果异常
- 验证模型是否针对正确的类别进行训练
- 检查模型的输入尺寸要求
- 调整置信度阈值以获得更好的结果

## 示例

假设您有一个训练好的行人检测模型 `pedestrian_detector.pt`：

1. 将文件复制到 `models/` 目录
2. 重启应用
3. 在Web界面中选择 "pedestrian_detector (自定义)" 模型
4. 上传图片进行检测

## 技术支持

如果您在使用自定义模型时遇到问题，请检查：
- 模型文件完整性
- 系统日志输出
- 模型与框架的兼容性