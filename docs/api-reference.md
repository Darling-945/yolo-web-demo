# API 参考文档

## 概览

YOLO Web Demo 提供 RESTful API 接口，支持目标检测、模型管理和系统配置。

**基础URL**: `http://localhost:5000`

## 认证

当前版本暂不需要认证，但在生产环境中建议配置：
- API密钥认证
- JWT令牌认证
- IP白名单限制

## 错误处理

### 标准错误格式

```json
{
  "success": false,
  "error": "错误描述",
  "error_code": "ERROR_CODE",
  "timestamp": "2023-12-07T10:30:00Z"
}
```

### HTTP状态码

- `200` - 成功
- `400` - 请求参数错误
- `401` - 未授权（需要认证）
- `403` - 禁止访问
- `413` - 文件过大
- `429` - 请求过于频繁
- `500` - 服务器内部错误

## 1. 目标检测接口

### POST /api/detect

执行图片目标检测

**请求**:
- **方法**: POST
- **Content-Type**: multipart/form-data
- **Body**:
  - `file` (required): 图片文件
  - `model` (optional): 模型名称，默认为 `yolo11n.pt`
  - `confidence` (optional): 置信度阈值，范围 0.0-1.0，默认 0.25
  - `iou` (optional): IOU阈值，范围 0.0-1.0，默认 0.45

**示例请求**:
```bash
curl -X POST \
  http://localhost:5000/api/detect \
  -F "file=@example.jpg" \
  -F "model=yolo11n.pt" \
  -F "confidence=0.3"
```

**响应**:
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
      "confidence_threshold": 0.3,
      "iou_threshold": 0.45,
      "image_dimensions": {
        "width": 640,
        "height": 480
      },
      "inference_time": 0.125,
      "model_format": "pytorch"
    },
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [100, 200, 150, 300],
        "bbox_normalized": [0.156, 0.417, 0.234, 0.625],
        "area": 5000
      },
      {
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.72,
        "bbox": [300, 150, 450, 280],
        "bbox_normalized": [0.469, 0.313, 0.703, 0.583],
        "area": 19500
      }
    ],
    "output_image_path": "static/outputs/abc123_example.jpg"
  },
  "original_image": "static/uploads/abc123_example.jpg",
  "output_image": "static/outputs/abc123_example.jpg"
}
```

## 2. 模型管理接口

### GET /api/models

获取可用模型列表

**请求**:
- **方法**: GET
- **参数**: 无

**示例请求**:
```bash
curl http://localhost:5000/api/models
```

**响应**:
```json
{
  "success": true,
  "predefined_models": {
    "pytorch": [
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
    ]
  },
  "custom_models": {
    "pytorch": ["custom_model.pt"],
    "onnx": ["custom_model.onnx"],
    "tensorrt": ["custom_model.engine"]
  },
  "default_model": "yolo11n.pt",
  "max_file_size": 16777216,
  "allowed_extensions": ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff", "tif"],
  "supported_formats": ["pytorch", "onnx", "tensorrt"]
}
```

## 3. 系统状态接口

### GET /api/health

系统健康检查

**请求**:
- **方法**: GET
- **参数**: 无

**响应**:
```json
{
  "success": true,
  "status": "healthy",
  "timestamp": "2023-12-07T10:30:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "memory_usage": {
    "total": "2.5GB",
    "used": "1.2GB",
    "percent": 48
  },
  "disk_usage": {
    "total": "50GB",
    "used": "10GB",
    "percent": 20
  }
}
```

## 4. 配置管理接口

### GET /api/config

获取当前系统配置（管理员权限）

**请求**:
- **方法**: GET
- **Headers**:
  - `Authorization: Bearer <admin_token>`

**响应**:
```json
{
  "success": true,
  "config": {
    "server": {
      "host": "0.0.0.0",
      "port": 5000,
      "debug": false
    },
    "upload": {
      "max_file_size": 16777216,
      "allowed_extensions": ["png", "jpg", "jpeg"],
      "upload_folder": "static/uploads",
      "max_file_age": 3600
    },
    "inference": {
      "default_model": "yolo11n.pt",
      "default_confidence": 0.25,
      "default_iou": 0.45
    },
    "security": {
      "rate_limit": {
        "default": "100 per hour",
        "api": "20 per minute"
      }
    }
  }
}
```

## 5. 文件管理接口

### GET /api/files

获取上传文件列表

**请求**:
- **方法**: GET
- **Query参数**:
  - `type`: 文件类型 (`uploads` 或 `outputs`)
  - `limit`: 返回数量限制，默认 20
  - `offset`: 偏移量，默认 0

**示例请求**:
```bash
curl "http://localhost:5000/api/files?type=uploads&limit=10"
```

**响应**:
```json
{
  "success": true,
  "files": [
    {
      "filename": "example1.jpg",
      "path": "static/uploads/example1.jpg",
      "size": 1024000,
      "created_at": "2023-12-07T10:00:00Z",
      "mime_type": "image/jpeg"
    },
    {
      "filename": "example2.jpg",
      "path": "static/uploads/example2.jpg",
      "size": 2048000,
      "created_at": "2023-12-07T09:30:00Z",
      "mime_type": "image/jpeg"
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

### DELETE /api/files/{filename}

删除指定文件

**请求**:
- **方法**: DELETE
- **路径参数**:
  - `filename`: 文件名

**示例请求**:
```bash
curl -X DELETE http://localhost:5000/api/files/example1.jpg
```

**响应**:
```json
{
  "success": true,
  "message": "文件删除成功"
}
```

## 6. 批量检测接口

### POST /api/batch-detect

批量图片检测

**请求**:
- **方法**: POST
- **Content-Type**: multipart/form-data
- **Body**:
  - `files[]` (required): 多个图片文件
  - `model` (optional): 模型名称
  - `confidence` (optional): 置信度阈值
  - `iou` (optional): IOU阈值

**示例请求**:
```bash
curl -X POST \
  http://localhost:5000/api/batch-detect \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "model=yolo11n.pt"
```

**响应**:
```json
{
  "success": true,
  "batch_id": "batch_20231207_103000",
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "detections": [...],
      "inference_time": 0.12
    },
    {
      "filename": "image2.jpg",
      "success": true,
      "detections": [...],
      "inference_time": 0.15
    }
  ],
  "summary": {
    "total_files": 2,
    "successful": 2,
    "failed": 0,
    "total_detections": 5,
    "total_inference_time": 0.27
  }
}
```

## 速率限制

| 接口 | 限制 | 时间窗口 |
|------|------|----------|
| POST /api/detect | 20次 | 1分钟 |
| GET /api/models | 无限制 | - |
| POST /api/batch-detect | 5次 | 1分钟 |
| 其他接口 | 100次 | 1小时 |

## SDK 示例

### Python SDK

```python
import requests

class YOLOClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def detect(self, image_path, model="yolo11n.pt", confidence=0.25):
        """单张图片检测"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'model': model,
                'confidence': confidence
            }
            response = requests.post(
                f"{self.base_url}/api/detect",
                files=files,
                data=data
            )
        return response.json()

    def get_models(self):
        """获取可用模型"""
        response = requests.get(f"{self.base_url}/api/models")
        return response.json()

# 使用示例
client = YOLOClient()
result = client.detect("example.jpg", confidence=0.3)
print(f"检测到 {result['result']['summary']['total_detections']} 个对象")
```

### JavaScript SDK

```javascript
class YOLOClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }

    async detect(file, model = 'yolo11n.pt', confidence = 0.25) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', model);
        formData.append('confidence', confidence);

        const response = await fetch(`${this.baseUrl}/api/detect`, {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    async getModels() {
        const response = await fetch(`${this.baseUrl}/api/models`);
        return await response.json();
    }
}

// 使用示例
const client = new YOLOClient();
document.getElementById('upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const result = await client.detect(file, 'yolo11n.pt', 0.3);
    console.log(`检测到 ${result.result.summary.total_detections} 个对象`);
});
```

## 测试用例

### 自动化测试

```python
import pytest
import requests

BASE_URL = "http://localhost:5000"

def test_health_check():
    response = requests.get(f"{BASE_URL}/api/health")
    assert response.status_code == 200
    assert response.json()['success'] is True

def test_get_models():
    response = requests.get(f"{BASE_URL}/api/models")
    assert response.status_code == 200
    data = response.json()
    assert 'yolo11n.pt' in data['predefined_models']['pytorch']

def test_detect_endpoint():
    # 测试文件上传
    with open('test_image.jpg', 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/api/detect", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'detections' in data['result']
```

## 故障排除

### 常见错误

1. **文件上传失败**
   - 检查文件格式是否支持
   - 确认文件大小在限制范围内
   - 验证文件完整性

2. **模型加载失败**
   - 确认模型文件存在
   - 检查模型格式是否支持
   - 验证模型文件完整性

3. **推理失败**
   - 检查图片格式和大小
   - 确认模型参数有效
   - 查看服务器日志

4. **速率限制**
   - 降低请求频率
   - 考虑使用批量接口
   - 联系管理员调整限制

### 调试工具

- **浏览器开发者工具**: 检查网络请求
- **curl命令**: 直接测试API
- **Postman**: 完整的API测试环境
- **日志文件**: 服务器端错误日志