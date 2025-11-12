#!/bin/bash

echo "YOLO Web Demo 启动脚本"
echo "======================="

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python 3 未安装"
    exit 1
fi

# 检查必要的Python包
echo "检查依赖包..."
python3 -c "import flask, cv2, ultralytics, PIL, werkzeug, flask_limiter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: 缺少必要的Python包"
    echo "请运行: pip install flask opencv-python ultralytics pillow werkzeug flask-limiter"
    exit 1
fi

echo "✅ 所有依赖已满足"

# 创建必要的目录
mkdir -p static/uploads static/outputs logs

# 检查环境文件
if [ ! -f ".env" ]; then
    echo "创建环境配置文件..."
    python3 -c "from run import ensure_env_file; ensure_env_file()"
fi

echo ""
echo "启动 YOLO Web Demo..."
echo "访问地址: http://127.0.0.1:5000"
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动应用
python3 app.py