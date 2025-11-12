@echo off
echo YOLO Web Demo 启动脚本
echo =======================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: Python 未安装或未在PATH中
    pause
    exit /b 1
)

REM 检查必要的Python包
echo 检查依赖包...
python -c "import flask, cv2, ultralytics, PIL, werkzeug, flask_limiter" 2>nul
if errorlevel 1 (
    echo ❌ 错误: 缺少必要的Python包
    echo 请运行: pip install flask opencv-python ultralytics pillow werkzeug flask-limiter
    pause
    exit /b 1
)

echo ✅ 所有依赖已满足

REM 创建必要的目录
if not exist "static\uploads" mkdir "static\uploads"
if not exist "static\outputs" mkdir "static\outputs"
if not exist "logs" mkdir "logs"

REM 检查环境文件
if not exist ".env" (
    echo 创建环境配置文件...
    python -c "from run import ensure_env_file; ensure_env_file()"
)

echo.
echo 启动 YOLO Web Demo...
echo 访问地址: http://127.0.0.1:5000
echo 按 Ctrl+C 停止服务器
echo.

REM 启动应用
python app.py

pause