#!/usr/bin/env python3
"""
YOLO Web Demo - 统一运行脚本
集成配置管理、环境文件管理和应用程序启动功能
"""
import os
import sys
import argparse
import logging
import socket
import json
import secrets
from typing import Optional

# 处理Windows控制台编码问题
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 > nul')
    except:
        pass

# =============================================================================
# 配置系统
# =============================================================================

# 默认配置常量 - 集中管理所有配置
DEFAULT_CONFIG = {
    # 服务器设置
    'HOST': '0.0.0.0',  # 默认允许局域网访问
    'PORT': '5000',

    # 安全设置
    'SECRET_KEY': secrets.token_hex(32),  # 生成默认密钥

    # 文件上传设置
    'MAX_CONTENT_LENGTH': '16777216',  # 16MB in bytes
    'UPLOAD_FOLDER': 'static/uploads',
    'OUTPUT_FOLDER': 'static/outputs',
    'MAX_FILE_AGE': '3600',  # 1 hour in seconds
    'CLEANUP_INTERVAL': '300',  # 5 minutes in seconds

    # 模型设置
    'DEFAULT_MODEL': 'yolo11n.pt',
    'DEFAULT_CONFIDENCE': '0.25',
    'DEFAULT_IOU': '0.45',

    # 日志设置
    'LOG_LEVEL': 'INFO',
    'LOG_FILE': 'logs/app.log',

    # 速率限制设置
    'RATELIMIT_DEFAULT': '100 per hour',
    'RATELIMIT_API': '20 per minute',

    # 环境设置
    'FLASK_ENV': 'development'
}


def ensure_env_file():
    """确保.env文件存在，如果不存在则创建它"""
    env_file = '.env'

    if not os.path.exists(env_file):
        print("创建默认的.env文件...")

        env_content = """# YOLO Web Demo Environment Configuration
# 此文件由应用程序自动生成，包含默认配置

# Environment Settings
FLASK_ENV=development
# FLASK_ENV=production

# Security (REQUIRED for production)
SECRET_KEY={secret_key}

# Server Settings
HOST={host}
PORT={port}

# File Upload Settings
MAX_CONTENT_LENGTH={max_content_length}
UPLOAD_FOLDER={upload_folder}
OUTPUT_FOLDER={output_folder}
MAX_FILE_AGE={max_file_age}
CLEANUP_INTERVAL={cleanup_interval}

# Model Settings
DEFAULT_MODEL={default_model}
DEFAULT_CONFIDENCE={default_confidence}
DEFAULT_IOU={default_iou}

# Logging Settings
LOG_LEVEL={log_level}
LOG_FILE={log_file}

# Rate Limiting Settings
RATELIMIT_DEFAULT={ratelimit_default}
RATELIMIT_API={ratelimit_api}
""".format(
            secret_key=DEFAULT_CONFIG['SECRET_KEY'],
            host=DEFAULT_CONFIG['HOST'],
            port=DEFAULT_CONFIG['PORT'],
            max_content_length=DEFAULT_CONFIG['MAX_CONTENT_LENGTH'],
            upload_folder=DEFAULT_CONFIG['UPLOAD_FOLDER'],
            output_folder=DEFAULT_CONFIG['OUTPUT_FOLDER'],
            max_file_age=DEFAULT_CONFIG['MAX_FILE_AGE'],
            cleanup_interval=DEFAULT_CONFIG['CLEANUP_INTERVAL'],
            default_model=DEFAULT_CONFIG['DEFAULT_MODEL'],
            default_confidence=DEFAULT_CONFIG['DEFAULT_CONFIDENCE'],
            default_iou=DEFAULT_CONFIG['DEFAULT_IOU'],
            log_level=DEFAULT_CONFIG['LOG_LEVEL'],
            log_file=DEFAULT_CONFIG['LOG_FILE'],
            ratelimit_default=DEFAULT_CONFIG['RATELIMIT_DEFAULT'],
            ratelimit_api=DEFAULT_CONFIG['RATELIMIT_API']
        )

        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)

        print(f"已创建.env文件，配置如下:")
        print(f"   - 服务器地址: {DEFAULT_CONFIG['HOST']}:{DEFAULT_CONFIG['PORT']}")
        print(f"   - 上传文件夹: {DEFAULT_CONFIG['UPLOAD_FOLDER']}")
        print(f"   - 输出文件夹: {DEFAULT_CONFIG['OUTPUT_FOLDER']}")
        print(f"   - 默认模型: {DEFAULT_CONFIG['DEFAULT_MODEL']}")

    # 重新加载环境变量
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        print("警告: python-dotenv 未安装，环境变量加载可能不完整")


class Config:
    """Base configuration class"""

    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or DEFAULT_CONFIG['SECRET_KEY']

    # File upload settings
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', DEFAULT_CONFIG['MAX_CONTENT_LENGTH']))
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', DEFAULT_CONFIG['UPLOAD_FOLDER'])
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', DEFAULT_CONFIG['OUTPUT_FOLDER'])

    # Model settings
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', DEFAULT_CONFIG['DEFAULT_MODEL'])
    DEFAULT_CONFIDENCE = float(os.environ.get('DEFAULT_CONFIDENCE', DEFAULT_CONFIG['DEFAULT_CONFIDENCE']))
    DEFAULT_IOU = float(os.environ.get('DEFAULT_IOU', DEFAULT_CONFIG['DEFAULT_IOU']))

    # Performance settings
    MAX_FILE_AGE = int(os.environ.get('MAX_FILE_AGE', DEFAULT_CONFIG['MAX_FILE_AGE']))  # 1 hour in seconds
    CLEANUP_INTERVAL = int(os.environ.get('CLEANUP_INTERVAL', DEFAULT_CONFIG['CLEANUP_INTERVAL']))  # 5 minutes in seconds

    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', DEFAULT_CONFIG['LOG_LEVEL'])
    LOG_FILE = os.environ.get('LOG_FILE', DEFAULT_CONFIG['LOG_FILE'])

    # Allowed file extensions (case-insensitive)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tiff', 'tif'}

    # Rate limiting settings
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', DEFAULT_CONFIG['RATELIMIT_DEFAULT'])
    RATELIMIT_API = os.environ.get('RATELIMIT_API', DEFAULT_CONFIG['RATELIMIT_API'])


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

    def __init__(self):
        super().__init__()
        # In production, ensure SECRET_KEY is set
        if not os.environ.get('SECRET_KEY'):
            raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False


# Configuration mapping
config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    # 首先确保.env文件存在
    ensure_env_file()

    if config_name is None:
        # Check for common environment indicators
        if os.environ.get('FLASK_ENV') == 'production' or \
           os.environ.get('FLASK_DEBUG') == '0' or \
           os.environ.get('PYTHONUNBUFFERED') or \
           os.environ.get('WERKZEUG_RUN_MAIN'):
            config_name = 'production'
        else:
            config_name = os.environ.get('FLASK_ENV', 'default')

    # Ensure we always have a valid configuration
    if config_name not in config_mapping:
        config_name = 'default'

    return config_mapping[config_name]()


# =============================================================================
# 配置管理工具
# =============================================================================

def show_current_config():
    """显示当前配置"""
    print("=" * 60)
    print("YOLO Web Demo - 当前配置")
    print("=" * 60)

    # 确保环境文件存在
    ensure_env_file()

    # 获取当前配置
    config = get_config()

    print("基础配置:")
    print(f"   SECRET_KEY: {'*' * 20}...{config.SECRET_KEY[-8:]}")
    print(f"   DEBUG: {config.DEBUG}")
    print()

    print("服务器配置:")
    print(f"   HOST: {os.environ.get('HOST', DEFAULT_CONFIG['HOST'])}")
    print(f"   PORT: {os.environ.get('PORT', DEFAULT_CONFIG['PORT'])}")
    print()

    print("文件配置:")
    print(f"   MAX_CONTENT_LENGTH: {config.MAX_CONTENT_LENGTH} bytes ({config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB)")
    print(f"   UPLOAD_FOLDER: {config.UPLOAD_FOLDER}")
    print(f"   OUTPUT_FOLDER: {config.OUTPUT_FOLDER}")
    print(f"   MAX_FILE_AGE: {config.MAX_FILE_AGE} seconds")
    print(f"   CLEANUP_INTERVAL: {config.CLEANUP_INTERVAL} seconds")
    print()

    print("模型配置:")
    print(f"   DEFAULT_MODEL: {config.DEFAULT_MODEL}")
    print(f"   DEFAULT_CONFIDENCE: {config.DEFAULT_CONFIDENCE}")
    print(f"   DEFAULT_IOU: {config.DEFAULT_IOU}")
    print()

    print("日志配置:")
    print(f"   LOG_LEVEL: {config.LOG_LEVEL}")
    print(f"   LOG_FILE: {config.LOG_FILE}")
    print()

    print("速率限制:")
    print(f"   RATELIMIT_DEFAULT: {config.RATELIMIT_DEFAULT}")
    print(f"   RATELIMIT_API: {config.RATELIMIT_API}")
    print("=" * 60)


def show_env_file():
    """显示.env文件内容"""
    env_file = '.env'
    if os.path.exists(env_file):
        print("=" * 60)
        print(".env 文件内容")
        print("=" * 60)
        with open(env_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print("=" * 60)
    else:
        print("错误: .env文件不存在")


def export_config_json():
    """导出配置为JSON格式"""
    config_data = {
        "server": {
            "host": os.environ.get('HOST', DEFAULT_CONFIG['HOST']),
            "port": int(os.environ.get('PORT', DEFAULT_CONFIG['PORT']))
        },
        "files": {
            "max_content_length": int(os.environ.get('MAX_CONTENT_LENGTH', DEFAULT_CONFIG['MAX_CONTENT_LENGTH'])),
            "upload_folder": os.environ.get('UPLOAD_FOLDER', DEFAULT_CONFIG['UPLOAD_FOLDER']),
            "output_folder": os.environ.get('OUTPUT_FOLDER', DEFAULT_CONFIG['OUTPUT_FOLDER']),
            "max_file_age": int(os.environ.get('MAX_FILE_AGE', DEFAULT_CONFIG['MAX_FILE_AGE'])),
            "cleanup_interval": int(os.environ.get('CLEANUP_INTERVAL', DEFAULT_CONFIG['CLEANUP_INTERVAL']))
        },
        "models": {
            "default_model": os.environ.get('DEFAULT_MODEL', DEFAULT_CONFIG['DEFAULT_MODEL']),
            "default_confidence": float(os.environ.get('DEFAULT_CONFIDENCE', DEFAULT_CONFIG['DEFAULT_CONFIDENCE'])),
            "default_iou": float(os.environ.get('DEFAULT_IOU', DEFAULT_CONFIG['DEFAULT_IOU']))
        },
        "logging": {
            "log_level": os.environ.get('LOG_LEVEL', DEFAULT_CONFIG['LOG_LEVEL']),
            "log_file": os.environ.get('LOG_FILE', DEFAULT_CONFIG['LOG_FILE'])
        },
        "ratelimit": {
            "default": os.environ.get('RATELIMIT_DEFAULT', DEFAULT_CONFIG['RATELIMIT_DEFAULT']),
            "api": os.environ.get('RATELIMIT_API', DEFAULT_CONFIG['RATELIMIT_API'])
        },
        "environment": {
            "flask_env": os.environ.get('FLASK_ENV', DEFAULT_CONFIG['FLASK_ENV'])
        }
    }

    print("=" * 60)
    print("配置导出 (JSON格式)")
    print("=" * 60)
    print(json.dumps(config_data, indent=2, ensure_ascii=False))
    print("=" * 60)


def create_config_template():
    """创建配置模板文件"""
    template_content = """# YOLO Web Demo 配置模板
# 复制此文件并在其他设备上使用

# 服务器设置
HOST=0.0.0.0
PORT=5000

# 环境设置
FLASK_ENV=development

# 安全设置 (生产环境必须设置)
SECRET_KEY=请生成新的密钥

# 文件上传设置
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=static/uploads
OUTPUT_FOLDER=static/outputs
MAX_FILE_AGE=3600
CLEANUP_INTERVAL=300

# 模型设置
DEFAULT_MODEL=yolo11n.pt
DEFAULT_CONFIDENCE=0.25
DEFAULT_IOU=0.45

# 日志设置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# 速率限制设置
RATELIMIT_DEFAULT=100 per hour
RATELIMIT_API=20 per minute
"""

    with open('config_template.env', 'w', encoding='utf-8') as f:
        f.write(template_content)

    print("已创建配置模板文件: config_template.env")


# =============================================================================
# 应用程序启动
# =============================================================================

def get_local_ip():
    """获取本地局域网IP地址"""
    try:
        # 创建一个UDP socket连接到外部地址来获取本地IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def print_config_info(config, host, port):
    """打印配置信息"""
    print("=" * 60)
    print("YOLO Web Demo - 目标检测服务")
    print("=" * 60)
    print(f"运行环境: {config.__class__.__name__}")
    print(f"调试模式: {'开启' if config.DEBUG else '关闭'}")
    print()
    print("访问地址:")
    print(f"   本地访问: http://127.0.0.1:{port}")

    # 如果是0.0.0.0，显示局域网访问地址
    if host == '0.0.0.0':
        local_ip = get_local_ip()
        print(f"   局域网访问: http://{local_ip}:{port}")
        print(f"   服务器监听: 0.0.0.0:{port} (所有网络接口)")
    else:
        print(f"   服务器地址: http://{host}:{port}")

    print()
    print("配置信息:")
    print(f"   上传文件夹: {config.UPLOAD_FOLDER}")
    print(f"   输出文件夹: {config.OUTPUT_FOLDER}")
    print(f"   最大文件大小: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB")
    print(f"   默认模型: {config.DEFAULT_MODEL}")
    print(f"   默认置信度: {config.DEFAULT_CONFIDENCE}")
    print(f"   默认IOU阈值: {config.DEFAULT_IOU}")
    print(f"   日志级别: {config.LOG_LEVEL}")
    print(f"   文件清理间隔: {config.CLEANUP_INTERVAL}秒")

    if config.DEBUG:
        print()
        print("警告: 调试模式已启用，不建议在生产环境使用!")
    else:
        print()
        print("生产模式已启用")

    print("=" * 60)
    print("按 Ctrl+C 停止服务器")
    print()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO Web Demo - 目标检测Web服务')
    parser.add_argument('--host', default=None, help='服务器地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=None, help='服务器端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--production', action='store_true', help='启用生产模式')
    parser.add_argument('--config', action='store_true', help='显示当前配置并退出')
    parser.add_argument('--create-env', action='store_true', help='强制创建新的.env文件')

    # 配置管理命令
    parser.add_argument('--manage', choices=['show', 'env', 'json', 'template', 'sync'],
                       help='配置管理命令')

    return parser.parse_args()


def start_app(host, port, config):
    """启动Flask应用程序"""
    try:
        # 导入Flask应用
        from app import app
        from dotenv import load_dotenv

        # 确保环境变量已加载
        load_dotenv(override=True)

        # Run the Flask application
        app.run(
            host=host,
            port=port,
            debug=config.DEBUG,
            use_reloader=config.DEBUG  # Only use reloader in debug mode
        )
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")
        sys.exit(1)


def main():
    """Main function to run the Flask application"""
    # 解析命令行参数
    args = parse_arguments()

    # 处理配置管理命令
    if args.manage:
        if args.manage == 'show':
            show_current_config()
        elif args.manage == 'env':
            show_env_file()
        elif args.manage == 'json':
            export_config_json()
        elif args.manage == 'template':
            create_config_template()
        elif args.manage == 'sync':
            print("同步配置...")
            ensure_env_file()
            print("配置同步完成")
        return

    # 如果强制创建.env文件
    if args.create_env:
        if os.path.exists('.env'):
            backup = '.env.backup'
            os.rename('.env', backup)
            print(f"已备份现有.env文件为: {backup}")
        print("重新创建.env文件...")

    # 获取配置 (这会自动创建.env文件如果不存在)
    config = get_config()

    # 命令行参数覆盖配置
    host = args.host or os.environ.get('HOST', DEFAULT_CONFIG['HOST'])
    port = args.port or int(os.environ.get('PORT', DEFAULT_CONFIG['PORT']))

    # 环境模式设置
    if args.debug:
        os.environ['FLASK_ENV'] = 'development'
        config = get_config('development')
    elif args.production:
        os.environ['FLASK_ENV'] = 'production'
        config = get_config('production')

    # 如果只是显示配置
    if args.config:
        print_config_info(config, host, port)
        return

    # 打印配置信息
    print_config_info(config, host, port)

    # 启动应用程序
    start_app(host, port, config)


if __name__ == "__main__":
    main()