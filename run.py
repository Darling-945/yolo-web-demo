#!/usr/bin/env python3
"""
YOLO Web Demo - 运行脚本
"""
import os
import sys
import argparse
import socket
import secrets
from typing import Optional

# 处理Windows控制台编码问题
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

# 默认配置
DEFAULT_CONFIG = {
    'HOST': '0.0.0.0',
    'PORT': '5000',
    'SECRET_KEY': secrets.token_hex(32),
    'MAX_CONTENT_LENGTH': '104857600',
    'UPLOAD_FOLDER': 'static/uploads',
    'OUTPUT_FOLDER': 'static/outputs',
    'MAX_FILE_AGE': '3600',
    'CLEANUP_INTERVAL': '300',
    'DEFAULT_MODEL': 'yolo11n.pt',
    'DEFAULT_CONFIDENCE': '0.25',
    'DEFAULT_IOU': '0.45',
    'LOG_LEVEL': 'INFO',
    'LOG_FILE': 'logs/app.log',
    'RATELIMIT_DEFAULT': '100 per hour',
    'RATELIMIT_API': '20 per minute',
    'FLASK_ENV': 'development'
}


def ensure_env_file():
    """确保.env文件存在，如果不存在则创建它"""
    if not os.path.exists('.env'):
        print("创建默认.env文件...")

        env_content = f"""# YOLO Web Demo Configuration
FLASK_ENV=development
SECRET_KEY={DEFAULT_CONFIG['SECRET_KEY']}
HOST={DEFAULT_CONFIG['HOST']}
PORT={DEFAULT_CONFIG['PORT']}
MAX_CONTENT_LENGTH={DEFAULT_CONFIG['MAX_CONTENT_LENGTH']}
UPLOAD_FOLDER={DEFAULT_CONFIG['UPLOAD_FOLDER']}
OUTPUT_FOLDER={DEFAULT_CONFIG['OUTPUT_FOLDER']}
MAX_FILE_AGE={DEFAULT_CONFIG['MAX_FILE_AGE']}
CLEANUP_INTERVAL={DEFAULT_CONFIG['CLEANUP_INTERVAL']}
DEFAULT_MODEL={DEFAULT_CONFIG['DEFAULT_MODEL']}
DEFAULT_CONFIDENCE={DEFAULT_CONFIG['DEFAULT_CONFIDENCE']}
DEFAULT_IOU={DEFAULT_CONFIG['DEFAULT_IOU']}
LOG_LEVEL={DEFAULT_CONFIG['LOG_LEVEL']}
LOG_FILE={DEFAULT_CONFIG['LOG_FILE']}
RATELIMIT_DEFAULT={DEFAULT_CONFIG['RATELIMIT_DEFAULT']}
RATELIMIT_API={DEFAULT_CONFIG['RATELIMIT_API']}
"""

        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("已创建.env文件")

    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        print("警告: python-dotenv 未安装")


class Config:
    def __init__(self):
        self.SECRET_KEY = os.environ.get('SECRET_KEY') or DEFAULT_CONFIG['SECRET_KEY']
        self.MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', DEFAULT_CONFIG['MAX_CONTENT_LENGTH']))
        self.UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', DEFAULT_CONFIG['UPLOAD_FOLDER'])
        self.OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', DEFAULT_CONFIG['OUTPUT_FOLDER'])
        self.DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', DEFAULT_CONFIG['DEFAULT_MODEL'])
        self.DEFAULT_CONFIDENCE = float(os.environ.get('DEFAULT_CONFIDENCE', DEFAULT_CONFIG['DEFAULT_CONFIDENCE']))
        self.DEFAULT_IOU = float(os.environ.get('DEFAULT_IOU', DEFAULT_CONFIG['DEFAULT_IOU']))
        self.MAX_FILE_AGE = int(os.environ.get('MAX_FILE_AGE', DEFAULT_CONFIG['MAX_FILE_AGE']))
        self.CLEANUP_INTERVAL = int(os.environ.get('CLEANUP_INTERVAL', DEFAULT_CONFIG['CLEANUP_INTERVAL']))
        self.LOG_LEVEL = os.environ.get('LOG_LEVEL', DEFAULT_CONFIG['LOG_LEVEL'])
        self.LOG_FILE = os.environ.get('LOG_FILE', DEFAULT_CONFIG['LOG_FILE'])
        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tiff', 'tif'}
        self.RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', DEFAULT_CONFIG['RATELIMIT_DEFAULT'])
        self.RATELIMIT_API = os.environ.get('RATELIMIT_API', DEFAULT_CONFIG['RATELIMIT_API'])
        self.DEBUG = False


class DevelopmentConfig(Config):
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    def __init__(self):
        super().__init__()
        if not os.environ.get('SECRET_KEY'):
            raise ValueError("SECRET_KEY must be set in production")


def get_config(config_name: Optional[str] = None) -> Config:
    ensure_env_file()

    if config_name is None:
        config_name = 'production' if os.environ.get('FLASK_ENV') == 'production' else 'development'

    return DevelopmentConfig() if config_name == 'development' else ProductionConfig()


def show_current_config():
    config = get_config()
    print("=" * 60)
    print("YOLO Web Demo - 当前配置")
    print("=" * 60)
    print(f"DEBUG: {config.DEBUG}")
    print(f"HOST: {os.environ.get('HOST', DEFAULT_CONFIG['HOST'])}")
    print(f"PORT: {os.environ.get('PORT', DEFAULT_CONFIG['PORT'])}")
    print(f"MAX_CONTENT_LENGTH: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB")
    print(f"DEFAULT_MODEL: {config.DEFAULT_MODEL}")
    print("=" * 60)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"


def print_config_info(config, host, port):
    print("=" * 60)
    print("YOLO Web Demo - 目标检测服务")
    print("=" * 60)
    print(f"调试模式: {'开启' if config.DEBUG else '关闭'}")
    print(f"本地访问: http://127.0.0.1:{port}")

    if host == '0.0.0.0':
        local_ip = get_local_ip()
        print(f"局域网访问: http://{local_ip}:{port}")

    print(f"默认模型: {config.DEFAULT_MODEL}")
    print("=" * 60)


def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO Web Demo - 目标检测Web服务')
    parser.add_argument('--host', default=None, help='服务器地址')
    parser.add_argument('--port', type=int, default=None, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--production', action='store_true', help='启用生产模式')
    parser.add_argument('--config', action='store_true', help='显示当前配置并退出')
    parser.add_argument('--manage', choices=['show'], help='配置管理命令')
    return parser.parse_args()


def start_app(host, port, config):
    try:
        from app import app
        from dotenv import load_dotenv
        load_dotenv(override=True)

        app.run(host=host, port=port, debug=config.DEBUG, use_reloader=config.DEBUG)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")
        sys.exit(1)


def main():
    args = parse_arguments()

    if args.manage == 'show':
        show_current_config()
        return

    config = get_config()
    host = args.host or os.environ.get('HOST', DEFAULT_CONFIG['HOST'])
    port = args.port or int(os.environ.get('PORT', DEFAULT_CONFIG['PORT']))

    if args.debug:
        config = get_config('development')
    elif args.production:
        config = get_config('production')

    if args.config:
        print_config_info(config, host, port)
        return

    print_config_info(config, host, port)
    start_app(host, port, config)


if __name__ == "__main__":
    main()