"""
路径处理工具函数
"""
import os


def normalize_static_path(file_path):
    """
    标准化文件路径，确保可以正确用于Flask的url_for('static', filename=...)

    Args:
        file_path (str): 原始文件路径

    Returns:
        str: 标准化后的相对路径
    """
    if not file_path:
        return file_path

    # 确保使用正斜杠
    normalized = file_path.replace('\\', '/')

    # 移除开头的 'static/' 或 'static'
    if normalized.startswith('static/'):
        normalized = normalized[7:]  # 移除 'static/'
    elif normalized.startswith('static'):
        normalized = normalized[6:]  # 移除 'static'

    return normalized


def get_static_url_path(file_path):
    """
    获取静态文件的URL路径

    Args:
        file_path (str): 文件路径

    Returns:
        str: URL路径
    """
    normalized = normalize_static_path(file_path)
    return f"static/{normalized}" if normalized else "static"


def ensure_static_path(file_path):
    """
    确保路径包含 'static/' 前缀

    Args:
        file_path (str): 原始文件路径

    Returns:
        str: 包含 'static/' 的完整路径
    """
    if not file_path:
        return file_path

    normalized = file_path.replace('\\', '/')

    # 如果不包含 'static/'，则添加
    if not normalized.startswith('static/'):
        if normalized.startswith('static'):
            normalized = f"{normalized}/"
        else:
            normalized = f"static/{normalized}"

    return normalized