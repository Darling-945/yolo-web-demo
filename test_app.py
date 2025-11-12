#!/usr/bin/env python3
"""
测试YOLO Web Demo的基本功能
"""
import requests
import os
import tempfile
from PIL import Image
import numpy as np

def create_test_image():
    """创建一个测试图片"""
    # 创建一个简单的测试图片 (100x100像素，红色背景)
    img = Image.new('RGB', (100, 100), color='red')
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, 'JPEG')
    return temp_file.name

def test_upload():
    """测试图片上传功能"""
    try:
        # 创建测试图片
        test_image_path = create_test_image()
        print(f"创建测试图片: {test_image_path}")

        # 准备上传数据
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {
                'model': 'yolo11n.pt',
                'confidence': '0.25',
                'iou': '0.45'
            }

            print("发送POST请求到 /infer...")
            response = requests.post('http://127.0.0.1:5000/infer', files=files, data=data)

            print(f"响应状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")

            if response.status_code == 200:
                print("✅ 上传成功！")
                print(f"响应内容长度: {len(response.content)} 字节")
                # 保存响应为HTML文件用于检查
                with open('test_response.html', 'wb') as out_file:
                    out_file.write(response.content)
                print("响应内容已保存到 test_response.html")
            else:
                print(f"❌ 上传失败: {response.status_code}")
                print(f"错误内容: {response.text}")

        # 清理测试文件
        os.unlink(test_image_path)

    except requests.exceptions.ConnectionError:
        print("❌ 连接失败 - 请确保Flask应用正在运行在 http://127.0.0.1:5000")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

def test_api():
    """测试API端点"""
    try:
        print("测试 /api/models 端点...")
        response = requests.get('http://127.0.0.1:5000/api/models')

        if response.status_code == 200:
            print("✅ API测试成功！")
            print(f"模型列表: {response.json()}")
        else:
            print(f"❌ API测试失败: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("❌ 连接失败 - 请确保Flask应用正在运行")
    except Exception as e:
        print(f"❌ API测试失败: {str(e)}")

if __name__ == "__main__":
    print("开始测试 YOLO Web Demo...")
    print("=" * 50)

    # 测试API
    test_api()
    print()

    # 测试上传
    test_upload()

    print("=" * 50)
    print("测试完成")