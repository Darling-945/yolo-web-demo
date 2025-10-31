#!/usr/bin/env python3
"""
测试脚本 - 验证 YOLO Web Demo 功能
"""
import os
import sys
import requests
import json
from io import BytesIO

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有必要的导入"""
    print("1. 测试导入...")
    try:
        from app import app
        print("   ✅ Flask app 导入成功")

        # 测试 model_inference 模块
        import model_inference
        print("   ✅ model_inference 模块导入成功")

        # 测试 detect 模块
        from utils import detect
        print("   ✅ detect 模块导入成功")

        print("   ✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("2. 测试模型加载...")
    try:
        from model_inference import YOLOInference
        inference = YOLOInference(model_path='yolo11n.pt', conf_threshold=0.25)
        print(f"   ✅ 模型加载成功: {inference.model_path}")
        print(f"   ✅ 置信度阈值: {inference.conf_threshold}")
        return True
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return False

def test_available_models():
    """测试可用模型列表"""
    print("3. 测试可用模型列表...")
    try:
        from model_inference import get_available_models
        models_data = get_available_models()
        predefined_count = len(models_data['predefined_models'])
        custom_count = len(models_data['custom_models'])
        print(f"   ✅ 预定义模型: {predefined_count} 个")
        print(f"   ✅ 自定义模型: {custom_count} 个")
        print(f"   ✅ 模型列表: {models_data['predefined_models'][:3]}...")  # 只显示前3个
        return True
    except Exception as e:
        print(f"   ❌ 获取模型列表失败: {e}")
        return False

def test_directories():
    """测试必要的目录"""
    print("4. 测试目录结构...")
    required_dirs = ['static/uploads', 'static/outputs', 'templates']
    all_exist = True

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ 目录存在: {dir_path}")
        else:
            print(f"   ❌ 目录不存在: {dir_path}")
            all_exist = False

    return all_exist

def test_templates():
    """测试模板文件"""
    print("5. 测试模板文件...")
    required_templates = ['base.html', 'index.html', 'inference.html']
    all_exist = True

    for template in required_templates:
        template_path = os.path.join('templates', template)
        if os.path.exists(template_path):
            print(f"   ✅ 模板存在: {template}")
        else:
            print(f"   ❌ 模板不存在: {template}")
            all_exist = False

    return all_exist

def test_requirements():
    """测试依赖文件"""
    print("6. 测试依赖文件...")
    if os.path.exists('requirements.txt'):
        print("   ✅ requirements.txt 存在")

        # 检查关键依赖
        key_dependencies = ['Flask', 'ultralytics', 'torch', 'opencv-python']
        try:
            with open('requirements.txt', 'r', encoding='utf-8') as f:
                content = f.read()

            for dep in key_dependencies:
                if dep.lower() in content.lower():
                    print(f"   ✅ 依赖存在: {dep}")
                else:
                    print(f"   ⚠️  依赖可能缺失: {dep}")
        except Exception as e:
            print(f"   ❌ 读取依赖文件失败: {e}")
            return False

        return True
    else:
        print("   ❌ requirements.txt 不存在")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("YOLO Web Demo 功能测试")
    print("=" * 50)

    tests = [
        test_imports,
        test_model_loading,
        test_available_models,
        test_directories,
        test_templates,
        test_requirements
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！应用已准备就绪。")
        print("\n启动应用:")
        print("  python run.py")
        print("  或")
        print("  python app.py")
        print("\n然后在浏览器中访问: http://127.0.0.1:5000")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")

    print("=" * 50)

if __name__ == "__main__":
    main()