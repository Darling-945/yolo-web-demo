#!/usr/bin/env python3
"""
YOLO模型转换脚本 - PyTorch转ONNX
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: ultralytics库未安装")
    sys.exit(1)


class ModelConverter:
    def __init__(self, input_size: int = 640, simplify: bool = True):
        self.input_size = input_size
        self.simplify = simplify
        print(f"转换参数: 尺寸={input_size}, 简化={simplify}")

    def convert_single_model(self, model_path: str, output_dir: str = None) -> Dict[str, Any]:
        """转换单个模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if not model_path.endswith('.pt'):
            raise ValueError("输入文件必须是PyTorch模型(.pt)")

        model_path = Path(model_path)
        output_dir = Path(output_dir) if output_dir else model_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / f"{model_path.stem}.onnx"

        start_time = datetime.now()
        print(f"转换模型: {model_path.name}")

        try:
            model = YOLO(str(model_path))
            model.export(format='onnx', imgsz=self.input_size, simplify=self.simplify)

            # 移动生成的ONNX文件到指定输出目录
            generated_onnx = model_path.parent / f"{model_path.stem}.onnx"
            if generated_onnx.exists() and str(generated_onnx) != str(onnx_path):
                generated_onnx.rename(onnx_path)

            if onnx_path.exists():
                file_size = onnx_path.stat().st_size / (1024 * 1024)
                duration = (datetime.now() - start_time).total_seconds()
                print(f"✅ 转换成功: {onnx_path.name} ({file_size:.2f}MB, {duration:.2f}s)")
                return {
                    'status': 'success',
                    'model_name': model_path.name,
                    'output_path': str(onnx_path),
                    'file_size_mb': round(file_size, 2),
                    'duration_seconds': round(duration, 2)
                }
            else:
                raise FileNotFoundError("ONNX文件生成失败")

        except Exception as e:
            print(f"❌ 转换失败: {str(e)}")
            return {
                'status': 'failed',
                'model_name': model_path.name,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(description='YOLO模型转换工具 - PyTorch转ONNX')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入模型文件路径(.pt)')
    parser.add_argument('--output-dir', '-o', type=str, help='输出目录路径')
    parser.add_argument('--input-size', type=int, default=640, help='模型输入尺寸 (默认: 640)')
    parser.add_argument('--no-simplify', action='store_true', help='不简化ONNX模型')

    args = parser.parse_args()

    print("=" * 50)
    print("YOLO模型转换工具")
    print("=" * 50)

    converter = ModelConverter(
        input_size=args.input_size,
        simplify=not args.no_simplify
    )

    try:
        result = converter.convert_single_model(args.input, args.output_dir)
        if result['status'] == 'success':
            print("🎉 转换成功!")
        else:
            print(f"❌ 转换失败: {result.get('error', '未知错误')}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 转换错误: {e}")
        sys.exit(1)

    print("转换完成!")


if __name__ == "__main__":
    main()