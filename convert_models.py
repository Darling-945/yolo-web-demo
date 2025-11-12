#!/usr/bin/env python3
"""
YOLO模型转换脚本
支持将PyTorch模型(.pt)转换为ONNX格式(.onnx)
使用Ultralytics YOLO官方接口
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: ultralytics库未安装")
    print("请运行: pip install ultralytics")
    sys.exit(1)


class ModelConverter:
    """YOLO模型转换器"""

    def __init__(self,
                 input_size: int = 640,
                 batch_size: int = 1,
                 opset_version: int = 12,
                 simplify: bool = True,
                 dynamic: bool = False,
                 workspace: int = 4):
        """
        初始化转换器

        Args:
            input_size: 模型输入尺寸
            batch_size: 批处理大小
            opset_version: ONNX opset版本
            simplify: 是否简化ONNX模型
            dynamic: 是否使用动态输入尺寸
            workspace: TensorRT workspace大小(GB)
        """
        self.input_size = input_size
        self.batch_size = batch_size
        self.opset_version = opset_version
        self.simplify = simplify
        self.dynamic = dynamic
        self.workspace = workspace
        self.conversion_log = []

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('model_conversion.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def convert_single_model(self,
                           model_path: str,
                           output_dir: Optional[str] = None,
                           keep_original: bool = True) -> Dict[str, Any]:
        """
        转换单个模型

        Args:
            model_path: 输入模型路径(.pt)
            output_dir: 输出目录
            keep_original: 是否保留原始模型

        Returns:
            转换结果信息
        """
        start_time = datetime.now()

        # 验证输入文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if not model_path.endswith('.pt'):
            raise ValueError(f"输入文件必须是PyTorch模型(.pt): {model_path}")

        model_path = Path(model_path)

        # 设置输出路径
        if output_dir is None:
            output_dir = model_path.parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = output_dir / f"{model_path.stem}.onnx"

        # 记录转换开始
        conversion_info = {
            'model_name': model_path.name,
            'input_path': str(model_path),
            'output_path': str(onnx_path),
            'start_time': start_time.isoformat(),
            'parameters': {
                'input_size': self.input_size,
                'batch_size': self.batch_size,
                'opset_version': self.opset_version,
                'simplify': self.simplify,
                'dynamic': self.dynamic
            }
        }

        try:
            self.logger.info(f"开始转换模型: {model_path.name}")

            # 加载模型
            self.logger.info("正在加载PyTorch模型...")
            model = YOLO(str(model_path))

            # 验证模型
            self.logger.info("正在验证模型...")
            model_info = model.info()
            self.logger.info(f"模型信息: {json.dumps(model_info, indent=2, ensure_ascii=False)}")

            # 导出为ONNX
            self.logger.info("正在导出为ONNX格式...")
            export_args = {
                'imgsz': self.input_size,
                'batch': self.batch_size,
                'opset': self.opset_version,
                'simplify': self.simplify,
                'dynamic': self.dynamic
            }

            # 执行导出
            model.export(format='onnx', **export_args)

            # 移动生成的ONNX文件到指定输出目录
            generated_onnx = model_path.parent / f"{model_path.stem}.onnx"
            if generated_onnx.exists() and str(generated_onnx) != str(onnx_path):
                generated_onnx.rename(onnx_path)

            # 验证ONNX文件
            if onnx_path.exists():
                file_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
                conversion_info.update({
                    'status': 'success',
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - start_time).total_seconds(),
                    'output_size_mb': round(file_size, 2)
                })

                self.logger.info(f"✅ 转换成功: {onnx_path.name}")
                self.logger.info(f"   文件大小: {file_size:.2f} MB")
                self.logger.info(f"   耗时: {conversion_info['duration_seconds']:.2f} 秒")

                # 可选：删除原始文件
                if not keep_original:
                    os.remove(model_path)
                    self.logger.info(f"   已删除原始文件: {model_path.name}")

            else:
                raise FileNotFoundError("ONNX文件生成失败")

        except Exception as e:
            error_msg = f"转换失败: {str(e)}"
            self.logger.error(error_msg)
            conversion_info.update({
                'status': 'failed',
                'error': error_msg,
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            })

        self.conversion_log.append(conversion_info)
        return conversion_info

    def convert_batch_models(self,
                           input_dir: str,
                           output_dir: str,
                           pattern: str = "*.pt",
                           keep_original: bool = True) -> List[Dict[str, Any]]:
        """
        批量转换模型

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            keep_original: 是否保留原始模型

        Returns:
            批量转换结果列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        # 查找所有.pt文件
        model_files = list(input_dir.glob(pattern))

        if not model_files:
            self.logger.warning(f"在 {input_dir} 中未找到匹配 {pattern} 的文件")
            return []

        self.logger.info(f"找到 {len(model_files)} 个模型文件待转换")

        results = []
        for i, model_file in enumerate(model_files, 1):
            self.logger.info(f"\n[{i}/{len(model_files)}] 转换 {model_file.name}")

            try:
                result = self.convert_single_model(
                    str(model_file),
                    str(output_dir),
                    keep_original
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"转换 {model_file.name} 时出错: {e}")
                results.append({
                    'model_name': model_file.name,
                    'status': 'error',
                    'error': str(e)
                })

        # 生成批量转换报告
        self._generate_batch_report(results, output_dir)

        return results

    def _generate_batch_report(self, results: List[Dict[str, Any]], output_dir: Path):
        """生成批量转换报告"""
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') in ['failed', 'error']]

        report = {
            'batch_time': datetime.now().isoformat(),
            'total_models': len(results),
            'successful_conversions': len(successful),
            'failed_conversions': len(failed),
            'success_rate': f"{len(successful)/len(results)*100:.1f}%",
            'results': results
        }

        if failed:
            report['failed_models'] = [r['model_name'] for r in failed]

        # 保存报告
        report_path = output_dir / 'conversion_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n📊 批量转换完成!")
        self.logger.info(f"   成功: {len(successful)} 个")
        self.logger.info(f"   失败: {len(failed)} 个")
        self.logger.info(f"   成功率: {report['success_rate']}")
        self.logger.info(f"   报告已保存: {report_path}")

    def save_conversion_log(self, output_path: str = "conversion_history.json"):
        """保存转换历史记录"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'total_conversions': len(self.conversion_log),
                'conversions': self.conversion_log
            }, f, indent=2, ensure_ascii=False)

        self.logger.info(f"转换历史已保存: {output_path}")


def create_sample_images(output_dir: str, size: int = 640, count: int = 3):
    """创建测试用图片（可选功能）"""
    try:
        import numpy as np
        from PIL import Image

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"正在创建 {count} 张测试图片...")

        for i in range(count):
            # 创建随机彩色图片
            img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = output_dir / f"test_image_{i+1}.jpg"
            img.save(img_path)

        print(f"测试图片已保存到: {output_dir}")

    except ImportError:
        print("警告: 无法创建测试图片，请安装 pillow 和 numpy")
    except Exception as e:
        print(f"创建测试图片失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLO模型转换工具 - 将PyTorch模型(.pt)转换为ONNX格式(.onnx)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换单个模型
  python convert_models.py --input yolo11n.pt

  # 批量转换
  python convert_models.py --input-dir ./models --output-dir ./onnx_models

  # 自定义参数转换
  python convert_models.py --input custom_model.pt --output-dir ./outputs --input-size 1024

  # 转换后删除原始文件
  python convert_models.py --input yolo11n.pt --no-keep-original
        """
    )

    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str, help='输入模型文件路径(.pt)')
    input_group.add_argument('--input-dir', type=str, help='输入目录路径（批量转换）')

    # 输出参数
    parser.add_argument('--output-dir', '-o', type=str, help='输出目录路径')
    parser.add_argument('--no-keep-original', action='store_true', help='转换后删除原始模型文件')

    # 转换参数
    parser.add_argument('--input-size', type=int, default=640, help='模型输入尺寸 (默认: 640)')
    parser.add_argument('--batch-size', type=int, default=1, help='批处理大小 (默认: 1)')
    parser.add_argument('--opset-version', type=int, default=12, help='ONNX opset版本 (默认: 12)')
    parser.add_argument('--no-simplify', action='store_true', help='不简化ONNX模型')
    parser.add_argument('--dynamic', action='store_true', help='使用动态输入尺寸')

    # 其他功能
    parser.add_argument('--pattern', type=str, default='*.pt', help='批量转换文件模式 (默认: "*.pt")')
    parser.add_argument('--save-log', action='store_true', help='保存转换历史记录')
    parser.add_argument('--create-test-images', action='store_true', help='创建测试用图片')

    args = parser.parse_args()

    # 创建转换器
    converter = ModelConverter(
        input_size=args.input_size,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        simplify=not args.no_simplify,
        dynamic=args.dynamic
    )

    print("=" * 60)
    print("YOLO模型转换工具")
    print("=" * 60)
    print(f"转换参数:")
    print(f"  输入尺寸: {args.input_size}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  ONNX opset版本: {args.opset_version}")
    print(f"  简化模型: {'是' if not args.no_simplify else '否'}")
    print(f"  动态输入: {'是' if args.dynamic else '否'}")
    print("=" * 60)

    try:
        if args.input:
            # 单个模型转换
            result = converter.convert_single_model(
                args.input,
                args.output_dir,
                not args.no_keep_original
            )

            if result['status'] == 'success':
                print(f"\n🎉 转换成功!")
                print(f"   输出文件: {result['output_path']}")
                print(f"   文件大小: {result['output_size_mb']} MB")
                print(f"   耗时: {result['duration_seconds']:.2f} 秒")
            else:
                print(f"\n❌ 转换失败: {result.get('error', '未知错误')}")

        else:
            # 批量转换
            results = converter.convert_batch_models(
                args.input_dir,
                args.output_dir or f"{args.input_dir}_onnx",
                args.pattern,
                not args.no_keep_original
            )

            # 保存转换历史
            if args.save_log:
                converter.save_conversion_log()

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断转换过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 转换过程出错: {e}")
        sys.exit(1)

    # 创建测试图片（可选）
    if args.create_test_images:
        output_dir = args.output_dir or "test_images"
        create_sample_images(output_dir, args.input_size)

    print("\n转换完成!")


if __name__ == "__main__":
    main()