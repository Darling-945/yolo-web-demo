from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from flask.json.provider import DefaultJSONProvider
import os
import json
import numpy as np
import time
import socket
import select
import uuid
import threading
from typing import Optional, Dict
from model_inference import yolo_inference, get_available_models
from run import get_config
from utils import secure_file_upload, secure_multiple_files_upload, log_security_event, setup_app_logging, process_inference_parameters, generate_unique_filename, normalize_static_path
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Custom JSON provider to handle NumPy types (Flask 2.3+ compatible)
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Set custom JSON provider
app.json_provider_class = NumpyJSONProvider

# Load configuration
config = get_config()
app.config.from_object(config)

# Application constants
VIDEO_PROCESSING_TIMEOUT_SECONDS = 300  # 5 minutes timeout for video processing
BATCH_META_DIR = 'static/batch_meta'  # Directory for batch metadata files

# Set up logging, directories, and cleanup using utils
logger = setup_app_logging(config)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[config.RATELIMIT_DEFAULT]
)

logger.info("YOLO Web Demo started")

# ============================================
# 异步推理进度跟踪系统
# ============================================

# 全局任务存储
inference_tasks = {}
tasks_lock = threading.Lock()


def create_task(task_type='image', file_info=None):
    """创建新的推理任务"""
    task_id = str(uuid.uuid4())
    task = {
        'id': task_id,
        'type': task_type,  # 'image' 或 'video'
        'status': 'pending',  # pending, processing, completed, failed, cancelled
        'progress': 0,
        'message': '任务已创建，等待处理...',
        'file_info': file_info,
        'result': None,
        'error': None,
        'created_at': time.time(),
        'started_at': None,
        'completed_at': None
    }

    with tasks_lock:
        inference_tasks[task_id] = task

    logger.info(f"Created inference task: {task_id} (type: {task_type})")
    return task_id


def update_task_progress(task_id, progress, message, status='processing'):
    """更新任务进度"""
    with tasks_lock:
        if task_id in inference_tasks:
            inference_tasks[task_id]['progress'] = progress
            inference_tasks[task_id]['message'] = message
            inference_tasks[task_id]['status'] = status
            logger.info(f"Task {task_id}: {progress}% - {message}")


def complete_task(task_id, result):
    """标记任务完成"""
    with tasks_lock:
        if task_id in inference_tasks:
            inference_tasks[task_id]['status'] = 'completed'
            inference_tasks[task_id]['progress'] = 100
            inference_tasks[task_id]['message'] = '处理完成！'
            inference_tasks[task_id]['result'] = result
            inference_tasks[task_id]['completed_at'] = time.time()
            logger.info(f"Task {task_id} completed")


def fail_task(task_id, error_message):
    """标记任务失败"""
    with tasks_lock:
        if task_id in inference_tasks:
            inference_tasks[task_id]['status'] = 'failed'
            inference_tasks[task_id]['message'] = f'处理失败: {error_message}'
            inference_tasks[task_id]['error'] = error_message
            inference_tasks[task_id]['completed_at'] = time.time()
            logger.error(f"Task {task_id} failed: {error_message}")


def get_task(task_id: str) -> Optional[Dict]:
    """获取任务信息（返回深拷贝以避免竞争条件）"""
    import copy
    with tasks_lock:
        task = inference_tasks.get(task_id)
        if task is not None:
            return copy.deepcopy(task)
        return None


def cleanup_old_tasks(max_age_seconds: int = None):
    """清理旧任务"""
    if max_age_seconds is None:
        max_age_seconds = config.MAX_FILE_AGE
    current_time = time.time()
    with tasks_lock:
        to_remove = []
        for task_id, task in inference_tasks.items():
            age = current_time - task['created_at']
            if age > max_age_seconds:
                to_remove.append(task_id)

        for task_id in to_remove:
            del inference_tasks[task_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old tasks")


# 定期清理旧任务
def start_task_cleanup_daemon():
    """启动任务清理守护线程"""
    def cleanup():
        while True:
            time.sleep(config.CLEANUP_INTERVAL)
            cleanup_old_tasks()

    cleanup_thread = threading.Thread(target=cleanup, daemon=True)
    cleanup_thread.start()


start_task_cleanup_daemon()


# ============================================
# 工具函数
# ============================================

def is_client_connected() -> bool:
    """
    Check if the client is still connected.
    Returns False if client has disconnected, True otherwise.

    Note: On Windows, direct socket detection is unreliable due to platform limitations.
    The timeout mechanism in start_client_disconnect_monitor serves as the primary
    protection against abandoned requests on Windows.
    """
    import sys
    try:
        # Try to get the werkzeug socket
        wsgi_input = request.environ.get('wsgi.input')
        if wsgi_input is None:
            return False

        # Windows platform has limited socket detection capabilities
        if sys.platform == 'win32':
            # Try multiple detection methods on Windows
            # Method 1: Check if stream is closed
            if hasattr(wsgi_input, 'stream'):
                stream = wsgi_input.stream
                if hasattr(stream, 'closed') and stream.closed:
                    return False
                # Check for raw socket
                if hasattr(stream, 'raw') and hasattr(stream.raw, 'closed'):
                    if stream.raw.closed:
                        return False

            # Method 2: Check werkzeug's response context
            if hasattr(request, 'is_multiprocess'):
                # In development server, assume connected
                # Timeout will handle abandoned requests
                return True

            # Default: assume connected (rely on timeout)
            return True
        else:
            # Linux/Mac: use select for reliable socket detection
            if hasattr(wsgi_input, '_sock'):
                readable, _, _ = select.select([wsgi_input._sock], [], [], 0)
                if readable:
                    try:
                        data = wsgi_input._sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                        if not data:
                            return False
                    except (socket.error, OSError):
                        return False
            return True
    except Exception:
        # If any error occurs, assume client is still connected
        # to avoid false positives
        return True


def start_client_disconnect_monitor(yolo_instance, timeout_seconds: int = None):
    """
    Start a background thread that monitors client connection and cancels processing if disconnected.
    """
    if timeout_seconds is None:
        timeout_seconds = VIDEO_PROCESSING_TIMEOUT_SECONDS
    def monitor():
        start_time = time.time()
        while True:
            # Timeout check
            if time.time() - start_time > timeout_seconds:
                logger.warning("Video processing timeout - cancelling")
                yolo_instance.cancel_video_processing()
                break

            # Check client connection
            if not is_client_connected():
                logger.warning("Client disconnected - cancelling video processing")
                yolo_instance.cancel_video_processing()
                break

            # Check every 100ms
            time.sleep(0.1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread


# Template context processor for path utilities
@app.context_processor
def utility_processor():
    def normalize_path(path):
        """Normalize path for template use"""
        return normalize_static_path(path)
    return dict(normalize_path=normalize_path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


# ============================================================================
# 摄像头实时检测功能 - 已禁用
# 如需启用，将 CAMERA_FEATURE_ENABLED 改为 True
# ============================================================================
CAMERA_FEATURE_ENABLED = False


@app.route('/camera')
def camera():
    """摄像头检测页面"""
    if not CAMERA_FEATURE_ENABLED:
        abort(404)
    return render_template('camera.html')


@app.route('/camera/debug')
def camera_debug():
    """摄像头诊断工具页面"""
    if not CAMERA_FEATURE_ENABLED:
        abort(404)
    return render_template('camera_debug.html')


@app.route('/api/camera_detect', methods=['POST'])
@limiter.limit('1000/minute')  # 摄像头实时检测专用限制（覆盖默认限制）
def camera_detect():
    """接收摄像头帧进行推理的API接口"""
    if not CAMERA_FEATURE_ENABLED:
        return {'success': False, 'error': '功能已禁用'}, 404

    try:
        # 检查是否有图像数据
        if 'image' not in request.files:
            return {'success': False, 'error': '没有图像数据'}, 400

        file = request.files['image']
        if file.filename == '':
            return {'success': False, 'error': '没有选择文件'}, 400

        # 获取缩放比例（用于将检测框还原到原始尺寸）
        scale = 1.0
        if 'scale' in request.form:
            try:
                scale = float(request.form['scale'])
            except ValueError:
                scale = 1.0

        # 保存临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # 获取推理参数
            params = process_inference_parameters(request, config)

            # 切换模型
            if params['model_name'] != yolo_inference.model_path:
                yolo_inference.change_model(params['model_name'])

            # 设置阈值
            original_conf = yolo_inference.conf_threshold
            original_iou = yolo_inference.iou_threshold
            yolo_inference.conf_threshold = params['conf_threshold']
            yolo_inference.iou_threshold = params['iou_threshold']

            # 进行推理（不保存图像）
            import cv2
            img = cv2.imread(temp_path)
            if img is None:
                return {'success': False, 'error': '无法读取图像'}, 400

            # 执行检测
            results = yolo_inference.model.predict(
                img,
                conf=yolo_inference.conf_threshold,
                iou=yolo_inference.iou_threshold,
                verbose=False
            )

            # 处理结果
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes, 'cls') else 0

                        # 获取类别名称
                        class_name = yolo_inference.model.names[cls] if hasattr(yolo_inference.model, 'names') else f'class_{cls}'

                        # 将检测框坐标还原到原始尺寸
                        original_box = [float(x / scale) for x in box]

                        detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': original_box,
                            'class_id': cls
                        })

            # 恢复阈值
            yolo_inference.conf_threshold = original_conf
            yolo_inference.iou_threshold = original_iou

            return jsonify({
                'success': True,
                'detections': detections,
                'count': len(detections)
            })

        finally:
            # 清理临时文件
            try:
                import os
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        logger.error(f"Camera detection error: {str(e)}")
        return {'success': False, 'error': f'检测失败: {str(e)}'}, 500

# ============================================================================
# 摄像头功能代码结束
# ============================================================================


@app.route('/infer', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def infer():
    """Handle file upload and inference for web interface (supports single and multiple files)"""
    if request.method != 'POST':
        return redirect(url_for('home'))

    try:
        # Check if this is a multiple file upload
        files = request.files.getlist('files') if 'files' in request.files else []

        if len(files) == 0:
            # Try to get single file (backward compatibility)
            file = request.files.get('file')
            if file:
                files = [file]
            else:
                flash('没有选择文件', 'error')
                return redirect(url_for('home'))

        # If only one file and it's empty, skip
        if len(files) == 1 and files[0].filename == '':
            flash('没有选择文件', 'error')
            return redirect(url_for('home'))

        # Process multiple files
        if len(files) == 1:
            # Single file processing (backward compatibility)
            upload_result = secure_file_upload(files[0], config.UPLOAD_FOLDER)

            if not upload_result['success']:
                log_security_event('UPLOAD_FAILED', {
                    'filename': files[0].filename if files[0] else 'None',
                    'error': upload_result['error'],
                    'ip': get_remote_address()
                })
                flash(upload_result['error'], 'error')
                return redirect(url_for('home'))

            file_info = upload_result
            file_path = file_info['file_path']
            original_filename = file_info['filename']
            file_type = file_info.get('file_type', 'image')

            # Process and validate inference parameters
            params = process_inference_parameters(request, config)

            unique_filename = generate_unique_filename(original_filename)
            output_path = os.path.join(config.OUTPUT_FOLDER, unique_filename)

            try:
                if params['model_name'] != yolo_inference.model_path:
                    yolo_inference.change_model(params['model_name'])

                original_conf = yolo_inference.conf_threshold
                original_iou = yolo_inference.iou_threshold

                yolo_inference.conf_threshold = params['conf_threshold']
                yolo_inference.iou_threshold = params['iou_threshold']

                # Process based on file type
                if file_type == 'video':
                    # Start client disconnect monitoring thread
                    monitor_thread = start_client_disconnect_monitor(yolo_inference)

                    try:
                        result = yolo_inference.detect_video(
                            file_path,
                            output_path,
                            frame_skip=1,  # Process all frames to maintain full duration
                            max_frames=None  # No limit to maintain full video duration
                        )
                    except InterruptedError:
                        # Video processing was cancelled due to client disconnect
                        logger.info("Video processing cancelled due to client disconnect")
                        flash('视频处理已取消（客户端断开连接）', 'warning')
                        return redirect(url_for('home'))
                    finally:
                        # Ensure the monitor thread stops
                        monitor_thread.join(timeout=1)
                else:
                    result = yolo_inference.detect(file_path, output_path)

                yolo_inference.conf_threshold = original_conf
                yolo_inference.iou_threshold = original_iou

                # Render appropriate template
                if file_type == 'video':
                    return render_template('video_inference.html',
                                         input_file=file_path,
                                         output_video=result['output_video_path'],
                                         result=result)
                else:
                    return render_template('inference.html',
                                         saveLocation=file_path,
                                         output_image=output_path,
                                         result=result)

            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                return render_template('error.html', error_message=f"检测错误: {str(e)}"), 500

        else:
            # Multiple file processing
            upload_result = secure_multiple_files_upload(files, config.UPLOAD_FOLDER)

            if not upload_result['success']:
                log_security_event('BATCH_UPLOAD_FAILED', {
                    'error': upload_result['error'],
                    'ip': get_remote_address()
                })
                flash(upload_result['error'], 'error')
                return redirect(url_for('home'))

            uploaded_files = upload_result['uploaded_files']
            failed_files = upload_result['failed_files']

            if len(failed_files) > 0:
                logger.warning(f"Failed to upload {len(failed_files)} files")
                failed_names = []
                for failed in failed_files:
                    logger.warning(f"Failed: {failed['filename']} - {failed['error']}")
                    failed_names.append(f"{failed['filename']} ({failed['error']})")
                # Show user which files failed
                flash(f'{len(failed_files)} 个文件上传失败: {", ".join(failed_names[:5])}{"..." if len(failed_names) > 5 else ""}', 'warning')

            if len(uploaded_files) == 0:
                flash('没有成功上传的文件', 'error')
                return redirect(url_for('home'))

            # Process and validate inference parameters
            params = process_inference_parameters(request, config)

            # Separate image and video files
            image_files = [f for f in uploaded_files if f['file_type'] == 'image']
            video_files = [f for f in uploaded_files if f['file_type'] == 'video']

            # Create unique output directory for this batch
            import datetime
            batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_output_dir = os.path.join(config.OUTPUT_FOLDER, f'batch_{batch_id}')
            os.makedirs(batch_output_dir, exist_ok=True)

            results = []

            try:
                if params['model_name'] != yolo_inference.model_path:
                    yolo_inference.change_model(params['model_name'])

                original_conf = yolo_inference.conf_threshold
                original_iou = yolo_inference.iou_threshold

                yolo_inference.conf_threshold = params['conf_threshold']
                yolo_inference.iou_threshold = params['iou_threshold']

                # Process images
                if image_files:
                    image_paths = [f['file_path'] for f in image_files]
                    original_filenames = [f['original_filename'] for f in image_files]
                    image_output_dir = os.path.join(batch_output_dir, 'images')
                    os.makedirs(image_output_dir, exist_ok=True)

                    image_results = yolo_inference.detect_multiple_images(
                        image_paths, image_output_dir, original_filenames=original_filenames
                    )

                    results.extend(image_results.get('results', []))

                # Process videos
                if video_files:
                    for video_file in video_files:
                        video_output_dir = os.path.join(batch_output_dir, 'videos')
                        os.makedirs(video_output_dir, exist_ok=True)

                        video_name = os.path.splitext(video_file['original_filename'])[0]
                        video_output_path = os.path.join(video_output_dir, f"{video_name}_detected.mp4")

                        try:
                            # Start client disconnect monitoring thread
                            monitor_thread = start_client_disconnect_monitor(yolo_inference)

                            try:
                                video_result = yolo_inference.detect_video(
                                    video_file['file_path'],
                                    video_output_path,
                                    frame_skip=1,  # Process all frames to maintain full duration
                                    max_frames=None  # No limit to maintain full video duration
                                )
                            except InterruptedError:
                                # Video processing was cancelled due to client disconnect
                                logger.info("Batch video processing cancelled due to client disconnect")
                                raise
                            finally:
                                # Ensure the monitor thread stops
                                monitor_thread.join(timeout=1)

                            # Add batch info to video result
                            video_result['original_filename'] = video_file['original_filename']
                            video_result['batch_id'] = batch_id
                            results.append(video_result)

                        except Exception as e:
                            logger.error(f"Video processing error for {video_file['filename']}: {str(e)}")
                            results.append({
                                'error': str(e),
                                'original_filename': video_file['original_filename'],
                                'file_type': 'video',
                                'success': False
                            })

                yolo_inference.conf_threshold = original_conf
                yolo_inference.iou_threshold = original_iou

                # Prepare batch result
                batch_result = {
                    'batch_id': batch_id,
                    'summary': {
                        'total_files': len(uploaded_files),
                        'processed_files': len([r for r in results if r.get('success', True)]),
                        'failed_files': len([r for r in results if not r.get('success', True)]),
                        'image_count': len(image_files),
                        'video_count': len(video_files),
                        'model_used': params['model_name'],
                        'confidence_threshold': params['conf_threshold'],
                        'iou_threshold': params['iou_threshold']
                    },
                    'results': results,
                    'uploaded_files': uploaded_files
                }

                # Save results to file
                import json
                results_file = os.path.join(batch_output_dir, 'batch_results.json')
                with open(results_file, 'w') as f:
                    json.dump(batch_result, f, indent=2)

                return render_template('batch_inference.html',
                                     batch_result=batch_result,
                                     output_dir=batch_output_dir)

            except Exception as e:
                logger.error(f"Batch inference error: {str(e)}")
                return render_template('error.html', error_message=f"批量检测错误: {str(e)}"), 500

    except Exception as e:
        logger.error(f"Upload processing error: {str(e)}")
        return render_template('error.html', error_message=f"文件处理错误: {str(e)}"), 500


@app.route('/api/detect', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def api_detect():
    if 'file' not in request.files:
        return {'success': False, 'error': 'No file provided'}, 400

    file = request.files['file']
    upload_result = secure_file_upload(file, config.UPLOAD_FOLDER)

    if not upload_result['success']:
        log_security_event('API_UPLOAD_FAILED', {
            'filename': file.filename if file else 'None',
            'error': upload_result['error'],
            'ip': get_remote_address()
        })
        return {'success': False, 'error': upload_result['error']}, 400

    save_location = upload_result['file_path']
    original_filename = upload_result['filename']
    params = process_inference_parameters(request, config)

    unique_filename = generate_unique_filename(original_filename)
    output_image_path = os.path.join(config.OUTPUT_FOLDER, unique_filename)

    try:
        if params['model_name'] != yolo_inference.model_path:
            yolo_inference.change_model(params['model_name'])

        original_conf = yolo_inference.conf_threshold
        original_iou = yolo_inference.iou_threshold

        yolo_inference.conf_threshold = params['conf_threshold']
        yolo_inference.iou_threshold = params['iou_threshold']

        result = yolo_inference.detect(image_path=save_location, output_path=output_image_path)

        yolo_inference.conf_threshold = original_conf
        yolo_inference.iou_threshold = original_iou

        # NumpyJSONProvider handles numpy type conversion automatically
        return jsonify({
            'success': True,
            'result': result,
            'original_image': save_location,
            'output_image': output_image_path
        })

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return {'success': False, 'error': f'检测失败: {str(e)}'}, 500


@app.route('/api/models', methods=['GET'])
@limiter.exempt
def api_models():
    try:
        models_data = get_available_models()
        return {
            'success': True,
            'predefined_models': models_data['predefined_models'],
            'custom_models': models_data['custom_models'],
            'default_model': config.DEFAULT_MODEL,
            'max_file_size': config.MAX_CONTENT_LENGTH,
            'allowed_extensions': list(config.ALLOWED_EXTENSIONS),
            'supported_formats': ['pytorch', 'onnx', 'tensorrt']
        }
    except Exception as e:
        logger.error(f"Models error: {str(e)}")
        return {'success': False, 'error': 'Failed to get models'}, 500


@app.errorhandler(429)
def ratelimit_handler(e):
    log_security_event('RATE_LIMIT_EXCEEDED', {
        'ip': get_remote_address(),
        'description': str(e.description)
    })
    return {
        'success': False,
        'error': 'Rate limit exceeded. Please try again later.',
        'retry_after': e.description
    }, 429


@app.errorhandler(413)
def too_large(e):
    log_security_event('FILE_TOO_LARGE', {
        'ip': get_remote_address(),
        'max_size': config.MAX_CONTENT_LENGTH
    })
    return {
        'success': False,
        'error': f'File too large. Maximum size is {config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB.'
    }, 413


@app.route('/api/batch_upload', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def batch_upload():
    """Handle multiple file uploads separately from inference processing"""
    try:
        if 'files' not in request.files:
            return {'success': False, 'error': 'No files provided'}, 400

        files = request.files.getlist('files')
        if not files or all(not f.filename for f in files):
            return {'success': False, 'error': 'No valid files provided'}, 400

        # Upload files first
        upload_result = secure_multiple_files_upload(files, config.UPLOAD_FOLDER)

        if not upload_result['success']:
            return {'success': False, 'error': upload_result['error']}, 400

        uploaded_files = upload_result['uploaded_files']
        failed_files = upload_result['failed_files']

        if len(uploaded_files) == 0:
            return {'success': False, 'error': 'No files were successfully uploaded'}, 400

        # Generate a unique batch ID
        import datetime
        batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        # Store batch info in a temporary location (you could use Redis or database for production)
        batch_info = {
            'batch_id': batch_id,
            'uploaded_files': uploaded_files,
            'failed_files': failed_files,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Store batch info in a dedicated metadata directory
        import json
        os.makedirs(BATCH_META_DIR, exist_ok=True)
        batch_file_path = os.path.join(BATCH_META_DIR, f'batch_{batch_id}.json')
        with open(batch_file_path, 'w') as f:
            json.dump(batch_info, f)

        return {
            'success': True,
            'batch_id': batch_id,
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'uploaded_files_count': len(uploaded_files),
            'failed_files_count': len(failed_files)
        }

    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        return {'success': False, 'error': f'Upload failed: {str(e)}'}, 500


@app.route('/api/batch_inference', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def batch_inference():
    """Process inference for previously uploaded batch"""
    try:
        batch_id = request.json.get('batch_id')
        if not batch_id:
            return {'success': False, 'error': 'Batch ID required'}, 400

        # Get batch info from metadata directory
        batch_file_path = os.path.join(BATCH_META_DIR, f'batch_{batch_id}.json')
        if not os.path.exists(batch_file_path):
            return {'success': False, 'error': 'Invalid batch ID'}, 400

        import json
        with open(batch_file_path, 'r') as f:
            batch_info = json.load(f)

        uploaded_files = batch_info['uploaded_files']
        if not uploaded_files:
            return {'success': False, 'error': 'No files to process'}, 400

        # Process inference parameters
        params = process_inference_parameters(request, config)

        # Separate image and video files
        image_files = [f for f in uploaded_files if f['file_type'] == 'image']
        video_files = [f for f in uploaded_files if f['file_type'] == 'video']

        # Create unique output directory for this batch
        batch_output_dir = os.path.join(config.OUTPUT_FOLDER, f'batch_{batch_id}')
        os.makedirs(batch_output_dir, exist_ok=True)

        results = []

        try:
            if params['model_name'] != yolo_inference.model_path:
                yolo_inference.change_model(params['model_name'])

            original_conf = yolo_inference.conf_threshold
            original_iou = yolo_inference.iou_threshold

            yolo_inference.conf_threshold = params['conf_threshold']
            yolo_inference.iou_threshold = params['iou_threshold']

            # Process images
            if image_files:
                image_paths = [f['file_path'] for f in image_files]
                original_filenames = [f['original_filename'] for f in image_files]
                image_output_dir = os.path.join(batch_output_dir, 'images')
                os.makedirs(image_output_dir, exist_ok=True)

                image_results = yolo_inference.detect_multiple_images(
                    image_paths, image_output_dir, original_filenames=original_filenames
                )
                results.extend(image_results.get('results', []))

            # Process videos
            if video_files:
                for video_file in video_files:
                    video_output_dir = os.path.join(batch_output_dir, 'videos')
                    os.makedirs(video_output_dir, exist_ok=True)

                    video_name = os.path.splitext(video_file['original_filename'])[0]
                    video_output_path = os.path.join(video_output_dir, f"{video_name}_detected.mp4")

                    try:
                        # Start client disconnect monitoring thread
                        monitor_thread = start_client_disconnect_monitor(yolo_inference)

                        try:
                            video_result = yolo_inference.detect_video(
                                video_file['file_path'],
                                video_output_path,
                                frame_skip=1,  # Process all frames to maintain full duration
                                max_frames=None  # No limit to maintain full video duration
                            )
                        except InterruptedError:
                            # Video processing was cancelled due to client disconnect
                            logger.info("Batch API video processing cancelled due to client disconnect")
                            results.append({
                                'error': 'Processing cancelled - client disconnected',
                                'original_filename': video_file['original_filename'],
                                'file_type': 'video',
                                'success': False
                            })
                            continue
                        finally:
                            # Ensure the monitor thread stops
                            monitor_thread.join(timeout=1)

                        video_result['original_filename'] = video_file['original_filename']
                        video_result['batch_id'] = batch_id
                        video_result['success'] = True
                        results.append(video_result)

                    except Exception as e:
                        logger.error(f"Video processing error for {video_file['original_filename']}: {str(e)}")
                        results.append({
                            'error': str(e),
                            'original_filename': video_file['original_filename'],
                            'file_type': 'video',
                            'success': False
                        })

            yolo_inference.conf_threshold = original_conf
            yolo_inference.iou_threshold = original_iou

            # Prepare batch result
            batch_result = {
                'batch_id': batch_id,
                'summary': {
                    'total_files': len(uploaded_files),
                    'processed_files': len([r for r in results if r.get('success', True)]),
                    'failed_files': len([r for r in results if not r.get('success', True)]),
                    'image_count': len(image_files),
                    'video_count': len(video_files),
                    'model_used': params['model_name'],
                    'confidence_threshold': params['conf_threshold'],
                    'iou_threshold': params['iou_threshold']
                },
                'results': results,
                'uploaded_files': uploaded_files
            }

            # Save results to file
            import json
            results_file = os.path.join(batch_output_dir, 'batch_results.json')
            with open(results_file, 'w') as f:
                json.dump(batch_result, f, indent=2)

            return {
                'success': True,
                'batch_result': batch_result,
                'output_dir': batch_output_dir
            }

        except Exception as e:
            logger.error(f"Batch inference error: {str(e)}")
            return {'success': False, 'error': f"Inference failed: {str(e)}"}, 500

    except Exception as e:
        logger.error(f"Batch inference API error: {str(e)}")
        return {'success': False, 'error': f"Processing failed: {str(e)}"}, 500


@app.route('/batch_results')
def batch_results():
    """Display batch inference results"""
    batch_id = request.args.get('batch_id')
    if not batch_id:
        return render_template('error.html', error_message='Batch ID not provided'), 400

    try:
        # Get batch info from metadata directory
        batch_file_path = os.path.join(BATCH_META_DIR, f'batch_{batch_id}.json')
        if not os.path.exists(batch_file_path):
            return render_template('error.html', error_message='Invalid batch ID'), 404

        import json
        with open(batch_file_path, 'r') as f:
            batch_info = json.load(f)

        # Load results from the batch inference (if available)
        output_dir = os.path.join(config.OUTPUT_FOLDER, f'batch_{batch_id}')
        results_file = os.path.join(output_dir, 'batch_results.json')

        batch_result = None
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                batch_result = json.load(f)

        return render_template('batch_results.html',
                             batch_id=batch_id,
                             batch_info=batch_info,
                             batch_result=batch_result,
                             output_dir=output_dir)

    except Exception as e:
        logger.error(f"Batch results error: {str(e)}")
        return render_template('error.html', error_message=f"Failed to load batch results: {str(e)}"), 500


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {str(e)}")
    return {
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }, 500


# ============================================
# 异步推理API路由
# ============================================

@app.route('/infer/async', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def infer_async():
    """异步推理入口 - 立即返回任务ID，后台处理推理"""
    try:
        # 检查文件
        files = request.files.getlist('files') if 'files' in request.files else []
        if len(files) == 0:
            file = request.files.get('file')
            if file:
                files = [file]

        if not files or files[0].filename == '':
            return {'success': False, 'error': '没有选择文件'}, 400

        # 处理单个文件上传
        upload_result = secure_file_upload(files[0], config.UPLOAD_FOLDER)
        if not upload_result['success']:
            return {'success': False, 'error': upload_result['error']}, 400

        file_path = upload_result['file_path']
        original_filename = upload_result['filename']
        file_type = upload_result.get('file_type', 'image')

        # 获取推理参数
        params = process_inference_parameters(request, config)

        # 创建任务
        task_type = 'video' if file_type == 'video' else 'image'
        task_id = create_task(task_type, {
            'filename': original_filename,
            'file_path': file_path,
            'file_type': file_type
        })

        # 生成输出路径
        unique_filename = generate_unique_filename(original_filename)
        output_path = os.path.join(config.OUTPUT_FOLDER, unique_filename)

        # 启动后台推理任务
        def run_inference():
            try:
                update_task_progress(task_id, 5, '正在加载模型...', 'processing')

                # 切换模型
                if params['model_name'] != yolo_inference.model_path:
                    yolo_inference.change_model(params['model_name'])

                original_conf = yolo_inference.conf_threshold
                original_iou = yolo_inference.iou_threshold
                yolo_inference.conf_threshold = params['conf_threshold']
                yolo_inference.iou_threshold = params['iou_threshold']

                if file_type == 'video':
                    update_task_progress(task_id, 10, '正在处理视频...', 'processing')

                    # 创建进度回调
                    def progress_callback(progress, frame, total, detections):
                        percent = 10 + int((progress / 100) * 80)
                        update_task_progress(task_id, percent, f'处理中: {frame}/{total} 帧')

                    result = yolo_inference.detect_video(
                        file_path,
                        output_path,
                        frame_skip=1,
                        max_frames=None,
                        progress_callback=progress_callback
                    )
                else:
                    update_task_progress(task_id, 20, '正在进行目标检测...', 'processing')
                    result = yolo_inference.detect(file_path, output_path)
                    update_task_progress(task_id, 90, '正在生成结果...', 'processing')

                yolo_inference.conf_threshold = original_conf
                yolo_inference.iou_threshold = original_iou

                # 任务完成
                complete_task(task_id, {
                    'result': result,
                    'output_path': output_path,
                    'input_path': file_path,
                    'original_filename': original_filename,
                    'file_type': file_type
                })

            except InterruptedError:
                fail_task(task_id, '任务已被取消')
            except Exception as e:
                logger.error(f"Async inference error: {str(e)}")
                fail_task(task_id, str(e))

        # 启动后台线程
        thread = threading.Thread(target=run_inference, daemon=True)
        thread.start()

        # 立即返回任务ID
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '任务已创建，正在后台处理'
        })

    except Exception as e:
        logger.error(f"Async infer error: {str(e)}")
        return {'success': False, 'error': str(e)}, 500


@app.route('/api/task/<task_id>', methods=['GET'])
@limiter.limit('60/minute')
def get_task_status(task_id):
    """查询任务状态"""
    task = get_task(task_id)

    if not task:
        return {'success': False, 'error': '任务不存在'}, 404

    return jsonify({
        'success': True,
        'task': {
            'id': task['id'],
            'type': task['type'],
            'status': task['status'],
            'progress': task['progress'],
            'message': task['message'],
            'file_info': task['file_info'],
            'created_at': task['created_at']
        }
    })


@app.route('/infer/waiting/<task_id>')
def infer_waiting(task_id):
    """异步推理等待页面"""
    task = get_task(task_id)

    if not task:
        return render_template('error.html', error_message='任务不存在'), 404

    return render_template('infer_waiting.html',
                         task_id=task_id,
                         task=task,
                         file_type=task['type'])


@app.route('/infer/result/<task_id>')
def infer_result(task_id):
    """异步推理结果页面"""
    task = get_task(task_id)

    if not task:
        return render_template('error.html', error_message='任务不存在'), 404

    if task['status'] == 'pending' or task['status'] == 'processing':
        return render_template('infer_waiting.html',
                             task_id=task_id,
                             task=task,
                             file_type=task['type'])

    if task['status'] == 'failed':
        error_msg = task.get('error', '处理失败，请重试')
        return render_template('error.html', error_message=error_msg), 500

    if task['status'] == 'completed':
        result = task['result']
        if result['file_type'] == 'video':
            # 构建 result 数据结构，确保与模板兼容
            video_result = result.get('result', {})
            video_result['output_video_path'] = result.get('output_path', '')

            return render_template('video_inference.html',
                                 input_file=result.get('input_path', ''),
                                 result=video_result)
        else:
            return render_template('inference.html',
                                 saveLocation=result['input_path'],
                                 output_image=result['output_path'],
                                 result=result['result'])

    return render_template('error.html', error_message='未知任务状态'), 500


# ============================================
# 优化的视频流式传输
# ============================================

def send_video_partial(file_path, mimetype='video/mp4'):
    """
    优化的视频传输函数，支持HTTP Range请求和缓存
    显著提高视频加载速度
    """
    from flask import Response
    import os

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)

    if range_header:
        # 解析Range头 (格式: "bytes=start-end")
        byte_range = range_header.replace('bytes=', '').split('-')
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1

        # 限制范围
        if start >= file_size or end >= file_size:
            return Response('Requested Range Not Satisfiable', status=416)

        chunk_size = end - start + 1

        def generate():
            with open(file_path, 'rb') as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    chunk_size_read = min(64 * 1024, remaining)  # 64KB chunks
                    data = f.read(chunk_size_read)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        response = Response(
            generate(),
            206,  # Partial Content
            mimetype=mimetype,
            direct_passthrough=True
        )
        response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(chunk_size))
    else:
        # 完整文件传输
        def generate():
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(64 * 1024)  # 64KB chunks
                    if not chunk:
                        break
                    yield chunk

        response = Response(
            generate(),
            200,
            mimetype=mimetype,
            direct_passthrough=True
        )
        response.headers.add('Content-Length', str(file_size))
        response.headers.add('Accept-Ranges', 'bytes')

    # 添加缓存头 - 24小时缓存
    response.headers.add('Cache-Control', 'public, max-age=86400, immutable')
    # 添加ETag支持条件请求
    response.headers.add('ETag', f'"{os.path.getmtime(file_path)}-{file_size}"')

    return response


# Enhanced static file serving for videos
@app.route('/static/<path:filename>')
def custom_static(filename):
    """Enhanced static file serving with better video support and caching"""
    from flask import send_from_directory
    import mimetypes
    import os

    file_path = os.path.join('static', filename)

    # Check if file exists
    if not os.path.exists(file_path):
        return {'success': False, 'error': 'File not found'}, 404

    # Determine MIME type
    mimetype, _ = mimetypes.guess_type(filename)

    # For video files, use optimized streaming
    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        if mimetype is None or not mimetype.startswith('video/'):
            mimetype = 'video/mp4'
        return send_video_partial(file_path, mimetype)

    # For other files, use standard serving
    if mimetype is None:
        mimetype = 'application/octet-stream'

    return send_from_directory('static', filename, mimetype=mimetype)