from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import json
import numpy as np
from model_inference import yolo_inference, get_available_models
from run import get_config
from utils import secure_file_upload, secure_multiple_files_upload, log_security_event, setup_app_logging, process_inference_parameters, generate_unique_filename, normalize_static_path
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
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

# Set custom JSON encoder
app.json_encoder = NumpyJSONEncoder

# Load configuration
config = get_config()
app.config.from_object(config)

# Set up logging, directories, and cleanup using utils
logger = setup_app_logging(config)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[config.RATELIMIT_DEFAULT]
)

logger.info("YOLO Web Demo started")


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
                    result = yolo_inference.detect_video(
                        file_path,
                        output_path,
                        frame_skip=3,  # Skip frames for reasonable processing time
                        max_frames=600  # Limit total frames for performance
                    )
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
                for failed in failed_files:
                    logger.warning(f"Failed: {failed['filename']} - {failed['error']}")

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
                            video_result = yolo_inference.detect_video(
                                video_file['file_path'],
                                video_output_path,
                                frame_skip=3,  # Consistent with other video processing
                                max_frames=600  # Consistent with other video processing
                            )

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

        # Manually convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        result = convert_numpy(result)

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

        # Store batch info in a temporary JSON file
        import json
        batch_file_path = os.path.join(config.UPLOAD_FOLDER, f'batch_{batch_id}.json')
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

        # Get batch info from storage
        batch_file_path = os.path.join(config.UPLOAD_FOLDER, f'batch_{batch_id}.json')
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
                        video_result = yolo_inference.detect_video(
                            video_file['file_path'],
                            video_output_path,
                            frame_skip=3,  # Same as single video processing
                            max_frames=600  # Same as single video processing
                        )

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
        # Get batch info from storage
        batch_file_path = os.path.join(config.UPLOAD_FOLDER, f'batch_{batch_id}.json')
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