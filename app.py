from flask import Flask, render_template, request, redirect, url_for, flash
import os
from model_inference import yolo_inference, get_available_models
from run import get_config
from utils import secure_file_upload, log_security_event, setup_app_logging, process_inference_parameters, generate_unique_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def infer():
    if request.method != 'POST':
        return redirect(url_for('home'))

    try:
        file = request.files.get('file')
        upload_result = secure_file_upload(file, config.UPLOAD_FOLDER)

        if not upload_result['success']:
            log_security_event('UPLOAD_FAILED', {
                'filename': file.filename if file else 'None',
                'error': upload_result['error'],
                'ip': get_remote_address()
            })
            flash(upload_result['error'], 'error')
            return redirect(url_for('home'))

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

            return render_template('inference.html', saveLocation=save_location,
                                 output_image=output_image_path, result=result)

        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return render_template('error.html', error_message=f"检测错误: {str(e)}"), 500

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return render_template('error.html', error_message=f"文件错误: {str(e)}"), 500


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

        return {
            'success': True,
            'result': result,
            'original_image': save_location,
            'output_image': output_image_path
        }

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


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {str(e)}")
    return {
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }, 500