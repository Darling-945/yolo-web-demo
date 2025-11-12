from flask import Flask, render_template, request, redirect, url_for, flash
import os
from model_inference import yolo_inference, get_available_models
from run import get_config
from utils import (
    secure_file_upload, log_security_event, setup_app_logging,
    process_inference_parameters, generate_unique_filename
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from typing import Optional

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

logger.info(f"YOLO Web Demo started with config: {config.__class__.__name__}")
logger.info(f"Upload folder: {config.UPLOAD_FOLDER}")
logger.info(f"Output folder: {config.OUTPUT_FOLDER}")
logger.info(f"Max file size: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def infer():
    """Handle image upload and inference for web interface"""
    if request.method != 'POST':
        return redirect(url_for('home'))

    try:
        # Handle file upload securely
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

        # Process and validate inference parameters
        params = process_inference_parameters(request, config)
        model_name = params['model_name']
        conf_threshold = params['conf_threshold']
        iou_threshold = params['iou_threshold']

        # Generate unique filename for output image
        unique_filename = generate_unique_filename(original_filename)
        output_image_path = os.path.join(config.OUTPUT_FOLDER, unique_filename)

        try:
            logger.info(f"Starting inference: {original_filename} with model {model_name}")

            # Update model if different from current one
            if model_name != yolo_inference.model_path:
                yolo_inference.change_model(model_name)
                logger.info(f"Changed model to: {model_name}")

            # Store original thresholds
            original_conf = yolo_inference.conf_threshold
            original_iou = yolo_inference.iou_threshold

            # Update thresholds temporarily
            yolo_inference.conf_threshold = conf_threshold
            yolo_inference.iou_threshold = iou_threshold

            # Perform object detection
            result = yolo_inference.detect(
                image_path=save_location,
                output_path=output_image_path
            )

            # Restore original thresholds
            yolo_inference.conf_threshold = original_conf
            yolo_inference.iou_threshold = original_iou

            logger.info(f"Inference completed: {len(result.get('detections', []))} objects detected")

            return render_template(
                'inference.html',
                saveLocation=save_location,
                output_image=output_image_path,
                result=result
            )

        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            error_message = f"检测过程中发生错误: {str(e)}"
            return render_template('error.html', error_message=error_message), 500

    except Exception as e:
        logger.error(f"Upload processing error: {str(e)}")
        error_message = f"文件处理过程中发生错误: {str(e)}"
        return render_template('error.html', error_message=error_message), 500


@app.route('/api/detect', methods=['POST'])
@limiter.limit(config.RATELIMIT_API)
def api_detect():
    """API endpoint for object detection"""
    if 'file' not in request.files:
        return {'success': False, 'error': 'No file provided'}, 400

    file = request.files['file']

    # Handle file upload securely
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

    # Process and validate inference parameters
    params = process_inference_parameters(request, config)
    model_name = params['model_name']
    conf_threshold = params['conf_threshold']
    iou_threshold = params['iou_threshold']

    # Generate unique filename for output image
    unique_filename = generate_unique_filename(original_filename)
    output_image_path = os.path.join(config.OUTPUT_FOLDER, unique_filename)

    try:
        logger.info(f"API inference: {original_filename} with model {model_name}")

        # Update model if different from current one
        if model_name != yolo_inference.model_path:
            yolo_inference.change_model(model_name)
            logger.info(f"API changed model to: {model_name}")

        # Store original thresholds
        original_conf = yolo_inference.conf_threshold
        original_iou = yolo_inference.iou_threshold

        # Update thresholds temporarily
        yolo_inference.conf_threshold = conf_threshold
        yolo_inference.iou_threshold = iou_threshold

        # Perform object detection
        result = yolo_inference.detect(
            image_path=save_location,
            output_path=output_image_path
        )

        # Restore original thresholds
        yolo_inference.conf_threshold = original_conf
        yolo_inference.iou_threshold = original_iou

        logger.info(f"API inference completed: {len(result.get('detections', []))} objects detected")

        return {
            'success': True,
            'result': result,
            'original_image': save_location,
            'output_image': output_image_path
        }

    except Exception as e:
        logger.error(f"API inference error: {str(e)}")
        return {'success': False, 'error': f'检测失败: {str(e)}'}, 500


@app.route('/api/models', methods=['GET'])
@limiter.exempt  # No rate limiting for models endpoint (read-only)
def api_models():
    """API endpoint to get available models"""
    try:
        models_data = get_available_models()
        return {
            'success': True,
            'predefined_models': models_data['predefined_models'],
            'custom_models': models_data['custom_models'],
            'default_model': config.DEFAULT_MODEL,
            'max_file_size': config.MAX_CONTENT_LENGTH,
            'allowed_extensions': list(config.ALLOWED_EXTENSIONS)
        }
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        return {'success': False, 'error': 'Failed to get models'}, 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded errors"""
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
    """Handle file too large errors"""
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
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return {
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }, 500


if __name__ == "__main__":
    logger.info("Starting YOLO Web Demo...")
    logger.info(f"Environment: {config.__class__.__name__}")
    logger.info(f"Debug mode: {config.DEBUG}")

    if config.DEBUG:
        logger.warning("Debug mode is enabled. This should not be used in production!")

    # Get host and port from environment or use defaults
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))

    logger.info(f"Server will be available at: http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")

    try:
        app.run(
            host=host,
            port=port,
            debug=config.DEBUG,
            use_reloader=config.DEBUG  # Only use reloader in debug mode
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")