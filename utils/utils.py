"""
Utility functions for YOLO Web Demo
"""
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any
from werkzeug.utils import secure_filename
from PIL import Image
from run import get_config

config_instance = get_config()
os.makedirs(os.path.dirname(config_instance.LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config_instance.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config_instance.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_valid_image_file(file_path: str) -> bool:
    """Validate if a file is actually an image"""
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        if file_size == 0 or file_size > 50 * 1024 * 1024:
            return False

        filename = os.path.basename(file_path).lower()
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return False

        try:
            with Image.open(file_path) as img:
                img.verify()
                return True
        except:
            try:
                with Image.open(file_path) as img:
                    img.load()
                    return True
            except:
                return False

    except Exception as e:
        logger.error(f"Validation error for {file_path}: {str(e)}")
        return False


def secure_file_upload(file, upload_folder: str) -> Dict[str, Any]:
    """Handle file upload securely"""
    try:
        if not file or not file.filename:
            return {'success': False, 'error': 'No file provided'}

        filename = secure_filename(file.filename)
        if not filename:
            return {'success': False, 'error': 'Invalid filename'}

        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        allowed_extensions = {ext.lower() for ext in get_config().ALLOWED_EXTENSIONS}
        if file_ext not in allowed_extensions:
            return {'success': False, 'error': f'Invalid file type: {file_ext}'}

        os.makedirs(upload_folder, exist_ok=True)
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        final_path = os.path.join(upload_folder, unique_filename)

        file.seek(0)
        file.save(final_path)

        if not os.path.exists(final_path):
            return {'success': False, 'error': 'File save failed'}

        if not is_valid_image_file(final_path):
            os.remove(final_path)
            return {'success': False, 'error': 'Invalid image file'}

        return {
            'success': True,
            'filename': unique_filename,
            'file_path': final_path
        }

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return {'success': False, 'error': f'Upload failed: {str(e)}'}


def cleanup_old_files(directory: str, max_age_seconds: int) -> int:
    """Remove files older than max_age_seconds from directory"""
    try:
        if not os.path.exists(directory):
            return 0

        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        removed_count = 0

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {str(e)}")

        return removed_count
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return 0


def schedule_file_cleanup():
    """Clean up old files in upload and output directories"""
    cfg = get_config()
    upload_removed = cleanup_old_files(cfg.UPLOAD_FOLDER, cfg.MAX_FILE_AGE)
    output_removed = cleanup_old_files(cfg.OUTPUT_FOLDER, cfg.MAX_FILE_AGE)
    if upload_removed > 0 or output_removed > 0:
        logger.info(f"Cleanup: {upload_removed} upload, {output_removed} output files removed")


def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security-related events"""
    logger.warning(f"SECURITY EVENT - {event_type}: {details}")


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file"""
    try:
        if not os.path.exists(file_path):
            return {}

        stat = os.stat(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        mime_type = mime_map.get(ext, 'unknown')

        return {
            'filename': os.path.basename(file_path),
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'mime_type': mime_type
        }
    except Exception as e:
        logger.error(f"File info error: {str(e)}")
        return {}


def setup_app_logging(config):
    """Set up logging configuration for the Flask app"""
    import logging
    from threading import Timer

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

    def schedule_periodic_cleanup():
        schedule_file_cleanup()
        Timer(config.CLEANUP_INTERVAL, schedule_periodic_cleanup).start()

    schedule_periodic_cleanup()
    return logging.getLogger(__name__)


def process_inference_parameters(request, config):
    """Process and validate inference parameters from request"""
    model_name = request.form.get('model', config.DEFAULT_MODEL)

    try:
        conf_threshold = float(request.form.get('confidence', config.DEFAULT_CONFIDENCE))
        conf_threshold = max(0.0, min(1.0, conf_threshold))
    except ValueError:
        conf_threshold = config.DEFAULT_CONFIDENCE

    try:
        iou_threshold = float(request.form.get('iou', config.DEFAULT_IOU))
        iou_threshold = max(0.0, min(1.0, iou_threshold))
    except ValueError:
        iou_threshold = config.DEFAULT_IOU

    return {
        'model_name': model_name,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold
    }


def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename for output image"""
    from uuid import uuid4
    return f"{uuid4()}_{original_filename}"