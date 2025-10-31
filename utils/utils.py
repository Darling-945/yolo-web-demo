"""
Utility functions for YOLO Web Demo
Includes security, file validation, and cleanup utilities
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from werkzeug.utils import secure_filename
from PIL import Image
from run import get_config

# Set up logging
config_instance = get_config()
log_level = getattr(logging, config_instance.LOG_LEVEL)
log_file = config_instance.LOG_FILE

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_valid_image_file(file_path: str) -> bool:
    """
    Validate if a file is actually an image using PIL with robust error handling

    Args:
        file_path (str): Path to the file to validate

    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return False

        # Check file size (prevent empty files and extremely large files)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"File is empty: {file_path}")
            return False

        # Check for reasonable file size (max 50MB)
        if file_size > 50 * 1024 * 1024:
            logger.warning(f"File too large: {file_size} bytes for {file_path}")
            return False

        # Check file extension first for basic validation (case-insensitive)
        filename = os.path.basename(file_path)
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}

        has_valid_extension = any(filename.lower().endswith(ext) for ext in allowed_extensions)
        if not has_valid_extension:
            logger.warning(f"Invalid file extension for file: {file_path}")
            return False

        # Multiple validation strategies
        validation_methods = [
            _validate_with_pil_standard,
            _validate_with_pil_force_load,
            _validate_with_opencv_fallback
        ]

        for method_name, validation_method in [
            ("PIL Standard", _validate_with_pil_standard),
            ("PIL Force Load", _validate_with_pil_force_load),
            ("OpenCV Fallback", _validate_with_opencv_fallback)
        ]:
            try:
                logger.debug(f"Trying {method_name} validation for {file_path}")
                if validation_method(file_path):
                    logger.info(f"Successfully validated image: {file_path} (using {method_name})")
                    return True
                else:
                    logger.debug(f"{method_name} validation failed for {file_path}")
                    continue
            except Exception as e:
                logger.debug(f"{method_name} validation error for {file_path}: {str(e)}")
                continue

        # If all methods fail, try file header analysis as last resort
        if _validate_by_file_header(file_path):
            logger.warning(f"Image validated by file header only (may have issues): {file_path}")
            return True

        logger.error(f"All validation methods failed for {file_path}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error validating file {file_path}: {str(e)}")
        return False


def _validate_with_pil_standard(file_path: str) -> bool:
    """Standard PIL validation"""
    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except Exception:
        return False


def _validate_with_pil_force_load(file_path: str) -> bool:
    """PIL validation with forced loading"""
    try:
        with Image.open(file_path) as img:
            img.load()  # Force load the image data

            # Check image properties
            if img.width <= 0 or img.height <= 0:
                return False

            # Try to convert to RGB to ensure it's a valid image
            rgb_img = img.convert('RGB')
            return True
    except Exception as e:
        logger.debug(f"PIL force load failed: {str(e)}")
        return False


def _validate_with_opencv_fallback(file_path: str) -> bool:
    """OpenCV fallback validation"""
    try:
        import cv2
        img = cv2.imread(file_path)
        if img is None:
            return False

        # Check if image has valid dimensions
        if img.shape[0] <= 0 or img.shape[1] <= 0:
            return False

        return True
    except ImportError:
        logger.debug("OpenCV not available for fallback validation")
        return False
    except Exception as e:
        logger.debug(f"OpenCV validation failed: {str(e)}")
        return False


def _validate_by_file_header(file_path: str) -> bool:
    """Validate image by checking file header bytes"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)

        # Common image file signatures
        image_signatures = {
            b'\xFF\xD8\xFF': 'JPEG',  # JPEG
            b'\x89PNG\r\n\x1a\n': 'PNG',  # PNG
            b'GIF87a': 'GIF87a',  # GIF87a
            b'GIF89a': 'GIF89a',  # GIF89a
            b'BM': 'BMP',  # BMP
            b'II*\x00': 'TIFF',  # TIFF (little-endian)
            b'MM\x00*': 'TIFF',  # TIFF (big-endian)
            b'RIFF': 'WEBP',  # WEBP (RIFF container)
        }

        for signature, format_name in image_signatures.items():
            if header.startswith(signature):
                logger.debug(f"File header matches {format_name} format")
                return True

        return False
    except Exception as e:
        logger.debug(f"File header validation failed: {str(e)}")
        return False


def secure_file_upload(file, upload_folder: str) -> Dict[str, Any]:
    """
    Securely handle file upload with comprehensive validation and error handling

    Args:
        file: File object from request
        upload_folder (str): Directory to save uploaded file

    Returns:
        dict: Result with success status and file path or error message
    """
    try:
        # Check if file exists and has filename
        if not file or not file.filename:
            return {'success': False, 'error': 'No file provided'}

        original_filename = file.filename
        logger.info(f"Processing file upload: {original_filename}")

        # Check file extension
        filename = secure_filename(original_filename)
        if not filename:
            logger.error(f"Invalid filename after secure_filename: {original_filename}")
            return {'success': False, 'error': 'Invalid filename'}

        # Check file extension (case-insensitive)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        allowed_extensions = {ext.lower() for ext in get_config().ALLOWED_EXTENSIONS}
        if file_ext not in allowed_extensions:
            logger.warning(f"Invalid file extension: {file_ext} for file: {original_filename}")
            return {
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(sorted(allowed_extensions))}'
            }

        # Create upload directory if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)

        # Generate unique filename first to avoid conflicts
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        final_path = os.path.join(upload_folder, unique_filename)

        logger.debug(f"Saving file to: {final_path}")

        # Save file with comprehensive error handling
        try:
            # Reset file pointer to beginning
            file.seek(0)
            file.save(final_path)

            # Verify file was saved correctly
            if not os.path.exists(final_path):
                logger.error(f"File was not saved: {final_path}")
                return {'success': False, 'error': 'File save failed - file not found after save'}

            saved_size = os.path.getsize(final_path)
            logger.info(f"File saved successfully: {final_path} ({saved_size} bytes)")

        except Exception as save_error:
            logger.error(f"Failed to save file: {str(save_error)}")
            return {'success': False, 'error': f'Failed to save file: {str(save_error)}'}

        # Validate the file is actually an image after saving
        logger.debug(f"Validating image file: {final_path}")
        if not is_valid_image_file(final_path):
            logger.warning(f"Image validation failed for: {final_path}")
            try:
                os.remove(final_path)  # Remove invalid file
                logger.warning(f"Removed invalid image file: {final_path}")
            except Exception as remove_error:
                logger.error(f"Failed to remove invalid file: {str(remove_error)}")

            # Provide more detailed error information
            return {
                'success': False,
                'error': 'Invalid image file. The file may be corrupted, in an unsupported format, or contain invalid metadata.'
            }

        logger.info(f"File upload successful: {unique_filename}")
        return {
            'success': True,
            'filename': unique_filename,
            'file_path': final_path
        }

    except Exception as e:
        logger.error(f"Error during secure file upload: {str(e)}", exc_info=True)
        return {'success': False, 'error': f'Upload failed: {str(e)}'}


def cleanup_old_files(directory: str, max_age_seconds: int) -> int:
    """
    Remove files older than max_age_seconds from directory

    Args:
        directory (str): Directory to clean
        max_age_seconds (int): Maximum age in seconds

    Returns:
        int: Number of files removed
    """
    try:
        if not os.path.exists(directory):
            return 0

        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        removed_count = 0

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)

                if file_mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        logger.info(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove file {file_path}: {str(e)}")

        return removed_count

    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {str(e)}")
        return 0


def schedule_file_cleanup():
    """Clean up old files in upload and output directories"""
    cfg = get_config()

    upload_removed = cleanup_old_files(cfg.UPLOAD_FOLDER, cfg.MAX_FILE_AGE)
    output_removed = cleanup_old_files(cfg.OUTPUT_FOLDER, cfg.MAX_FILE_AGE)

    if upload_removed > 0 or output_removed > 0:
        logger.info(f"Cleanup completed. Removed {upload_removed} upload files and {output_removed} output files.")


def log_security_event(event_type: str, details: Dict[str, Any]):
    """
    Log security-related events

    Args:
        event_type (str): Type of security event
        details (dict): Event details
    """
    logger.warning(f"SECURITY EVENT - {event_type}: {details}")


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file

    Args:
        file_path (str): Path to file

    Returns:
        dict: File information
    """
    try:
        if not os.path.exists(file_path):
            return {}

        stat = os.stat(file_path)

        # Use PIL to determine image type, fallback to extension
        try:
            with Image.open(file_path) as img:
                format_name = img.format.lower() if img.format else None
                if format_name:
                    mime_type = f"image/{format_name}"
                else:
                    raise Exception("No format detected")
        except Exception:
            # Fallback to extension-based detection
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
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return {}


def setup_app_logging(config):
    """
    Set up logging configuration for the Flask app

    Args:
        config: Configuration object with logging settings
    """
    import logging
    from threading import Timer

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

    # Create necessary directories
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

    # Schedule periodic file cleanup
    def schedule_periodic_cleanup():
        """Schedule periodic cleanup of old files"""
        schedule_file_cleanup()
        Timer(config.CLEANUP_INTERVAL, schedule_periodic_cleanup).start()

    # Start cleanup timer
    schedule_periodic_cleanup()

    return logging.getLogger(__name__)


def process_inference_parameters(request, config):
    """
    Process and validate inference parameters from request

    Args:
        request: Flask request object
        config: Configuration object

    Returns:
        dict: Validated parameters
    """
    # Get model name
    model_name = request.form.get('model', config.DEFAULT_MODEL)

    # Validate confidence threshold
    try:
        conf_threshold = float(request.form.get('confidence', config.DEFAULT_CONFIDENCE))
        conf_threshold = max(0.0, min(1.0, conf_threshold))
    except ValueError:
        conf_threshold = config.DEFAULT_CONFIDENCE

    # Validate IOU threshold
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
    """
    Generate unique filename for output image

    Args:
        original_filename (str): Original filename

    Returns:
        str: Unique filename
    """
    from uuid import uuid4
    return f"{uuid4()}_{original_filename}"