"""
Model inference module for YOLO object detection
Supports YOLOv5, YOLOv8, and YOLOv11 models through Ultralytics
Extended support for ONNX and TensorRT model formats
Multi-image and video processing capabilities
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import gc
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
from contextlib import contextmanager
from threading import Lock, Event

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model cache for preloading
MODEL_CACHE = {}
CACHE_LOCK = Lock()


def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class YOLOInference:
    """
    YOLO Inference class to handle object detection
    Supports PyTorch (.pt), ONNX (.onnx), and TensorRT (.engine) models
    """
    def __init__(self, model_path: str = 'yolo11n.pt', conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the YOLO model

        Args:
            model_path (str): Path to YOLO model file or model name (.pt, .onnx, .engine)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.model_format = self._detect_model_format(model_path)
        self._stop_event = Event()
        self.load_model()

    def _detect_model_format(self, model_path: str) -> str:
        """Detect model format from file extension"""
        if model_path.endswith('.onnx'):
            return 'onnx'
        elif model_path.endswith('.engine'):
            return 'tensorrt'
        else:
            return 'pytorch'  # .pt files or default

    def load_model(self):
        """Load the YOLO model with format-specific optimizations"""
        try:
            # Check if model is a custom model in models directory
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            custom_model_path = os.path.join(models_dir, self.model_path)

            model_file = custom_model_path if os.path.exists(custom_model_path) else self.model_path

            # Format-specific loading
            if self.model_format == 'onnx':
                logger.info(f"Loading ONNX model: {self.model_path}")
                self.model = YOLO(model_file, task='detect')
                # ONNX optimizations
                logger.info("ONNX model loaded with CPU optimizations")
            elif self.model_format == 'tensorrt':
                logger.info(f"Loading TensorRT model: {self.model_path}")
                self.model = YOLO(model_file, task='detect')
                # TensorRT optimizations
                logger.info("TensorRT model loaded with GPU optimizations")
            else:  # PyTorch
                if os.path.exists(custom_model_path):
                    logger.info(f"Loading custom PyTorch model: {self.model_path}")
                else:
                    logger.info(f"Loading predefined PyTorch model: {self.model_path}")
                self.model = YOLO(model_file)

            logger.info(f"Model loaded successfully ({self.model_format} format)")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path} ({self.model_format}): {str(e)}")
            raise

    def cancel_video_processing(self):
        """Signal to stop the current video processing"""
        self._stop_event.set()
        logger.info("Video processing cancellation requested")

    def reset_stop_event(self):
        """Reset the stop event for a new processing task"""
        self._stop_event.clear()
        logger.debug("Stop event reset")

    def is_cancelled(self) -> bool:
        """Check if processing has been cancelled"""
        return self._stop_event.is_set()

    def detect(self, image_path: str, output_path: str) -> Dict:
        """
        Detect objects in an image and save the annotated result
        Optimized with explicit memory management

        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image with detections

        Returns:
            dict: Detection results including summary and details
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        img = None
        annotated_img = None
        results = None

        try:
            # Validate input image path
            if not os.path.exists(image_path):
                raise ValueError(f"Input image file does not exist: {image_path}")

            # Read the input image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path} - file may be corrupted or invalid format")

            # Get original image dimensions for reference
            img_height, img_width = img.shape[:2]
            logger.info(f"Processing image: {image_path} ({img_width}x{img_height})")

            # Perform inference with timing
            logger.info(f"Running inference with model {self.model_path}, conf={self.conf_threshold}, iou={self.iou_threshold}")
            inference_start_time = time.time()
            results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            logger.info(f"Inference completed in {inference_time:.4f} seconds")

            # Draw results on the image
            annotated_img = results[0].plot()

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the output image
            success = cv2.imwrite(output_path, annotated_img)
            if not success:
                raise ValueError(f"Failed to save output image to {output_path}")

            logger.info(f"Output image saved successfully: {output_path}")

            # Get detection information
            detections = []
            detection_summary = {}

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    class_name = self.model.names[class_id]

                    # Calculate normalized bounding box coordinates
                    bbox_normalized = [
                        bbox[0] / img_width,   # x1 normalized
                        bbox[1] / img_height,  # y1 normalized
                        bbox[2] / img_width,   # x2 normalized
                        bbox[3] / img_height   # y2 normalized
                    ]

                    detection_info = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': bbox.tolist(),  # Original pixel coordinates
                        'bbox_normalized': bbox_normalized,  # Normalized coordinates
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Area in pixels
                    }

                    detections.append(detection_info)

                    # Update detection summary
                    if class_name in detection_summary:
                        detection_summary[class_name] += 1
                    else:
                        detection_summary[class_name] = 1

            # Calculate summary statistics
            summary = {
                'total_detections': len(detections),
                'detection_summary': detection_summary,
                'model_used': self.model_path,
                'confidence_threshold': self.conf_threshold,
                'image_dimensions': {
                    'width': img_width,
                    'height': img_height
                },
                'inference_time': round(inference_time, 4),  # Add inference time in seconds
                'model_format': self.model_format
            }

            result = {
                'summary': summary,
                'detections': detections,
                'output_image_path': output_path
            }

            # Convert NumPy types to Python native types for JSON serialization
            result = convert_numpy_types(result)

            logger.info(f"Detection completed. Found {len(detections)} objects.")
            return result

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise
        finally:
            # Explicit memory cleanup
            if img is not None:
                del img
            if annotated_img is not None:
                del annotated_img
            if results is not None:
                del results
            gc.collect()

    def detect_and_return_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Detect objects in an image and return both the annotated image and results

        Args:
            image_path (str): Path to input image

        Returns:
            tuple: (annotated_image, detection_results)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Read the input image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Perform inference with timing
        inference_start_time = time.time()
        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
        inference_time = time.time() - inference_start_time

        # Draw results on the image
        annotated_img = results[0].plot()

        # Get detection information
        detections = []
        detection_summary = {}

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            class_name = self.model.names[class_id]

            detection_info = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': bbox.tolist()  # Convert to list for JSON serialization
            }

            detections.append(detection_info)

            # Update detection summary
            if class_name in detection_summary:
                detection_summary[class_name] += 1
            else:
                detection_summary[class_name] = 1

        # Calculate summary statistics
        summary = {
            'total_detections': len(detections),
            'detection_summary': detection_summary,
            'model_used': self.model_path,
            'confidence_threshold': self.conf_threshold,
            'inference_time': round(inference_time, 4),  # Add inference time in seconds
            'model_format': self.model_format
        }

        result = {
            'summary': summary,
            'detections': detections
        }

        # Convert NumPy types to Python native types for JSON serialization
        result = convert_numpy_types(result)

        logger.info(f"Detection completed. Found {len(detections)} objects.")
        return annotated_img, result

    def detect_video(self, video_path: str, output_path: str,
                    frame_skip: int = 1, max_frames: Optional[int] = None,
                    batch_size: int = 1, progress_callback: Optional[callable] = None) -> Dict:
        """
        Detect objects in a video and save annotated video
        Optimized with batch processing support

        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            frame_skip (int): Process every Nth frame (default: 1)
            max_frames (int): Maximum number of frames to process (default: None)
            batch_size (int): Number of frames to process in batch for inference (default: 1)
            progress_callback (callable): Function to call with progress updates

        Returns:
            dict: Video detection results with statistics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Reset stop event for new processing
        self.reset_stop_event()

        # Validate input video path
        if not os.path.exists(video_path):
            raise ValueError(f"Input video file does not exist: {video_path}")

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
        logger.info(f"Batch processing enabled with batch_size={batch_size}")

        # Setup video writer with better browser compatibility
        # Try different encoders for better compatibility
        fourcc_attempts = [
            cv2.VideoWriter_fourcc(*'H264'),  # H.264 - best compatibility
            cv2.VideoWriter_fourcc(*'XVID'),  # XVID - good compatibility
            cv2.VideoWriter_fourcc(*'MP4V'),  # MP4V - fallback
            cv2.VideoWriter_fourcc(*'mp4v'),  # mp4v - original fallback
        ]

        out = None
        for fourcc in fourcc_attempts:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                logger.info(f"Using video codec: {fourcc}")
                break
            else:
                out.release()
                out = None

        if out is None:
            cap.release()
            raise ValueError("Could not initialize video writer with any codec")

        # Process frames
        frame_count = 0
        processed_frames = 0
        total_detections_across_video = []
        video_detection_summary = {}
        inference_times = []

        # Variables to store the last processed results for skipped frames
        last_annotated_frame = None
        last_frame_detections = []

        # Store original thresholds
        original_conf = self.conf_threshold
        original_iou = self.iou_threshold

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Starting video processing with frame_skip={frame_skip}")

        # Frame batch for batch processing
        frame_batch = []
        frame_indices = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if processing was cancelled (client disconnected)
                if self._stop_event.is_set():
                    logger.warning("Video processing cancelled by client")
                    # Clean up partial output file
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                            logger.info(f"Removed partial output file: {output_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove partial output file: {e}")
                    raise InterruptedError("Video processing was cancelled")

                # Check if this frame should be processed for inference
                should_process_inference = (frame_count % frame_skip == 0)

                if should_process_inference:
                    # Add frame to batch
                    frame_batch.append(frame)
                    frame_indices.append(frame_count)

                    # Process batch when it reaches batch_size
                    if len(frame_batch) >= batch_size:
                        batch_results = self._process_frame_batch(frame_batch, frame_indices)
                        inference_times.extend(batch_results['inference_times'])
                        total_detections_across_video.extend(batch_results['detections'])

                        # Update detection summary
                        for det in batch_results['detections']:
                            class_name = det['class_name']
                            video_detection_summary[class_name] = video_detection_summary.get(class_name, 0) + 1

                        # Store last annotated frame for skipped frames
                        last_annotated_frame = batch_results['annotated_frames'][-1]
                        last_frame_detections = [d for d in batch_results['detections']
                                                 if d['frame_number'] == frame_indices[-1]]

                        # Clear batch
                        frame_batch = []
                        frame_indices = []

                        processed_frames += len(batch_results['annotated_frames'])

                        # Check max_frames limit
                        if max_frames and processed_frames >= max_frames:
                            logger.warning(f"Reached maximum processed frames limit ({max_frames}), stopping video processing")
                            break
                else:
                    # Check if processing was cancelled (client disconnected)
                    if self._stop_event.is_set():
                        logger.warning("Video processing cancelled by client")
                        # Clean up partial output file
                        if os.path.exists(output_path):
                            try:
                                os.remove(output_path)
                                logger.info(f"Removed partial output file: {output_path}")
                            except Exception as e:
                                logger.warning(f"Failed to remove partial output file: {e}")
                        raise InterruptedError("Video processing was cancelled")

                    # For skipped frames, use the last processed results if available
                    if last_annotated_frame is not None:
                        annotated_frame = last_annotated_frame.copy()
                        current_frame_detections = last_frame_detections
                    else:
                        # First frame before any inference - use original frame
                        annotated_frame = frame.copy()
                        current_frame_detections = []

                    # Add frame number and detection count to frame
                    cv2.putText(annotated_frame, f'Frame: {frame_count}',
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'Detections: {len(current_frame_detections)}',
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Write ALL frames to output video to maintain original duration
                    out.write(annotated_frame)

                    # Progress callback
                    if progress_callback:
                        progress = (frame_count / total_frames) * 100
                        progress_callback(progress, frame_count, total_frames, len(current_frame_detections))

                    frame_count += 1

            # Process remaining frames in batch
            if frame_batch:
                batch_results = self._process_frame_batch(frame_batch, frame_indices)
                inference_times.extend(batch_results['inference_times'])
                total_detections_across_video.extend(batch_results['detections'])

                # Update detection summary
                for det in batch_results['detections']:
                    class_name = det['class_name']
                    video_detection_summary[class_name] = video_detection_summary.get(class_name, 0) + 1

        finally:
            # Ensure resources are released
            cap.release()
            out.release()
            # Clear frame batch
            if frame_batch:
                frame_batch.clear()
            gc.collect()

        # Calculate statistics
        total_processing_time = sum(inference_times) if inference_times else 0
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        processing_fps = processed_frames / (total_processing_time if total_processing_time > 0 else 1)

        result = {
            'summary': {
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'total_detections': len(total_detections_across_video),
                'detection_summary': video_detection_summary,
                'model_used': self.model_path,
                'confidence_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'fps': fps,
                'frame_skip': frame_skip,
                'batch_size': batch_size,
                'avg_inference_time': round(avg_inference_time, 4),
                'processing_fps': round(processing_fps, 2),
                'processing_time': round(total_processing_time, 2),
                'model_format': self.model_format
            },
            'detections': total_detections_across_video,
            'output_video_path': output_path,
            'video_properties': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames
            }
        }

        # Convert NumPy types to Python native types for JSON serialization
        result = convert_numpy_types(result)

        logger.info(f"Video processing completed:")
        logger.info(f"  - Total frames: {total_frames}")
        logger.info(f"  - Processed frames: {processed_frames}")
        logger.info(f"  - Total detections: {len(total_detections_across_video)}")
        logger.info(f"  - Average inference time: {avg_inference_time:.4f}s")
        logger.info(f"  - Processing FPS: {processing_fps:.2f}")
        logger.info(f"  - Output saved to: {output_path}")

        return result

    def _process_frame_batch(self, frame_batch: List[np.ndarray], frame_indices: List[int]) -> Dict:
        """
        Process a batch of frames for inference (internal method)
        Optimized for batch processing to improve GPU utilization

        Args:
            frame_batch (List[np.ndarray]): List of frames to process
            frame_indices (List[int]): Frame numbers for the batch

        Returns:
            dict: Batch processing results with annotated frames and detections
        """
        if not frame_batch:
            return {
                'annotated_frames': [],
                'detections': [],
                'inference_times': []
            }

        annotated_frames = []
        all_detections = []
        inference_times = []

        # Process each frame in the batch
        for frame, frame_idx in zip(frame_batch, frame_indices):
            # Perform inference
            inference_start = time.time()
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # Get detections
            frame_detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    class_name = self.model.names[class_id]

                    detection_info = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': bbox.tolist(),
                        'frame_number': frame_idx
                    }

                    frame_detections.append(detection_info)
                    all_detections.append(detection_info)

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Add frame number and detection count
            cv2.putText(annotated_frame, f'Frame: {frame_idx}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Detections: {len(frame_detections)}',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            annotated_frames.append(annotated_frame)

        return {
            'annotated_frames': annotated_frames,
            'detections': all_detections,
            'inference_times': inference_times
        }

    def detect_multiple_images(self, image_paths: List[str], output_dir: str,
                            progress_callback: Optional[callable] = None,
                            original_filenames: Optional[List[str]] = None) -> Dict:
        """
        Detect objects in multiple images and save annotated images

        Args:
            image_paths (List[str]): List of paths to input images
            output_dir (str): Directory to save output images
            progress_callback (callable): Function to call with progress updates

        Returns:
            dict: Batch detection results
        """
        if not image_paths:
            return {
                'summary': {
                    'total_images': 0,
                    'processed_images': 0,
                    'total_detections': 0,
                    'detection_summary': {},
                    'model_used': self.model_path,
                    'confidence_threshold': self.conf_threshold,
                    'iou_threshold': self.iou_threshold
                },
                'results': []
            }

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        results = []
        total_detections = 0
        batch_detection_summary = {}
        inference_times = []

        # Store original thresholds
        original_conf = self.conf_threshold
        original_iou = self.iou_threshold

        logger.info(f"Starting batch processing of {len(image_paths)} images")

        for i, image_path in enumerate(image_paths):
            try:
                # Generate output filename
                image_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{image_name}_detected.jpg")

                # Process single image
                inference_start = time.time()
                result = self.detect(image_path, output_path)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)

                # Add metadata
                result['source_image_path'] = image_path
                result['output_path'] = output_path  # Match template expectations
                result['output_image_path'] = output_path  # Keep for backward compatibility
                result['processing_order'] = i + 1
                # Use original filename if provided, otherwise use basename
                result['original_filename'] = (original_filenames[i] if original_filenames and i < len(original_filenames)
                                             else os.path.basename(image_path))
                result['file_type'] = 'image'  # Add file type for template

                results.append(result)
                total_detections += result['summary']['total_detections']

                # Update batch detection summary
                for class_name, count in result['summary']['detection_summary'].items():
                    if class_name in batch_detection_summary:
                        batch_detection_summary[class_name] += count
                    else:
                        batch_detection_summary[class_name] = count

                # Progress callback
                if progress_callback:
                    progress = ((i + 1) / len(image_paths)) * 100
                    progress_callback(progress, i + 1, len(image_paths),
                                   result['summary']['total_detections'], image_name)

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                # Use original filename if provided, otherwise use basename
                original_filename = (original_filenames[i] if original_filenames and i < len(original_filenames)
                                   else os.path.basename(image_path))
                results.append({
                    'source_image_path': image_path,
                    'output_path': None,  # Match template expectations
                    'output_image_path': None,  # Keep for backward compatibility
                    'processing_order': i + 1,
                    'original_filename': original_filename,
                    'file_type': 'image',
                    'error': str(e),
                    'success': False
                })

        # Calculate batch statistics
        avg_inference_time = np.mean(inference_times) if inference_times else 0

        batch_result = {
            'summary': {
                'total_images': len(image_paths),
                'processed_images': len([r for r in results if r.get('success', True)]),
                'failed_images': len([r for r in results if not r.get('success', True)]),
                'total_detections': total_detections,
                'detection_summary': batch_detection_summary,
                'model_used': self.model_path,
                'confidence_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'avg_inference_time': round(avg_inference_time, 4),
                'model_format': self.model_format
            },
            'results': results
        }

        # Convert NumPy types to Python native types for JSON serialization
        batch_result = convert_numpy_types(batch_result)

        logger.info(f"Batch processing completed:")
        logger.info(f"  - Total images: {len(image_paths)}")
        logger.info(f"  - Processed: {batch_result['summary']['processed_images']}")
        logger.info(f"  - Failed: {batch_result['summary']['failed_images']}")
        logger.info(f"  - Total detections: {total_detections}")
        logger.info(f"  - Average inference time: {avg_inference_time:.4f}s")

        return batch_result

    def change_model(self, new_model_path: str):
        """
        Change the YOLO model being used
        Now uses caching mechanism for faster model switching

        Args:
            new_model_path (str): Path to new YOLO model file or model name
        """
        self.model_path = new_model_path
        self.model_format = self._detect_model_format(new_model_path)

        # Check if model is already cached
        with CACHE_LOCK:
            cache_key = f"{new_model_path}_{self.model_format}"
            if cache_key in MODEL_CACHE:
                logger.info(f"Loading model from cache: {new_model_path}")
                self.model = MODEL_CACHE[cache_key]
                return

        # Load and cache the model
        self.load_model()

        # Add to cache
        with CACHE_LOCK:
            MODEL_CACHE[cache_key] = self.model
            logger.info(f"Model cached: {cache_key}")


def preload_models(model_list: List[str] = None, max_cache_size: int = 5):
    """
    Preload commonly used models into cache for faster access

    Args:
        model_list (List[str]): List of model names to preload (default: common models)
        max_cache_size (int): Maximum number of models to keep in cache
    """
    if model_list is None:
        model_list = [
            'yolo11n.pt',
            'yolov8n.pt',
            'yolov5n.pt'
        ]

    logger.info(f"Preloading {len(model_list)} models...")

    for model_path in model_list[:max_cache_size]:
        try:
            cache_key = f"{model_path}_pytorch"
            if cache_key not in MODEL_CACHE:
                logger.info(f"Preloading model: {model_path}")
                model = YOLO(model_path)
                with CACHE_LOCK:
                    MODEL_CACHE[cache_key] = model
                logger.info(f"Successfully preloaded: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to preload model {model_path}: {str(e)}")

    logger.info(f"Model preloading complete. Cache size: {len(MODEL_CACHE)}")


def clear_model_cache():
    """Clear the model cache to free memory"""
    with CACHE_LOCK:
        MODEL_CACHE.clear()
        gc.collect()
        logger.info("Model cache cleared")


def get_cache_info() -> Dict:
    """Get information about the current model cache"""
    with CACHE_LOCK:
        return {
            'cached_models': list(MODEL_CACHE.keys()),
            'cache_size': len(MODEL_CACHE),
            'cache_memory_mb': sum(getattr(model, '__sizeof__', lambda: 0)() for model in MODEL_CACHE.values()) / (1024 * 1024)
        }


# Global instance of the inference class
# This allows for model caching to improve performance
yolo_inference = YOLOInference(iou_threshold=0.45)


def get_available_models() -> Dict[str, Dict[str, List[str]]]:
    """
    Get a list of all available models including custom models in different formats

    Returns:
        dict: Dictionary containing predefined models and custom models by format
    """
    # Predefined PyTorch models
    predefined_pytorch = [
        'yolo11n.pt',      # YOLOv11 Nano
        'yolo11s.pt',      # YOLOv11 Small
        'yolo11m.pt',      # YOLOv11 Medium
        'yolo11l.pt',      # YOLOv11 Large
        'yolo11x.pt',      # YOLOv11 Extra Large
        'yolov8n.pt',      # YOLOv8 Nano
        'yolov8s.pt',      # YOLOv8 Small
        'yolov8m.pt',      # YOLOv8 Medium
        'yolov8l.pt',      # YOLOv8 Large
        'yolov8x.pt',      # YOLOv8 Extra Large
        'yolov5n.pt',      # YOLOv5 Nano
        'yolov5s.pt',      # YOLOv5 Small
        'yolov5m.pt',      # YOLOv5 Medium
        'yolov5l.pt',      # YOLOv5 Large
        'yolov5x.pt',      # YOLOv5 Extra Large
    ]

    # Get custom models from models directory by format
    custom_pytorch = []
    custom_onnx = []
    custom_tensorrt = []

    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pt'):
                custom_pytorch.append(file)
            elif file.endswith('.onnx'):
                custom_onnx.append(file)
            elif file.endswith('.engine'):
                custom_tensorrt.append(file)

    return {
        'predefined_models': {
            'pytorch': predefined_pytorch
        },
        'custom_models': {
            'pytorch': custom_pytorch,
            'onnx': custom_onnx,
            'tensorrt': custom_tensorrt
        }
    }


if __name__ == "__main__":
    # Example usage
    inference = YOLOInference(model_path='yolo11n.pt', conf_threshold=0.25)

    # For testing, you would need an actual image file
    # result = inference.detect('path/to/test/image.jpg', 'path/to/output/image.jpg')
    # print(result)
    pass