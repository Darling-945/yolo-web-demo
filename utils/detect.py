"""
Detection script for YOLO models using Ultralytics
Supports YOLOv5, YOLOv8, and YOLOv11 models
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse


def detect_objects(image_path, output_path, model_path='yolo11n.pt', conf_threshold=0.25):
    """
    Detect objects in an image using YOLO model

    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image with detections
        model_path (str): Path to YOLO model file or model name
        conf_threshold (float): Confidence threshold for detections
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Perform inference
    results = model(img, conf=conf_threshold)

    # Draw results on the image
    annotated_img = results[0].plot()

    # Save the output image
    cv2.imwrite(output_path, annotated_img)

    # Get detection information
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

        detections.append({
            'class_id': class_id,
            'class_name': model.names[class_id],
            'confidence': conf,
            'bbox': bbox.tolist()
        })

    return detections


def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--weights', type=str, default='yolo11n.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder
    parser.add_argument('--output', type=str, default='results/', help='output folder')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    opt = parser.parse_args()

    print(opt)

    # Create output directory if it doesn't exist
    os.makedirs(opt.output, exist_ok=True)

    # Process a single image or a directory of images
    if os.path.isfile(opt.source):
        # Single image file
        output_path = os.path.join(opt.output, os.path.basename(opt.source))
        detect_objects(opt.source, output_path, opt.weights, opt.conf_thres)
    elif os.path.isdir(opt.source):
        # Directory of images
        for img_file in os.listdir(opt.source):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img_path = os.path.join(opt.source, img_file)
                output_path = os.path.join(opt.output, img_file)
                detect_objects(img_path, output_path, opt.weights, opt.conf_thres)

    print('Detection complete.')


if __name__ == '__main__':
    main()