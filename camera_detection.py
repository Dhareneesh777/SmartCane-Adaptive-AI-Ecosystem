"""
Pi Camera with TensorFlow Lite Object Detection
Identifies objects in front of the user
"""
import time
import logging
import numpy as np
from typing import List, Tuple, Optional
try:
    from picamera2 import Picamera2
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("Warning: picamera2 or tflite_runtime not installed")
    print("Install: sudo apt install python3-picamera2")
    print("Install: pip3 install tflite-runtime")

import config

logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self):
        """Initialize camera and ML model"""
        self.camera = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.initialized = False
        
    def initialize(self) -> bool:
        """Setup camera and load TensorFlow Lite model"""
        try:
            # Initialize camera
            self.camera = Picamera2()
            camera_config = self.camera.create_still_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            time.sleep(2)  # Camera warmup
            
            # Load TFLite model
            self.interpreter = Interpreter(model_path=config.MODEL_PATH)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Load labels
            with open(config.LABEL_PATH, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            self.initialized = True
            logger.info("Camera and object detection model initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize object detection: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from camera
        Returns: RGB image array (320x320x3)
        """
        if not self.initialized:
            return None
        
        try:
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def detect_objects(self, frame: Optional[np.ndarray] = None) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Run object detection on frame
        Args:
            frame: RGB image, or None to capture new frame
        Returns: List of (label, confidence, (x1, y1, x2, y2))
        """
        if not self.initialized:
            return []
        
        # Capture if no frame provided
        if frame is None:
            frame = self.capture_frame()
            if frame is None:
                return []
        
        try:
            # Prepare input
            input_data = np.expand_dims(frame, axis=0)
            
            # Normalize if model expects float input
            if self.input_details[0]['dtype'] == np.float32:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get outputs
            # Output format: [boxes, classes, scores, num_detections]
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            num = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
            
            # Filter by confidence
            detections = []
            for i in range(num):
                score = scores[i]
                if score >= config.DETECTION_CONFIDENCE:
                    class_id = int(classes[i])
                    label = self.labels[class_id] if class_id < len(self.labels) else "Unknown"
                    
                    # Convert box coordinates (normalized) to pixels
                    ymin, xmin, ymax, xmax = boxes[i]
                    x1 = int(xmin * config.CAMERA_WIDTH)
                    y1 = int(ymin * config.CAMERA_HEIGHT)
                    x2 = int(xmax * config.CAMERA_WIDTH)
                    y2 = int(ymax * config.CAMERA_HEIGHT)
                    
                    detections.append((label, float(score), (x1, y1, x2, y2)))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def get_primary_obstacle(self) -> Optional[Tuple[str, str]]:
        """
        Identify the main obstacle in view
        Returns: (object_name, direction) or None
        """
        detections = self.detect_objects()
        
        if not detections:
            return None
        
        # Find largest object (by bounding box area)
        largest = None
        max_area = 0
        
        for label, score, (x1, y1, x2, y2) in detections:
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest = (label, score, x1, y1, x2, y2)
        
        if largest is None:
            return None
        
        label, score, x1, y1, x2, y2 = largest
        
        # Determine direction based on center x coordinate
        center_x = (x1 + x2) / 2
        width = config.CAMERA_WIDTH
        
        if center_x < width * 0.33:
            direction = "on your left"
        elif center_x > width * 0.66:
            direction = "on your right"
        else:
            direction = "ahead"
        
        return label, direction
    
    def get_obstacle_description(self) -> Optional[str]:
        """
        Generate natural language description of scene
        Returns: String like "car ahead, person on your left"
        """
        detections = self.detect_objects()
        
        if not detections:
            return None
        
        # Sort by bounding box area (largest first)
        sorted_det = sorted(
            detections,
            key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]),
            reverse=True
        )
        
        # Take top 2 objects
        descriptions = []
        for label, score, (x1, y1, x2, y2) in sorted_det[:2]:
            center_x = (x1 + x2) / 2
            width = config.CAMERA_WIDTH
            
            if center_x < width * 0.33:
                direction = "on your left"
            elif center_x > width * 0.66:
                direction = "on your right"
            else:
                direction = "ahead"
            
            descriptions.append(f"{label} {direction}")
        
        return ", ".join(descriptions)
    
    def shutdown(self):
        """Stop camera and cleanup"""
        if self.camera:
            try:
                self.camera.stop()
                logger.info("Camera shutdown")
            except:
                pass


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = ObjectDetector()
    if detector.initialize():
        print("Object detection ready. Running for 10 detections...")
        
        for i in range(10):
            desc = detector.get_obstacle_description()
            if desc:
                print(f"Detection {i+1}: {desc}")
            else:
                print(f"Detection {i+1}: No objects")
            
            time.sleep(2)
        
        detector.shutdown()
