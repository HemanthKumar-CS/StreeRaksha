"""
StreeRaksha - Main Application Entry Point

This is the main entry point for the StreeRaksha application.
The application monitors camera feed to detect potential safety concerns for women.

Created by: Stree Raksha Team
"""

import cv2
import time
import warnings
from datetime import datetime

# Import StreeRaksha modules
from detector import StreeRakshaDetector
from gender_detector import GenderDetector
from tracker import PersonTracker
from pose_analyzer import PoseAnalyzer
from alert_system import AlertSystem
from visualizer import Visualizer
from logger import Logger
from debugger import Debugger

# Suppress warnings
warnings.filterwarnings("ignore")


def check_dependencies():
    """Check and report on required dependencies"""
    dependencies = {
        "opencv": True,
        "numpy": True,
        "ultralytics": True,
        "mediapipe": True,
        "tensorflow": True,
        "transformers": True,
    }

    # Check dependencies
    try:
        import numpy
    except ImportError:
        dependencies["numpy"] = False

    try:
        import ultralytics
    except ImportError:
        dependencies["ultralytics"] = False

    try:
        import mediapipe
    except ImportError:
        dependencies["mediapipe"] = False

    try:
        import tensorflow
    except ImportError:
        dependencies["tensorflow"] = False

    try:
        from transformers import AutoModelForImageClassification
    except ImportError:
        dependencies["transformers"] = False

    print("\n=== Dependency Check ===")
    for dep, status in dependencies.items():
        status_msg = "✓ Installed" if status else "✗ Missing"
        print(f"{dep}: {status_msg}")

    if not dependencies["tensorflow"]:
        print("\n====== TensorFlow Installation ======")
        print("TensorFlow is missing but not required for core functionality.")
        print("If you want to install it (for Python 3.10 or 3.11), run:")
        print("pip install tensorflow==2.12.0")
        print("Note: TensorFlow doesn't officially support Python 3.12 yet.")

    if not dependencies["transformers"]:
        print("\n====== Hugging Face Installation ======")
        print("Hugging Face transformers is missing. For better gender detection, install:")
        print("pip install transformers")

    return all(dependencies.values())


def main():
    """Main function to run the StreeRaksha application"""
    print("Starting StreeRaksha Safety Monitoring System...")

    # Initialize logger
    logger = Logger()
    logger.info("Starting StreeRaksha application")

    # Initialize debugger
    debugger = Debugger(logger)

    # Initialize components
    logger.info("Initializing components...")

    # Initialize gender detector
    gender_detector = GenderDetector()

    # Initialize pose analyzer
    pose_analyzer = PoseAnalyzer()

    # Initialize person tracker
    tracker = PersonTracker()

    # Initialize alert system
    alert_system = AlertSystem()

    # Initialize visualizer
    visualizer = Visualizer()

    try:
        # Initialize YOLO detector
        from ultralytics import YOLO
        logger.info("Loading YOLO model")
        model = YOLO("yolov8n.pt")
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        print("Failed to load YOLO model. Please make sure you have ultralytics installed.")
        print("Install with: pip install ultralytics")
        return

    # Connect to camera
    logger.info("Connecting to camera")
    try:
        cap = connect_camera(logger)
    except Exception as e:
        logger.error(f"Failed to connect to camera: {e}")
        print("Failed to connect to camera. Please check connections and permissions.")
        return

    # Main processing loop
    frame_count = 0
    last_alert_time = 0

    logger.info("Entering main processing loop")
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()

        if not ret:
            logger.warning("Failed to read frame from camera")
            print("Failed to read frame from camera. Attempting to reconnect...")
            cap.release()
            time.sleep(1)
            try:
                cap = connect_camera(logger)
                continue
            except Exception as e:
                logger.error(f"Failed to reconnect to camera: {e}")
                print("Failed to reconnect to camera. Exiting.")
                break

        frame_count += 1
        current_time = datetime.now()

        # Process frame with YOLO
        try:
            # Detect persons
            results = model(frame, conf=0.5, classes=0)  # Class 0 is person

            # Extract bounding boxes
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append((x1, y1, x2, y2))

            # Filter valid detections
            valid_detections = tracker.filter_valid_detections(detections)

            # Apply non-maximum suppression
            filtered_detections = tracker.non_max_suppression(valid_detections)

            # Update person trackers
            persons = tracker.update_trackers(
                filtered_detections, frame, gender_detector)

            # Process pose for each person
            for person in persons:
                # Extract person data
                bbox = person['bbox']

                # Detect pose
                pose_landmarks, pose_analysis = pose_analyzer.detect_pose(
                    frame, bbox)

                if pose_analysis:
                    person['pose'] = pose_analysis

                # Draw pose if available
                if pose_landmarks:
                    frame = pose_analyzer.draw_pose(frame, pose_landmarks)

            # Calculate risk scores and check for alerts
            hour = current_time.hour
            is_night = hour >= 18 or hour < 6
            females = [p for p in persons if p['gender'] == "Female"]
            males = [p for p in persons if p['gender'] == "Male"]

            # Process each female to calculate risk
            for female in females:
                risk_score = alert_system.calculate_risk_score(
                    female, females, males, is_night, persons)
                female['risk_score'] = risk_score

            # Detect alerts
            alert_detected, alert_message = alert_system.detect_alerts(
                persons, current_time)

            # Draw visualizations
            # Draw detection boxes
            frame = visualizer.draw_detection_boxes(frame, persons)

            # Add risk indicators
            frame = alert_system.add_risk_indicators(
                frame, females, males, is_night)

            # Draw statistics
            frame = visualizer.draw_stats(
                frame, frame_count, persons, current_time)

            # Handle alerts
            if alert_detected and time.time() - last_alert_time > alert_system.alert_config['ALERT_COOLDOWN']:
                last_alert_time = time.time()
                # Draw alert
                frame = visualizer.draw_alert(frame, alert_message)

                # Save evidence
                evidence_path = alert_system.save_evidence(frame)
                logger.warning(
                    f"Alert detected: {alert_message}. Evidence saved to {evidence_path}")
            elif alert_detected:
                # Show alert but with cooldown
                frame = visualizer.draw_alert(frame, alert_message)

            # Add debug overlay if needed
            debug_info = {
                "Detector": "YOLO",
                "Detections": len(detections),
                "Valid Tracks": len(persons),
                "Night Mode": str(is_night)
            }
            frame = debugger.add_debug_overlay(frame, debug_info)

            # Show the result
            cv2.imshow('StreeRaksha Safety Monitoring', frame)

            # Check for exit keys
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or 'q' to exit
                logger.info("User requested exit")
                break

        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {e}")
            print(f"Error processing frame: {e}")
            # Save debug frame
            debugger.save_debug_frame(frame, "error")

    # Clean up
    logger.info("Shutting down StreeRaksha")
    cap.release()
    cv2.destroyAllWindows()


def connect_camera(logger):
    """Connect to camera with fallback options"""
    # Try multiple camera indices
    for camera_idx in [0, 1, 2]:
        try:
            print(f"Trying camera index {camera_idx}...")
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"Successfully connected to camera {camera_idx}")
                logger.info(f"Connected to camera index {camera_idx}")
                return cap
        except Exception as e:
            print(f"Failed to connect to camera {camera_idx}: {e}")

    # Fallback to DirectShow backend (Windows)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("Successfully connected using DirectShow")
            logger.info("Connected to camera using DirectShow")
            return cap
    except Exception as e:
        print(f"Failed to connect using DirectShow: {e}")
        logger.error(f"DirectShow camera connection failed: {e}")

    # If all attempts fail
    logger.error("Failed to connect to any camera")
    raise RuntimeError(
        "Could not connect to any camera. Please check connections and permissions.")


if __name__ == "__main__":
    check_dependencies()
    main()
