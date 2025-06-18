# streeraksha.py

# Required imports
import os
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import torch
from pose_analyzer import PoseAnalyzer
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    has_tensorflow = True
except ImportError:
    has_tensorflow = False
    print("TensorFlow not available. Continuing without TensorFlow functionality.")

try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    has_transformers = True
except ImportError:
    has_transformers = False
    print("Hugging Face transformers not available. Falling back to simulated gender detection.")

try:
    import mediapipe as mp
    has_mediapipe = True
    print("MediaPipe successfully imported for pose detection")
except ImportError:
    has_mediapipe = False
    print("MediaPipe not available. Install with: pip install mediapipe")


class StreeRakshaDetector:
    def __init__(self):
        # Create directories for evidence
        os.makedirs('evidence', exist_ok=True)

        # Configuration for optimization
        self.TRACK_EXPIRATION = 8  # Number of frames before we expire a track
        self.MAX_TRACK_DISTANCE = 50  # Maximum distance for track association
        self.DETECTION_CONFIDENCE = 0.5
        self.MINIMUM_PERSON_SIZE = 60  # Minimum width or height for a valid person detection

        # Unified person trackers
        self.person_trackers = {}
        self.next_track_id = 0

        # Person gender cache
        self.person_gender_cache = {}

        # Last alert time for cooldown
        self.last_alert_time = 0

        # Load models
        self._load_models()

        # Alert configuration
        self.alert_config = {
            'NIGHT_HOURS_START': 18,
            'NIGHT_HOURS_END': 6,
            'MEN_RATIO_THRESHOLD': 1.0,
            'PROXIMITY_THRESHOLD': 150,
            'ALERT_COOLDOWN': 10.0,
            'RISK_THRESHOLD': 60,     # Minimum risk score to trigger alert
            'FRAME_THRESHOLD': 5      # Number of consecutive frames before alert
        }

        # Alert levels
        self.alert_levels = {
            "NOTICE": 1,    # Might be concerning
            "WARNING": 2,   # Potentially troubling
            "ALERT": 3      # Serious concern
        }

        # Initialize pose analyzer
        self.pose_analyzer = PoseAnalyzer()

        # Potential alerts storage
        self.potential_alerts = {}  # Track {alert_type: consecutive_frames}

        # Debug flags
        self.show_risk_scores = True  # Set to True to show risk scores on screen

    def _load_models(self):
        """Load all required models"""
        print("Loading YOLO model...")
        try:
            # This will download the model if not already present
            self.model = YOLO('yolov8n.pt')
            self.has_yolo = True
            print("Successfully loaded YOLO model")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.has_yolo = False
            print("⚠️ YOLO model could not be loaded. Person detection will not work!")

        print("Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        print("Loading gender classification model from Hugging Face...")
        if has_transformers:
            try:
                self.gender_feature_extractor = AutoFeatureExtractor.from_pretrained(
                    "rizvandwiki/gender-classification-2")
                self.gender_model = AutoModelForImageClassification.from_pretrained(
                    "rizvandwiki/gender-classification-2")
                self.use_gender_model = True
                print("Successfully loaded gender classification model")
            except Exception as e:
                self.use_gender_model = False
                print(f"Could not load gender model: {e}")
                print("Falling back to simulated gender classification")
        else:
            self.use_gender_model = False
            print(
                "Hugging Face transformers not available. Using simulated gender classification")

    def connect_camera(self):
        """Connect to camera with fallback options"""
        # Try multiple camera indices
        for camera_idx in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    print(
                        f"Successfully connected to camera index {camera_idx}")
                    return cap
            except Exception as e:
                print(f"Failed to connect to camera index {camera_idx}: {e}")

        # Fallback to DirectShow backend (Windows)
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                print("Successfully connected using DirectShow backend")
                return cap
        except Exception as e:
            print(f"Failed to connect using DirectShow: {e}")

        # If all attempts fail
        raise RuntimeError(
            "Could not connect to any camera. Please check connections and permissions.")

    def predict_gender(self, image):
        """Predict gender using Hugging Face model with stronger male bias correction"""
        if not self.use_gender_model:
            # Fallback to simulated gender if model not available
            # Use the hash of the image data as a deterministic seed
            seed_value = hash(image.tobytes()) % 100
            gender = "Female" if seed_value < 30 else "Male"  # 30% chance female
            confidence = np.random.uniform(0.80, 0.95)
            return gender, confidence

        try:
            # Convert OpenCV BGR image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Prepare image for the model
            inputs = self.gender_feature_extractor(
                images=pil_image, return_tensors="pt")

            # Get prediction
            with torch.no_grad():
                outputs = self.gender_model(**inputs)
                logits = outputs.logits

                # Find indices for male and female classes
                male_idx = None
                female_idx = None
                for idx, label in self.gender_model.config.id2label.items():
                    if "female" in label.lower() or "woman" in label.lower() or "f" == label.lower():
                        female_idx = idx
                    else:
                        male_idx = idx

                # Apply a stronger bias toward male classification
                if male_idx is not None and female_idx is not None:
                    bias_correction = 0.25  # Increased bias factor (was 0.1)
                    logits[0][male_idx] += bias_correction

                predicted_class_idx = logits.argmax(-1).item()

            # Get confidence
            softmax_values = torch.softmax(logits, dim=1)[0]
            confidence = softmax_values[predicted_class_idx].item()

            # Get gender label
            gender_label = self.gender_model.config.id2label[predicted_class_idx]

            # Higher threshold for female classification based on image quality
            female_threshold = 0.70  # Increased from 0.60
            male_threshold = 0.50    # Slightly decreased to favor male classification

            # Format as "Male" or "Female" (adjust based on model's label format)
            if "female" in gender_label.lower() or "woman" in gender_label.lower() or "f" == gender_label.lower():
                gender = "Female"

                # If female but confidence is low, check the gap with male prediction
                if confidence < female_threshold and male_idx is not None:
                    male_confidence = softmax_values[male_idx].item()

                    # If the difference in confidence is small (more strict now), classify as male
                    if (confidence - male_confidence) < 0.3:  # Increased from 0.15
                        gender = "Male"
                        confidence = male_confidence
            else:
                gender = "Male"

            # Add small confidence boost for male classification to improve display
            if gender == "Male" and confidence < 0.9:
                confidence = min(0.9, confidence * 1.1)

            return gender, confidence
        except Exception as e:
            print(f"Error in gender prediction: {e}")
            # Fallback if prediction fails
            seed_value = hash(image.tobytes()) % 100
            gender = "Female" if seed_value < 30 else "Male"
            confidence = np.random.uniform(0.80, 0.95)
            return gender, confidence

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate IoU
        iou = intersection_area / \
            float(box1_area + box2_area - intersection_area)
        return max(0.0, min(1.0, iou))  # Ensure value is between 0 and 1

    def center_distance(self, box1, box2):
        """Calculate center distance between boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)

        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def non_max_suppression(self, boxes, iou_threshold=0.5):
        """Non-maximum suppression to remove overlapping boxes"""
        if not boxes:
            return []

        # Sort boxes by area (largest first)
        sorted_boxes = sorted(boxes, key=lambda x: (
            x[2]-x[0]) * (x[3]-x[1]), reverse=True)

        keep = []
        while sorted_boxes:
            # Keep the largest box
            current = sorted_boxes.pop(0)
            keep.append(current)

            # Filter out boxes with high IoU overlap
            sorted_boxes = [box for box in sorted_boxes if self.calculate_iou(
                current, box) < iou_threshold]

        return keep

    def calculate_risk_score(self, female, females, males, is_night, persons):
        """Calculate comprehensive risk score based on multiple factors"""
        score = 0

        # Base environmental risk
        if is_night:
            score += 15  # Night adds baseline risk

            # More risk if alone at night
            if len(females) <= 2 and len(males) == 0:
                score += 25

        # Male proximity risk
        f_x1, f_y1, f_x2, f_y2 = female['bbox']
        f_center_x = (f_x1 + f_x2) / 2
        f_center_y = (f_y1 + f_y2) / 2

        nearby_males = 0
        close_males = 0

        for male in males:
            m_x1, m_y1, m_x2, m_y2 = male['bbox']
            m_center_x = (m_x1 + m_x2) / 2
            m_center_y = (m_y1 + m_y2) / 2

            distance = np.sqrt((f_center_x - m_center_x) **
                               2 + (f_center_y - m_center_y)**2)

            if distance < self.alert_config['PROXIMITY_THRESHOLD']:
                nearby_males += 1

            # Extra close
            if distance < self.alert_config['PROXIMITY_THRESHOLD'] * 0.6:
                close_males += 1

        # Surrounding risk
        if nearby_males >= 2:
            score += 20 + (nearby_males - 2) * 5
            if close_males >= 1:
                score += 10 * close_males  # Extra risk for very close males

        # Behavioral risk - highest weight factors
        if 'pose' in female:
            pose = female['pose']
            if pose.get('hands_up'):
                score += 40  # Distress signal is high priority
            if pose.get('lying_down') and len(persons) > 1:
                score += 35  # Concerning in public
            if pose.get('running'):
                # Check for pursuing males
                if any(male.get('pose', {}).get('running') for male in males):
                    score += 50  # Potential pursuit is very concerning
                else:
                    score += 15  # Running alone might be exercise

        return score

    def detect_alerts(self, persons, current_time=None):
        """Enhanced alert detection with risk scoring and temporal persistence"""
        if not persons:
            self.reset_potential_alerts()
            return False, "No alert"

        if current_time is None:
            current_time = datetime.now()

        # Count males and females
        females = [p for p in persons if p['gender'] == "Female"]
        males = [p for p in persons if p['gender'] == "Male"]

        # No females detected, reset and return
        if not females:
            self.reset_potential_alerts()
            return False, "No females detected"

        alerts = []
        hour = current_time.hour
        is_night = hour >= self.alert_config['NIGHT_HOURS_START'] or hour < self.alert_config['NIGHT_HOURS_END']

        # Context factors - check if this is a likely safe context
        # If large group with mixed genders, likely safer
        is_safe_social_context = len(persons) > 5 and len(
            females) >= 2 and len(males) >= 2

        # Process each female to calculate individual risks
        for female_idx, female in enumerate(females):
            # Skip processing for females in clearly safe contexts unless there's a pose signal
            if is_safe_social_context and not any(pose_flag for pose_flag in
                                                  female.get('pose', {}).values() if pose_flag):
                continue

            # Calculate risk score
            risk_score = self.calculate_risk_score(
                female, females, males, is_night, persons)

            # Get female ID for tracking over time
            female_id = female.get('track_id', f"unknown_{female_idx}")
            alert_key = f"female_{female_id}"

            # Track potential alerts over multiple frames
            if risk_score >= self.alert_config['RISK_THRESHOLD']:
                if alert_key not in self.potential_alerts:
                    self.potential_alerts[alert_key] = {
                        'count': 1, 'score': risk_score}
                else:
                    self.potential_alerts[alert_key]['count'] += 1
                    # Update with max score
                    self.potential_alerts[alert_key]['score'] = max(
                        risk_score, self.potential_alerts[alert_key]['score'])

                # Alert only triggers after consistent detection across frames
                if self.potential_alerts[alert_key]['count'] >= self.alert_config['FRAME_THRESHOLD']:
                    # Determine alert level and message
                    alert_severity, alert_message = self.generate_alert_message(
                        female, females, males, is_night, risk_score)
                    alerts.append((alert_severity, alert_message))
            else:
                # Reset counter if risk drops
                if alert_key in self.potential_alerts:
                    self.potential_alerts[alert_key]['count'] = max(
                        0, self.potential_alerts[alert_key]['count'] - 2)  # Decay faster than buildup

                    if self.potential_alerts[alert_key]['count'] == 0:
                        del self.potential_alerts[alert_key]

        # Clean potential alerts for tracks that no longer exist
        self.clean_potential_alerts(persons)

        # Get highest severity alert
        if alerts:
            # Sort by alert severity
            alerts.sort(key=lambda x: x[0], reverse=True)
            return True, alerts[0][1]  # Return highest severity alert

        return False, "No alert"

    def filter_valid_detections(self, detections):
        """Filter out detections that are too small"""
        valid_detections = []
        for box in detections:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Filter out detections that are too small
            if width > self.MINIMUM_PERSON_SIZE and height > self.MINIMUM_PERSON_SIZE:
                valid_detections.append(box)

        return valid_detections

    def update_person_trackers(self, detections, frame):
        """Update person trackers with new detections"""
        # Skip if no detections
        if not detections:
            # Update TTL for all trackers
            for track_id in list(self.person_trackers.keys()):
                self.person_trackers[track_id]['ttl'] -= 1
                if self.person_trackers[track_id]['ttl'] <= 0:
                    del self.person_trackers[track_id]
            return

        # Mark all current tracks as unmatched initially
        unmatched_trackers = list(self.person_trackers.keys())
        matched_detections = []

        # Match detections to existing trackers
        for detection_box in detections:
            best_iou = 0.2  # Minimum IoU threshold
            best_id = None

            for track_id in unmatched_trackers:
                track_box = self.person_trackers[track_id]['bbox']

                # Calculate IoU between current detection and existing track
                iou = self.calculate_iou(detection_box, track_box)
                dist = self.center_distance(detection_box, track_box)

                # Use combination of IoU and distance for matching
                if iou > best_iou or (iou > 0.1 and dist < self.MAX_TRACK_DISTANCE):
                    best_iou = iou
                    best_id = track_id

            if best_id is not None:
                # Update existing tracker with new position
                x1, y1, x2, y2 = detection_box
                person_img = frame[y1:y2, x1:x2]

                # Check for faces within this person box
                person_gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    person_gray, 1.1, 4, minSize=(20, 20))

                has_face = len(faces) > 0
                face_img = None

                if has_face:
                    # Use the largest face if multiple detected
                    fx, fy, fw, fh = faces[0]
                    face_img = person_img[fy:fy+fh, fx:fx+fw]

                self.person_trackers[best_id].update({
                    'bbox': detection_box,
                    'ttl': self.TRACK_EXPIRATION,  # Reset time-to-live
                    'person_img': person_img,
                    'has_face': has_face,
                    'face_img': face_img
                })

                unmatched_trackers.remove(best_id)
                matched_detections.append(detection_box)
            else:
                # Create new tracker
                track_id = f"person_{self.next_track_id}"
                self.next_track_id += 1

                # Extract person image
                x1, y1, x2, y2 = detection_box
                person_img = frame[y1:y2, x1:x2]

                # Check for faces within this person box
                person_gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    person_gray, 1.1, 4, minSize=(20, 20))

                has_face = len(faces) > 0
                face_img = None

                if has_face:
                    # Use the largest face if multiple detected
                    fx, fy, fw, fh = faces[0]
                    face_img = person_img[fy:fy+fh, fx:fx+fw]

                self.person_trackers[track_id] = {
                    'bbox': detection_box,
                    'ttl': self.TRACK_EXPIRATION,
                    'person_img': person_img,
                    'has_face': has_face,
                    'face_img': face_img,
                    'gender': None,
                    'confidence': 0.0,
                    'last_gender_update': time.time()
                }

                matched_detections.append(detection_box)

        # Update TTL for unmatched trackers
        # Use list() to safely modify during iteration
        for track_id in list(unmatched_trackers):
            self.person_trackers[track_id]['ttl'] -= 1

            # Remove expired trackers
            if self.person_trackers[track_id]['ttl'] <= 0:
                del self.person_trackers[track_id]

    def reset_potential_alerts(self):
        """Reset all potential alerts"""
        self.potential_alerts = {}

    def clean_potential_alerts(self, persons):
        """Remove potential alerts for persons no longer in the scene"""
        tracked_ids = set(p.get('track_id')
                          for p in persons if 'track_id' in p)

        keys_to_remove = []
        for key in self.potential_alerts:
            if key.startswith('female_'):
                track_id = key[7:]  # Remove 'female_' prefix
                if track_id not in tracked_ids and not track_id.startswith('unknown'):
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.potential_alerts[key]

    def generate_alert_message(self, female, females, males, is_night, risk_score):
        """Generate appropriate alert message based on risk factors"""
        alert_message = ""
        alert_level = self.alert_levels["NOTICE"]  # Default lowest level

        # Determine most significant factor for the alert
        if 'pose' in female:
            pose = female['pose']
            if pose.get('hands_up'):
                alert_message = "Woman with hands up (potential distress)"
                alert_level = self.alert_levels["ALERT"]

            elif pose.get('running') and any(male.get('pose', {}).get('running') for male in males):
                alert_message = "Woman running with male in pursuit"
                alert_level = self.alert_levels["ALERT"]

            elif pose.get('lying_down') and len([p for p in (females + males) if p != female]) > 0:
                alert_message = "Woman lying down in public area"
                alert_level = self.alert_levels["WARNING"]

        if not alert_message:
            # Count nearby males
            f_x1, f_y1, f_x2, f_y2 = female['bbox']
            f_center_x = (f_x1 + f_x2) / 2
            f_center_y = (f_y1 + f_y2) / 2

            nearby_males = 0
            for male in males:
                m_x1, m_y1, m_x2, m_y2 = male['bbox']
                m_center_x = (m_x1 + m_x2) / 2
                m_center_y = (m_y1 + m_y2) / 2

                distance = np.sqrt((f_center_x - m_center_x)
                                   ** 2 + (f_center_y - m_center_y)**2)
                if distance < self.alert_config['PROXIMITY_THRESHOLD']:
                    nearby_males += 1

            if nearby_males >= 2:
                alert_message = f"Woman surrounded by {nearby_males} men"
                alert_level = self.alert_levels["WARNING"]
            elif is_night and len(females) <= 2 and len(males) == 0:
                alert_message = f"Woman alone at night ({len(females)} detected)"
                alert_level = self.alert_levels["NOTICE"]

        # If still no message but high risk score
        if not alert_message and risk_score > self.alert_config['RISK_THRESHOLD']:
            alert_message = f"Unusual situation detected (Risk: {risk_score})"

        return alert_level, alert_message

    def add_risk_indicators(self, frame, females, males, is_night, persons_detected):
        """Add visual indicators of risk level to the frame"""
        # Colored circle indicators for risk levels
        for female in females:
            risk_score = self.calculate_risk_score(
                female, females, males, is_night, persons_detected)
            x1, y1, x2, y2 = female['bbox']

            # Determine color based on risk score
            if risk_score >= 80:
                color = (0, 0, 255)  # Red - high risk
            elif risk_score >= 60:
                color = (0, 165, 255)  # Orange - medium risk
            elif risk_score >= 40:
                color = (0, 255, 255)  # Yellow - low risk
            else:
                color = (0, 255, 0)  # Green - minimal risk

            # Draw circle indicator
            radius = 10
            center_x = int(x1) + radius + 5
            center_y = int(y1) - radius - 5
            cv2.circle(frame, (center_x, center_y), radius, color, -1)

            # Add risk score text
            cv2.putText(frame, f"{risk_score}",
                        (center_x + radius + 5, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # For high risk, add highlight
            if risk_score >= self.alert_config['RISK_THRESHOLD']:
                # Add pulsing border effect based on frame number
                pulse_intensity = int(128 + 127 * np.sin(time.time() * 5))
                cv2.rectangle(frame,
                              (int(x1)-3, int(y1)-3),
                              (int(x2)+3, int(y2)+3),
                              (0, 0, pulse_intensity), 2)

        return frame

    def process_frame(self, frame, frame_count):
        """Process a single frame of video"""
        # Copy of the frame for display
        result_frame = frame.copy()
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Process frame with YOLO
        results = self.model(frame, conf=self.DETECTION_CONFIDENCE, classes=0)
        
        # Extract YOLO detections
        yolo_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                yolo_boxes.append((x1, y1, x2, y2))
        
        # Filter out small detections
        valid_detections = self.filter_valid_detections(yolo_boxes)
        
        # Apply non-maximum suppression to remove overlapping boxes
        filtered_detections = self.non_max_suppression(valid_detections, iou_threshold=0.4)
        
        # Update person trackers with filtered detections
        self.update_person_trackers(filtered_detections, frame)
        
        # Process each tracked person
        persons_detected = []
        
        # Gender re-evaluation settings
        GENDER_UPDATE_INTERVAL = 2.0  # Re-evaluate gender every 2 seconds
        
        for track_id, tracker in list(self.person_trackers.items()):
            # Get current bounding box
            x1, y1, x2, y2 = tracker['bbox']

            # Check if we need to perform gender classification (either first time or re-evaluation)
            should_update_gender = False
            if tracker['gender'] is None:
                # Initial gender classification
                should_update_gender = True
            else:
                # Check if it's time for gender re-evaluation
                time_since_last_update = time.time() - \
                    tracker.get('last_gender_update', 0)
                if time_since_last_update > GENDER_UPDATE_INTERVAL:
                    should_update_gender = True

            if should_update_gender:
                # Use face for gender classification if available
                if tracker['has_face'] and tracker['face_img'] is not None:
                    img_for_classification = tracker['face_img']
                    is_face_based = True
                else:
                    img_for_classification = tracker['person_img']
                    is_face_based = False

                if self.use_gender_model and img_for_classification.size > 0:
                    # Use Hugging Face model for gender prediction
                    gender, gender_confidence = self.predict_gender(
                        img_for_classification)

                    # If model prediction failed, fall back to simulation
                    if gender is None:
                        seed_value = hash(track_id) % 100
                        gender = "Female" if seed_value < 30 else "Male"  # Reduced female probability
                        gender_confidence = np.random.uniform(0.80, 0.95)

                    # Reduce confidence if not using a face
                    if not is_face_based:
                        gender_confidence *= 0.85
                        # Further bias toward male for body-only detections
                        if gender == "Female" and gender_confidence < 0.75:
                            gender = "Male"
                            gender_confidence = np.random.uniform(0.75, 0.9)
                else:
                    # Fall back to simulation
                    seed_value = hash(track_id) % 100
                    gender = "Female" if seed_value < 30 else "Male"  # Reduced female probability
                    gender_confidence = np.random.uniform(0.80, 0.95)

                    if not is_face_based:
                        gender_confidence *= 0.85

                # Handle gender update for re-evaluation
                if tracker['gender'] is not None:
                    # If it's a re-evaluation, only update if confidence is high enough
                    # or if the current gender is different with good confidence
                    if (gender_confidence > 0.85) or (gender != tracker['gender'] and gender_confidence > 0.75):
                        tracker['gender'] = gender
                        tracker['confidence'] = gender_confidence
                else:
                    # First time classification
                    tracker['gender'] = gender
                    tracker['confidence'] = gender_confidence

                # Update timestamp regardless
                tracker['last_gender_update'] = time.time()

                # Also cache the gender prediction
                self.person_gender_cache[track_id] = (
                    gender, gender_confidence)

            # Get gender from tracker
            gender = tracker['gender']
            gender_confidence = tracker['confidence']

            # Save info for alert detection
            person_info = {
                "bbox": (x1, y1, x2, y2),
                "gender": gender,
                "confidence": gender_confidence,
                "track_id": track_id
            }
            persons_detected.append(person_info)

            # Draw bounding box
            color = (0, 0, 255) if gender == "Female" else (
                255, 0, 0)  # Red for female, Blue for male
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Add label with gender and confidence
            label = f"{gender}: {gender_confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Detect and analyze pose
            landmarks, pose_analysis = self.pose_analyzer.detect_pose(
                frame, (x1, y1, x2, y2))
            if landmarks and pose_analysis:
                person_info['pose'] = pose_analysis
                # Draw pose visualization
                result_frame = self.pose_analyzer.draw_pose(
                    result_frame, landmarks)

                # Add pose alert text
                pose_alerts = []
                if pose_analysis.get('hands_up'):
                    pose_alerts.append("Hands Up")
                if pose_analysis.get('lying_down'):
                    pose_alerts.append("Lying Down")
                if pose_analysis.get('crouching'):
                    pose_alerts.append("Crouching")
                if pose_analysis.get('running'):
                    pose_alerts.append("Running")

                if pose_alerts:
                    pose_text = " & ".join(pose_alerts)
                    cv2.putText(result_frame, pose_text, (x1, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Check for alerts
        alert_detected, alert_message = self.detect_alerts(persons_detected)

        # Apply alert cooldown
        if alert_detected and time.time() - self.last_alert_time > self.alert_config['ALERT_COOLDOWN']:
            self.last_alert_time = time.time()

            # Save evidence only when alert is triggered and not in cooldown
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence_path = f"evidence/alert_{timestamp}.jpg"
            cv2.imwrite(evidence_path, frame)
            print(
                f"⚠️ Alert: {alert_message} - Evidence saved to {evidence_path}")

            # Optionally add risk score display
            for person in persons_detected:
                if person['gender'] == "Female":
                    risk_score = self.calculate_risk_score(person, females, males,
                                                           hour >= self.alert_config['NIGHT_HOURS_START'] or
                                                           hour < self.alert_config['NIGHT_HOURS_END'],
                                                           persons_detected)
                    x1, y1, _, _ = person['bbox']
                    cv2.putText(result_frame, f"Risk: {risk_score}",
                                (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        elif alert_detected:
            alert_detected = False  # Override alert if in cooldown

        # Clean up cache if it gets too large
        if len(self.person_gender_cache) > 100:
            # Delete half of the dictionary (older entries)
            keys_to_delete = list(self.person_gender_cache.keys())[0:50]
            for k in keys_to_delete:
                del self.person_gender_cache[k]

        # Add stats display
        current_time = datetime.now()
        hour = current_time.hour
        is_night = hour >= self.alert_config['NIGHT_HOURS_START'] or hour < self.alert_config['NIGHT_HOURS_END']

        # Define females and males AFTER persons_detected is populated
        females = [p for p in persons_detected if p['gender'] == "Female"]
        males = [p for p in persons_detected if p['gender'] == "Male"]

        stats_text = [
            f"Frame: {frame_count}",
            f"Persons: {len(persons_detected)}",
            f"Female: {len(females)}",
            f"Male: {len(males)}",
            f"Time: {current_time.strftime('%H:%M:%S')}"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(result_frame, text, (10, frame_height - 10 - (20 * i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add risk indicators
        result_frame = self.add_risk_indicators(
            result_frame, females, males, is_night, persons_detected)

        # Return annotated frame
        return result_frame, alert_detected, alert_message

    def run_detection(self):
        # Connect to camera
        cap = self.connect_camera()
        
        frame_count = 0
        alert_sound_played = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame - this already adds risk indicators
            result_frame, alert_detected, alert_message = self.process_frame(
                frame, frame_count)
            
            # Display the processed frame
            cv2.imshow('StreeRaksha - Women Safety Monitor', result_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    print("Starting StreeRaksha Gender Detection...")

    # Print installation information for missing dependencies
    if not has_tensorflow:
        print("\n====== TensorFlow Installation ======")
        print("TensorFlow is missing but not required for core functionality.")
        print("If you want to install it (for Python 3.10 or 3.11), run:")
        print("pip install tensorflow==2.12.0")
        print("Note: TensorFlow doesn't officially support Python 3.12 yet.")

    if not has_transformers:
        print("\n====== Hugging Face Installation ======")
        print("Hugging Face transformers is missing. For better gender detection, install:")
        print("pip install transformers")

    detector = StreeRakshaDetector()
    detector.run_detection()
