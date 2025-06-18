# pose_analyzer.py

import cv2
import numpy as np
import mediapipe as mp


class PoseAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Configuration thresholds
        self.config = {
            'HANDS_UP_THRESHOLD': 0.4,    # Threshold for detecting hands up position
            'LYING_DOWN_THRESHOLD': 0.3,   # Threshold for detecting lying down
            'PROXIMITY_THRESHOLD': 120,    # Pixel distance for proximity alerts
            'KNEE_BEND_THRESHOLD': 120,    # Angle threshold for crouching detection
            'RUNNING_STRIDE_THRESHOLD': 0.2  # Relative threshold for running detection
        }

    def detect_pose(self, frame, person_box):
        """Detect and analyze pose within a person bounding box"""
        try:
            # Extract person region
            x1, y1, x2, y2 = person_box
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None, None

            # Convert to RGB for MediaPipe
            rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # Process the image
            results = self.pose_detector.process(rgb_img)

            if not results.pose_landmarks:
                return None, None

            # Normalize landmarks to the person box coordinates
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                px = int(landmark.x * person_img.shape[1]) + x1
                py = int(landmark.y * person_img.shape[0]) + y1
                visibility = landmark.visibility
                landmarks.append((px, py, visibility))

            # Analyze pose
            pose_analysis = self.analyze_pose(landmarks)

            return landmarks, pose_analysis

        except Exception as e:
            print(f"Error in pose detection: {e}")
            return None, None

    def analyze_pose(self, landmarks):
        """Analyze pose landmarks to detect specific poses"""
        if not landmarks or len(landmarks) < 33:  # MediaPipe provides 33 landmarks
            return {}

        analysis = {
            'hands_up': False,
            'lying_down': False,
            'crouching': False,
            'running': False
        }

        # Extract key landmarks
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # Check hands up position
        if (left_wrist[1] < left_shoulder[1] - self.config['HANDS_UP_THRESHOLD'] * (left_hip[1] - left_shoulder[1]) and
            left_wrist[2] > 0.5) or (right_wrist[1] < right_shoulder[1] - self.config['HANDS_UP_THRESHOLD'] * (right_hip[1] - right_shoulder[1]) and
                                     right_wrist[2] > 0.5):
            analysis['hands_up'] = True

        # Check lying down position
        shoulder_hip_angle = abs(
            (left_shoulder[1] + right_shoulder[1])/2 -
            (left_hip[1] + right_hip[1])/2
        ) / ((left_hip[1] - left_shoulder[1]) if left_hip[1] != left_shoulder[1] else 1)

        if shoulder_hip_angle < self.config['LYING_DOWN_THRESHOLD']:
            analysis['lying_down'] = True

        # Check crouching position
        knee_angle = self.calculate_knee_bend(landmarks)
        if knee_angle and knee_angle < self.config['KNEE_BEND_THRESHOLD']:
            analysis['crouching'] = True

        # Check running motion
        ankle_distance = (
            (left_ankle[0] - right_ankle[0])**2 + (left_ankle[1] - right_ankle[1])**2)**0.5
        ankle_height_diff = abs(left_ankle[1] - right_ankle[1])
        hip_width = abs(right_hip[0] - left_hip[0])

        if (ankle_distance > self.config['RUNNING_STRIDE_THRESHOLD'] * hip_width and
                ankle_height_diff > 20):
            analysis['running'] = True

        return analysis

    def calculate_knee_bend(self, landmarks):
        """Calculate approximate knee bend angle"""
        try:
            # Get hip, knee, ankle landmarks
            left_hip = np.array(landmarks[23][:2])
            left_knee = np.array(landmarks[25][:2])
            left_ankle = np.array(landmarks[27][:2])

            # Calculate vectors
            hip_to_knee = left_knee - left_hip
            ankle_to_knee = left_knee - left_ankle

            # Calculate angle
            dot_product = np.dot(hip_to_knee, ankle_to_knee)
            norm_product = np.linalg.norm(
                hip_to_knee) * np.linalg.norm(ankle_to_knee)

            if norm_product == 0:
                return None

            angle_rad = np.arccos(
                np.clip(dot_product / norm_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            return angle_deg
        except:
            return None

    def draw_pose(self, frame, landmarks):
        """Draw pose skeleton and landmarks on the frame"""
        if not landmarks:
            return frame

        # Define connections for skeleton
        connections = [
            # Torso
            (11, 12), (12, 24), (24, 23), (23, 11),
            # Right arm
            (12, 14), (14, 16),
            # Left arm
            (11, 13), (13, 15),
            # Right leg
            (24, 26), (26, 28),
            # Left leg
            (23, 25), (25, 27)
        ]

        # Draw skeleton
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][0]), int(
                    landmarks[start_idx][1]))
                end_point = (int(landmarks[end_idx][0]),
                             int(landmarks[end_idx][1]))

                # Color coding
                if connection in [(11, 12), (12, 24), (24, 23), (23, 11)]:  # Torso
                    color = (0, 255, 255)  # Yellow
                elif connection in [(12, 14), (14, 16), (11, 13), (13, 15)]:  # Arms
                    color = (0, 165, 255)  # Orange
                else:  # Legs
                    color = (255, 255, 0)  # Cyan

                cv2.line(frame, start_point, end_point, color, 2)

        # Draw landmarks
        for landmark in landmarks:
            x, y, visibility = landmark
            if visibility > 0.5:  # Only draw visible landmarks
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        return frame
