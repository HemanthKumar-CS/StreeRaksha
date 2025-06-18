"""
StreeRaksha Visualization Module
Handles UI rendering and visualization.
"""

import cv2
import numpy as np
from datetime import datetime


class Visualizer:
    def __init__(self):
        """Initialize the visualizer"""
        pass

    def draw_detection_boxes(self, frame, persons):
        """Draw detection boxes for all persons"""
        for person in persons:
            # Extract info from tracker
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            gender = person.get('gender', 'Unknown')
            gender_confidence = person.get('gender_confidence', 0)
            track_id = person.get('track_id', 0)

            # Draw person bounding box
            label_color = (255, 0, 0) if gender == "Male" else (
                255, 0, 255)  # Blue for male, pink for female
            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)

            # Draw person label
            gender_label = f"{gender} ({gender_confidence:.2f})"
            cv2.putText(frame, f"ID:{track_id} {gender_label}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        label_color, 1, cv2.LINE_AA)

            # Draw pose labels if any pose detected
            if 'pose' in person:
                pose = person['pose']
                pose_text = []
                if pose.get('hands_up', False):
                    pose_text.append("Hands Up")
                if pose.get('lying_down', False):
                    pose_text.append("Lying Down")
                if pose.get('running', False):
                    pose_text.append("Running")
                if pose.get('crouching', False):
                    pose_text.append("Crouching")

                if pose_text:
                    cv2.putText(frame, ", ".join(pose_text),
                                (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, label_color, 1, cv2.LINE_AA)

        return frame

    def draw_stats(self, frame, frame_count, persons, current_time=None):
        """Draw statistics overlay"""
        if current_time is None:
            current_time = datetime.now()

        # Calculate statistics
        females = [p for p in persons if p.get('gender') == "Female"]
        males = [p for p in persons if p.get('gender') == "Male"]

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Prepare stats text
        stats_text = [
            f"Frame: {frame_count}",
            f"Persons: {len(persons)}",
            f"Female: {len(females)}",
            f"Male: {len(males)}",
            f"Time: {current_time.strftime('%H:%M:%S')}"
        ]

        # Draw stats at the bottom of the frame
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text,
                        (10, frame_height - 20 - i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_alert(self, frame, alert_message):
        """Draw alert message"""
        if alert_message:
            # Add large alert message at the top
            cv2.putText(frame, alert_message,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

        return frame
