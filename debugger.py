"""
StreeRaksha Debug Utility Module
Provides debug utilities for the StreeRaksha system.
"""

import cv2
import numpy as np
import time
import os


class Debugger:
    def __init__(self, logger=None):
        """Initialize the debugger"""
        self.logger = logger
        self.debug_overlay = True
        self.debug_dir = "debug"
        self.frame_history = []
        self.max_frames = 10  # Maximum frames to keep in history

        # Create debug directory
        os.makedirs(self.debug_dir, exist_ok=True)

        # Performance monitoring
        self.fps_history = []
        self.last_time = time.time()

    def log_fps(self):
        """Calculate and log FPS"""
        current_time = time.time()
        fps = 1.0 / \
            (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
        self.last_time = current_time

        # Keep a history of FPS values
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)

        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        if self.logger:
            self.logger.debug(f"FPS: {avg_fps:.2f}")

        return avg_fps

    def save_debug_frame(self, frame, suffix="debug"):
        """Save a debug frame"""
        filename = os.path.join(
            self.debug_dir, f"{suffix}_{time.time():.2f}.jpg")
        cv2.imwrite(filename, frame)
        if self.logger:
            self.logger.debug(f"Saved debug frame to {filename}")

    def add_debug_overlay(self, frame, info_dict):
        """Add debug overlay to frame"""
        if not self.debug_overlay:
            return frame

        # Create a copy of the frame
        debug_frame = frame.copy()

        # Add FPS
        fps = self.log_fps()
        cv2.putText(debug_frame, f"FPS: {fps:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Add any additional debug information
        y_offset = 70
        for key, value in info_dict.items():
            cv2.putText(debug_frame, f"{key}: {value}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 30

        return debug_frame

    def add_frame_to_history(self, frame):
        """Add frame to history"""
        self.frame_history.append(frame.copy())
        if len(self.frame_history) > self.max_frames:
            self.frame_history.pop(0)

    def save_frame_history(self):
        """Save all frames in history"""
        for i, frame in enumerate(self.frame_history):
            self.save_debug_frame(frame, f"history_{i}")

    def draw_track_history(self, frame, persons):
        """Draw track history lines"""
        # Implement track visualization
        return frame
