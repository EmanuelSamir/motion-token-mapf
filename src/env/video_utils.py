import gymnasium as gym
import cv2
import numpy as np
from typing import Any, Dict, Tuple

class HUDWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper that adds a time overlay to the rendered frames.
    Synchronized with the A* planner steps (dt=0.2s).
    """
    def __init__(self, env: gym.Env, dt: float = 0.2):
        super().__init__(env)
        self.dt = dt
        self.current_step = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.color = (255, 255, 255) # White text
        self.shadow_color = (0, 0, 0) # Black shadow
        self.thickness = 2

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        self.current_step = 0
        return self.env.reset(**kwargs)

    def render(self) -> Any:
        frame = self.env.render()
        
        # Only process if we have an RGB array
        if frame is not None and isinstance(frame, np.ndarray):
            # IMPORTANT: Ensure frame is a mutable, contiguous copy for OpenCV
            frame = np.ascontiguousarray(frame.copy())
            
            # --- Diagnostic Border (Neon Green) ---
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 4)

            # --- Simulation Time HUD ---
            sim_time = self.current_step * self.dt
            cv2.putText(frame, f"TIME: {sim_time:.1f}s | STEP: {self.current_step}", (30, 40), self.font, 
                        0.8, (255, 255, 0), 2, cv2.LINE_AA) # Yellow Time
            
            # --- Precision X-Axis Ruler (Bottom Layout) ---
            # Robust manual projection based on environment config (works even without viewer)
            config = getattr(self.env.unwrapped, "config", {})
            roi_center = config.get("roi_center", 0)
            scaling = config.get("scaling", 1.0)
            width = frame.shape[1]
            
            # Draw a ruler at the bottom (yellow background bar)
            r_y = frame.shape[0] - 40
            cv2.rectangle(frame, (0, r_y - 30), (frame.shape[1], r_y + 10), (0, 255, 255), -1)
            
            # Marks every 10m
            for x_m in range(0, 1000, 10):
                # Manual Projection Formula for fixed camera
                px = int((x_m - roi_center) * scaling + width / 2)
                
                if 0 <= px < frame.shape[1]:
                    # Draw tick
                    h = 20 if x_m % 50 == 0 else 10
                    cv2.line(frame, (px, r_y + 10), (px, r_y + 10 - h), (0, 0, 0), 2)
                    # Draw label every 10m
                    cv2.putText(frame, f"{x_m}", (px - 10, r_y - 5), self.font,
                                0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
        return frame
