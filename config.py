# config.py
import json

DEFAULTS = {
    "window_sec": 10,
    "calibration_duration": 30,
    "ear_blink_thresh": 0.23,  # reasonable default; calibration will adapt
    "ear_ema_alpha": 0.3,
    "mar_yawn_thresh": 0.6,    # typical MAR threshold used in many tutorials
    "mar_ema_alpha": 0.3,
    "gaze_threshold": 0.35,    # normalized pupil offset below which gaze is on-screen
    "min_frames_required": 3,  # min frames to consider detection valid
}

def save_config(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)
