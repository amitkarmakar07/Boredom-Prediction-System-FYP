# calibrate.py
import time
import argparse
import json
import cv2
import mediapipe as mp
import numpy as np
from utils import eye_aspect_ratio, mouth_aspect_ratio, compute_gaze_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
from config import DEFAULTS, save_config


def run_calibration(duration_sec=30, out_path='config.json'):
    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    t0 = time.time()
    e_ears = []
    e_mars = []
    gaze_offsets = []

    print('Calibration started — please look at the screen naturally for {} seconds'.format(duration_sec))

    while time.time() - t0 < duration_sec:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # Calculate eye aspect ratios
            left_ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            if left_ear is not None and right_ear is not None:
                e_ears.append((left_ear + right_ear) / 2.0)

            # Calculate mouth aspect ratio
            mar = mouth_aspect_ratio(lm, w, h)
            if mar is not None:
                e_mars.append(mar)

            # Calculate gaze ratio
            try:
                g = compute_gaze_ratio(lm, w, h)
                if g is not None:
                    gaze_offsets.append(abs(g))
            except Exception:
                pass

        # Display countdown
        time_left = max(0, duration_sec - (time.time() - t0))
        cv2.putText(frame, 'Calibrating: {:.0f}s left'.format(time_left),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate calibration values
    ear_mean = float(np.median(e_ears)) if e_ears else DEFAULTS['ear_blink_thresh']
    mar_mean = float(np.median(e_mars)) if e_mars else DEFAULTS['mar_yawn_thresh']
    gaze_median = float(np.median(gaze_offsets)) if gaze_offsets else DEFAULTS['gaze_threshold']

    # Create configuration with calibrated thresholds
    cfg = DEFAULTS.copy()
    cfg['ear_blink_thresh'] = max(0.12, ear_mean * 0.7)
    cfg['mar_yawn_thresh'] = max(0.45, mar_mean * 1.5)
    cfg['gaze_threshold'] = max(0.15, gaze_median * 1.2)

    # Save configuration
    save_config(out_path, cfg)
    print('Calibration completed. Thresholds saved to', out_path)
    print(json.dumps(cfg, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=30, help='Calibration duration in seconds')
    parser.add_argument('--out', type=str, default='config.json', help='Output configuration file path')
    args = parser.parse_args()
    run_calibration(args.duration, args.out)