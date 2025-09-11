# capture.py

import time
import os
import csv
from pathlib import Path
import cv2
import mediapipe as mp
from deepface import DeepFace
from utils import *
from config import DEFAULTS, load_config
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

OUT_DIR = Path('output')
IM_DIR = OUT_DIR / 'images'
CSV_PATH = OUT_DIR / 'data.csv'

OUT_DIR.mkdir(exist_ok=True)
IM_DIR.mkdir(parents=True, exist_ok=True)

CFG_PATH = Path('config.json')
config = DEFAULTS.copy()
if CFG_PATH.exists():
    try:
        config = load_config(str(CFG_PATH))
    except Exception:
        pass


def write_csv_header(path):
    if not path.exists():
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'face_image', 'emotion', 'blink_rate', 'yawn_rate', 'gaze_ratio', 'head_pose',
                             'head_movement_rate'])


def main():
    write_csv_header(CSV_PATH)
    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window_sec = config.get('window_sec', 10)

    ear_ema = None
    mar_ema = None

    blink_count = 0
    blink_in_progress = False
    yawn_count = 0
    yawn_in_progress = False

    gaze_on_count = 0
    face_present_count = 0
    total_frames = 0

    head_last_label = None
    head_movement_count = 0

    candidate_faces = []

    window_start = time.time()
    frame_idx = 0

    print("Starting capture. Press 'q' in the window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame_idx += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        total_frames += 1

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            face_present_count += 1
            left_ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            if left_ear is not None and right_ear is not None:
                ear = (left_ear + right_ear) / 2.0
                ear_ema = ema(ear_ema, ear, config.get('ear_ema_alpha', 0.3))
                if ear_ema < config['ear_blink_thresh'] and not blink_in_progress:
                    blink_in_progress = True
                if ear_ema >= config['ear_blink_thresh'] and blink_in_progress:
                    blink_count += 1
                    blink_in_progress = False
            mar = mouth_aspect_ratio(lm, w, h)
            if mar is not None:
                mar_ema = ema(mar_ema, mar, config.get('mar_ema_alpha', 0.3))
                if mar_ema > config['mar_yawn_thresh'] and not yawn_in_progress:
                    yawn_in_progress = True
                if mar_ema <= config['mar_yawn_thresh'] and yawn_in_progress:
                    yawn_count += 1
                    yawn_in_progress = False
            g = compute_gaze_ratio(lm, w, h)
            if g is not None:
                if abs(g) < config['gaze_threshold']:
                    gaze_on_count += 1
            angles, label = head_pose(lm, w, h)
            if label is not None:
                if head_last_label is None:
                    head_last_label = label
                elif label != head_last_label:
                    head_movement_count += 1
                    head_last_label = label
            # crop
            xs = [int(p.x * w) for p in lm]
            ys = [int(p.y * h) for p in lm]
            xmin, xmax = max(min(xs) - 10, 0), min(max(xs) + 10, w)
            ymin, ymax = max(min(ys) - 10, 0), min(max(ys) + 10, h)
            if xmax > xmin and ymax > ymin:
                area = (xmax - xmin) * (ymax - ymin)
                candidate_faces.append((area, frame[ymin:ymax, xmin:xmax]))

        # window end - YOUR DESIRED LOGIC IMPLEMENTED
        if time.time() - window_start >= window_sec:
            timestamp = int(time.time())
            face_image_path = ''
            emotion = 'detection_issues'
            if face_present_count >= total_frames * 0.5 and len(candidate_faces) > 0:
                best = sorted(candidate_faces, key=lambda x: x[0], reverse=True)[0]
                img = best[1]
                fname = f'{timestamp}.jpg'
                fpath = IM_DIR / fname
                cv2.imwrite(str(fpath), img)
                face_image_path = str(fpath)
                try:
                    rgb_face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    df = DeepFace.analyze(rgb_face, actions=['emotion'], enforce_detection=False)
                    if isinstance(df, list):
                        df = df[0]
                    emotion = df.get('dominant_emotion', 'unknown')
                except Exception:
                    emotion = 'unknown'
            else:
                face_image_path = ''
                emotion = 'detection_issues'

            # YOUR DESIRED CALCULATION METHOD - Count per second rates
            blink_rate = round(blink_count / window_sec, 3)  # Blinks per second
            yawn_rate = round(yawn_count / window_sec, 3)  # Yawns per second
            head_movement_rate = round(head_movement_count / window_sec, 3)  # Movements per second
            gaze_ratio = round(gaze_on_count / max(1, total_frames), 3)  # Gaze ratio (0-1)

            final_head_pose = head_last_label if head_last_label else 'unknown'

            # Store the per-second rates
            row = [timestamp, face_image_path, emotion, blink_rate, yawn_rate,
                   gaze_ratio, final_head_pose, head_movement_rate]

            with open(CSV_PATH, 'a', newline='') as f:
                import csv as _csv
                writer = _csv.writer(f)
                writer.writerow(row)

            # reset window
            window_start = time.time()
            blink_count = 0
            blink_in_progress = False
            yawn_count = 0
            yawn_in_progress = False
            gaze_on_count = 0
            face_present_count = 0
            total_frames = 0
            candidate_faces = []
            head_last_label = None
            head_movement_count = 0

        cv2.imshow('Capture (press q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()