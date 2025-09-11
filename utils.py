# utils.py
import math
import numpy as np
import cv2

# MediaPipe indices used in many examples
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# iris indices used by MediaPipe
LEFT_IRIS_IDX = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]
# mouth indices (top, bottom, left corner, right corner)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# head pose indices (nose tip, left eye corner, right eye corner, left mouth, right mouth, chin-ish)
POSE_IDX = [1, 199, 33, 263, 61, 291]  # reorder slightly for more robust mapping

def _to_pixel_coords(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_landmark_coords(landmarks, indices, img_w, img_h):
    """
    landmarks: MediaPipe landmark list (each has x,y) OR dlib shape-like
    indices: list of indices
    returns list of (x,y) pixel coordinates
    """
    coords = []
    if landmarks is None:
        return coords
    # Try MediaPipe style
    try:
        first = landmarks[0]
        if hasattr(first, 'x'):
            for idx in indices:
                lm = landmarks[idx]
                coords.append(_to_pixel_coords(lm, img_w, img_h))
            return coords
    except Exception:
        pass

    # Fallback: if a dlib-like object (not implemented here fully)
    try:
        for idx in indices:
            p = landmarks.part(idx)
            coords.append((p.x, p.y))
        return coords
    except Exception:
        return coords

def eye_aspect_ratio(landmarks, eye_idx_list, img_w, img_h):
    # eye_idx_list: 6 indices following the EAR formula order
    pts = get_landmark_coords(landmarks, eye_idx_list, img_w, img_h)
    if len(pts) != 6:
        return None
    p1, p2, p3, p4, p5, p6 = pts
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    if C == 0:
        return None
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks, img_w, img_h):
    try:
        top = get_landmark_coords(landmarks, [MOUTH_TOP], img_w, img_h)[0]
        bottom = get_landmark_coords(landmarks, [MOUTH_BOTTOM], img_w, img_h)[0]
        left = get_landmark_coords(landmarks, [MOUTH_LEFT], img_w, img_h)[0]
        right = get_landmark_coords(landmarks, [MOUTH_RIGHT], img_w, img_h)[0]
    except Exception:
        return None
    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    if horizontal == 0:
        return None
    mar = vertical / horizontal
    return mar

def iris_center(landmarks, iris_idx_list, img_w, img_h):
    pts = get_landmark_coords(landmarks, iris_idx_list, img_w, img_h)
    if len(pts) == 0:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (int(np.mean(xs)), int(np.mean(ys)))

def compute_gaze_ratio(landmarks, img_w, img_h):
    try:
        left_eye = get_landmark_coords(landmarks, LEFT_EYE_IDX, img_w, img_h)
        right_eye = get_landmark_coords(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        left_iris = iris_center(landmarks, LEFT_IRIS_IDX, img_w, img_h)
        right_iris = iris_center(landmarks, RIGHT_IRIS_IDX, img_w, img_h)
        if not left_eye or not right_eye or left_iris is None or right_iris is None:
            return None
        left_width = euclidean(left_eye[0], left_eye[3])
        right_width = euclidean(right_eye[0], right_eye[3])
        left_offset = (left_iris[0] - left_eye[0][0]) / (left_width + 1e-6)
        right_offset = (right_iris[0] - right_eye[0][0]) / (right_width + 1e-6)
        left_centered = (left_offset - 0.5)
        right_centered = (right_offset - 0.5)
        return (left_centered + right_centered) / 2.0
    except Exception:
        return None

def head_pose(landmarks, img_w, img_h, camera_matrix=None, dist_coeffs=None):
    # returns (yaw, pitch, roll) in degrees and a label
    if landmarks is None:
        return None, None
    try:
        image_points = get_landmark_coords(landmarks, POSE_IDX, img_w, img_h)
        if len(image_points) != 6:
            return None, None
        model_points = np.array([
            (0.0, 0.0, 0.0),          # nose tip
            (0.0, -330.0, -65.0),     # chin
            (-225.0, 170.0, -135.0),  # left eye left corner
            (225.0, 170.0, -135.0),   # right eye right corner
            (-150.0, -150.0, -125.0), # left mouth corner
            (150.0, -150.0, -125.0)   # right mouth corner
        ], dtype=np.float64)
        image_points = np.array(image_points, dtype=np.float64)
        if camera_matrix is None:
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None, None
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0
        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)
        label = 'frontal'
        if abs(yaw) > 20:
            label = 'left' if yaw > 0 else 'right'
        elif pitch > 15:
            label = 'down'
        elif pitch < -15:
            label = 'up'
        return (yaw, pitch, roll), label
    except Exception:
        return None, None

def ema(prev, value, alpha=0.3):
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev
