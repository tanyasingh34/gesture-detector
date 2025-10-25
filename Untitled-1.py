"""
AI-Powered Real-Time Gesture-Controlled Keyboard System
--------------------------------------------------------
Controls web platforms (Google, YouTube, Instagram, etc.)
through hand gestures — no keyboard or mouse required.
--------------------------------------------------------
Developed using: OpenCV + MediaPipe + PyAutoGUI
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

# ------------------------- INITIALIZATION -----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1
)

CENTROID_BUFFER = 8
centroids = deque(maxlen=CENTROID_BUFFER)
last_action_time = 0
ACTION_COOLDOWN = 0.4  # seconds

# ------------------------- GESTURE DETECTION ---------------------------

def landmarks_to_feature(landmarks):
    """Convert landmarks to normalized relative coordinates"""
    wrist = landmarks[0]
    pts = []
    for lm in landmarks:
        pts.append(lm.x - wrist.x)
        pts.append(lm.y - wrist.y)
    arr = np.array(pts)
    maxabs = np.max(np.abs(arr)) + 1e-6
    return (arr / maxabs).astype(np.float32)

def detect_static_gesture(landmarks):
    """Simple heuristic for static gestures"""
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    pip = [2, 6, 10, 14, 18]
    folded = 0
    
    for t, p in zip(tips, pip):
        if landmarks[t].y > landmarks[p].y + 0.02:
            folded += 1

    thumb_tip = landmarks[4]
    wrist = landmarks[0]

    # Define gesture rules
    if folded >= 4 and abs(thumb_tip.x - wrist.x) > 0.07:
        return "thumbs_up"
    if folded >= 4 and abs(thumb_tip.x - wrist.x) <= 0.07:
        return "fist"
    if folded <= 1:
        return "open_palm"

    index_extended = (landmarks[8].y < landmarks[6].y - 0.02)
    others_folded = (
        landmarks[12].y > landmarks[10].y + 0.02 and
        landmarks[16].y > landmarks[14].y + 0.02 and
        landmarks[20].y > landmarks[18].y + 0.02
    )
    if index_extended and others_folded:
        return "index_point"

    dist_thumb_index = np.hypot(landmarks[4].x - landmarks[8].x,
                                landmarks[4].y - landmarks[8].y)
    if dist_thumb_index < 0.04:
        return "pinch"

    return None

# ------------------------- ACTION MAPPINGS -----------------------------

def gesture_to_action(gesture):
    """Map gesture to keyboard/mouse action"""
    if gesture == "open_palm":
        pyautogui.hotkey('ctrl', 'l')   # focus address bar
    elif gesture == "fist":
        pyautogui.press('space')        # play/pause
    elif gesture == "thumbs_up":
        pyautogui.press('l')            # like (YouTube)
    elif gesture == "index_point":
        pyautogui.click()               # mouse click
    elif gesture == "pinch":
        pyautogui.press('enter')        # confirm/comment/search

# ------------------------- MAIN LOOP -----------------------------------

def run_gesture_control():
    global last_action_time
    print("🖐️ Starting AI Gesture Control System...")
    print("💡 Focus on your browser window (YouTube/Instagram/Google)")
    print("Press ESC to exit.\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not detected!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        gesture = None

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            landmarks = lm.landmark
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            gesture = detect_static_gesture(landmarks)
            cx = np.mean([p.x for p in landmarks])
            cy = np.mean([p.y for p in landmarks])
            centroids.append((cx, cy, time.time()))

            # Detect swipe gestures
            if len(centroids) >= CENTROID_BUFFER:
                dx = centroids[-1][0] - centroids[0][0]
                dy = centroids[-1][1] - centroids[0][1]
                dt = centroids[-1][2] - centroids[0][2]
                if dt > 0:
                    vx = dx / dt
                    vy = dy / dt
                    if abs(vy) > 0.6 and abs(vy) > abs(vx):
                        gesture = "swipe_up" if vy < 0 else "swipe_down"

        # Perform actions with cooldown
        now = time.time()
        if gesture and now - last_action_time > ACTION_COOLDOWN:
            last_action_time = now
            print(f"✅ Gesture Detected: {gesture}")
            if gesture == "swipe_up":
                pyautogui.scroll(400)
            elif gesture == "swipe_down":
                pyautogui.scroll(-400)
            else:
                gesture_to_action(gesture)

        # Overlay
        if gesture:
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("AI Gesture-Controlled Keyboard", frame)

        # Exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 System stopped. Goodbye!")

# ------------------------- ENTRY POINT ---------------------------------

if __name__ == "__main__":
    run_gesture_control()
