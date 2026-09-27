import cv2
import numpy as np

cap = cv2.VideoCapture("test.mp4")
for sec in range(10):
    f_idx = sec * 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Average color in center
        h, w = frame.shape[:2]
        center = hsv[h//4:3*h//4, w//4:3*w//4]
        avg_h = np.mean(center[:, :, 0])
        avg_s = np.mean(center[:, :, 1])
        avg_v = np.mean(center[:, :, 2])
        print(f"Second {sec} (Frame {f_idx}): Avg HSV=({avg_h:.1f}, {avg_s:.1f}, {avg_v:.1f})")
cap.release()

