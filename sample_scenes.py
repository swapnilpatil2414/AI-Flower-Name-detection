import cv2
import os

cap = cv2.VideoCapture("test.mp4")
os.makedirs("temp_scenes", exist_ok=True)
for i in range(0, 240, 24):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"temp_scenes/frame_{i:03d}.jpg", frame)
        print(f"Saved frame_{i:03d}.jpg")
cap.release()

