import cv2
import numpy as np

cap = cv2.VideoCapture("test.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames: {total_frames}, FPS: {fps}")

# Check color histograms to locate exact scene cuts
prev_hist = None
scene_cuts = [0]

for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    if prev_hist is not None:
        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
        if diff > 150.0:  # Scene cut threshold
            scene_cuts.append(i)
            print(f"Scene cut detected at frame {i} ({i/fps:.2f}s), diff: {diff:.1f}")
    prev_hist = hist

cap.release()
print(f"Detected scene cuts at frames: {scene_cuts}")

