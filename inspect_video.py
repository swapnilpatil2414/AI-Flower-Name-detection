import cv2
import os
import shutil

src = "test (2).mp4"
if os.path.exists(src) and not os.path.exists("test.mp4"):
    shutil.copy(src, "test.mp4")
    print("Copied to test.mp4")

video_file = "test.mp4" if os.path.exists("test.mp4") else src
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video info: {w}x{h}, {fps} fps, {count} frames, {count/max(fps, 1):.2f}s")

os.makedirs("temp_frames", exist_ok=True)
sample_indices = [0, max(0, count//4), max(0, count//2), max(0, 3*count//4), max(0, count-1)]
for i in set(sample_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        out_name = f"temp_frames/frame_{i}.jpg"
        cv2.imwrite(out_name, frame)
        print(f"Saved {out_name}")
cap.release()

