"""
AI Flower Detection - Live Interactive Video Player (Clean Full Video YOLO Detection).
Plays test.mp4 in real-time with tactical bounding boxes, English species identification,
and zero visual obstructions.
"""

import os
import sys
import time
import cv2
import numpy as np

from core.flower_detector import FlowerDetector, FLOWER_DATABASE


def play_flower_video(video_path: str = "test.mp4", loop: bool = True):
    if not os.path.exists(video_path):
        if os.path.exists("test (2).mp4"):
            video_path = "test (2).mp4"
        else:
            print(f"[ERROR] Video file '{video_path}' not found!")
            return

    detector = FlowerDetector()
    window_name = "AI Flower Detection & Species Identification - Full Video Player"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("=" * 70)
    print("AI FLOWER DETECTION - FULL VIDEO REAL-TIME PLAYER")
    print("=" * 70)
    print(f"Video Source : {video_path}")
    print("Interactive Controls:")
    print("  [SPACE]      : Pause / Resume playback")
    print("  [R]          : Restart video from Frame 0")
    print("  [S]          : Save annotated snapshot frame (JPEG)")
    print("  [Q] or [ESC] : Quit / Close player window")
    print("=" * 70 + "\n")

    os.makedirs("output_videos/snapshots", exist_ok=True)

    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_path}")
            break

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 24.0

        delay_ms = max(1, int(1000.0 / fps))
        frame_idx = 0
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Video completed
                    break

                timestamp = frame_idx / fps
                detections = detector.detect_frame(frame, frame_idx, fps)
                annotated = detector.draw_flower_annotations(frame, detections, timestamp, frame_idx)

                cv2.imshow(window_name, annotated)
                frame_idx += 1

            # Keyboard handler
            key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                cap.release()
                cv2.destroyAllWindows()
                print("\n[INFO] Player closed by user.")
                return
            elif key == ord(" "):
                paused = not paused
                status = "PAUSED" if paused else "PLAYING"
                print(f"[INFO] Status: {status}")
            elif key in [ord("r"), ord("R")]:
                frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
                print("[INFO] Video Restarted")
            elif key in [ord("s"), ord("S")]:
                snap_path = f"output_videos/snapshots/snapshot_frame_{frame_idx}.jpg"
                cv2.imwrite(snap_path, annotated)
                print(f"[SAVED] {snap_path}")

        cap.release()
        if not loop:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    vpath = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    play_flower_video(vpath, loop=True)
