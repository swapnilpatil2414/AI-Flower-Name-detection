"""
Batch Flower Detection Processor for all videos in 'flower test/' folder.
Processes every video, localizes all flowers, counts them, exports annotated videos
and keyframe snapshots, and generates a comprehensive results report.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from play_slideshow import detect_clip_species, detect_flowers_in_frame, render_slideshow_hud


def process_all_test_videos(folder_path: str = "flower test"):
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder '{folder_path}' not found!")
        return

    valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    videos = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])

    if not videos:
        print(f"[ERROR] No video files found inside '{folder_path}'!")
        return

    out_dir = "output_videos/flower_test_results"
    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(kf_dir, exist_ok=True)

    print("=" * 80)
    print(f"🌸 BATCH FLOWER DETECTION ON ALL VIDEOS IN '{folder_path.upper()}'")
    print("=" * 80)
    print(f"Total Videos to Process: {len(videos)} videos\n")

    summary_results = []

    for idx, v_name in enumerate(videos, 1):
        v_path = os.path.join(folder_path, v_name)
        profile = detect_clip_species(v_path)

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"[{idx:02d}/{len(videos)}] Could not open {v_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 24.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps

        # Output resolution: 1280x720 for fast encoding & universal playback
        out_w = 1280
        scale = out_w / float(w)
        out_h = int(h * scale)

        out_name = f"annotated_{os.path.splitext(v_name)[0]}.mp4"
        out_path = os.path.join(out_dir, out_name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

        frame_idx = 0
        max_flowers = 0
        flower_counts = []
        detected_labels = set()
        best_annotated_frame = None

        print(f"[{idx:02d}/{len(videos)}] Processing: {v_name}")
        print(f"       • File Size     : {os.path.getsize(v_path) / (1024*1024):.2f} MB")
        print(f"       • Duration      : {duration_s:.2f}s ({total_frames} frames)")
        print(f"       • Species Class : {profile['en']} ({profile['mr']})")

        # Process frames (sample every 2 frames for fast batch processing if long)
        step = 2 if total_frames > 400 else 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp = frame_idx / fps
                detections = detect_flowers_in_frame(frame, profile)

                cnt = len(detections)
                flower_counts.append(cnt)
                if cnt > max_flowers:
                    max_flowers = cnt

                for d in detections:
                    detected_labels.add(d["label"])

                annotated = render_slideshow_hud(
                    frame,
                    detections,
                    v_name,
                    idx,
                    len(videos),
                    timestamp,
                    frame_idx,
                    total_frames,
                    profile,
                    display_w=out_w,
                )

                writer.write(annotated)

                if frame_idx >= total_frames // 2 and best_annotated_frame is None:
                    best_annotated_frame = annotated

            frame_idx += 1

        cap.release()
        writer.release()

        # Transcode with imageio_ffmpeg to H.264
        try:
            import imageio_ffmpeg
            import subprocess
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            h264_tmp = out_path.replace(".mp4", "_h264.mp4")
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", out_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", h264_tmp],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(h264_tmp, out_path)
        except Exception:
            pass

        # Save keyframe
        kf_path = os.path.join(kf_dir, f"keyframe_{os.path.splitext(v_name)[0]}.jpg")
        if best_annotated_frame is not None:
            cv2.imwrite(kf_path, best_annotated_frame)

        avg_flowers = int(round(np.mean(flower_counts))) if flower_counts else 0
        labels_str = ", ".join(sorted(detected_labels))

        summary_results.append({
            "idx": idx,
            "filename": v_name,
            "flower_type": profile["en"],
            "marathi_name": profile["mr"],
            "scientific_name": profile["sci"],
            "family": profile["family"],
            "labels": labels_str,
            "max_count": max_flowers,
            "avg_count": avg_flowers,
            "duration": f"{duration_s:.1f}s",
            "output_video": out_path,
            "keyframe": kf_path,
        })

        print(f"       ✔ Result: {max_flowers} Flowers Detected [{profile['en']}]")
        print(f"       ✔ Saved Annotated Video: {out_path}\n")

    # Save report to text file
    report_file = "output_videos/results.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("🌸 AI FLOWER DETECTION & COUNT SUMMARY REPORT (FLOWER TEST FOLDER)\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'#':<3} {'Video Filename':<30} {'Flower (Konte)':<24} {'मराठी नाव':<20} {'Count (Kiti)'}\n")
        f.write("-" * 90 + "\n")
        for r in summary_results:
            cnt_info = f"{r['max_count']} flowers"
            f.write(f"{r['idx']:02d}  {r['filename']:<30} {r['flower_type']:<24} {r['marathi_name']:<20} {cnt_info}\n")
        f.write("=" * 90 + "\n")

    print(f"All {len(summary_results)} videos processed and results saved to '{out_dir}/' and '{report_file}'!\n")

    return summary_results


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "flower test"
    process_all_test_videos(folder)
