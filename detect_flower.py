"""
CLI Flower Detection & Botanical Information Tool for test.mp4.
Runs AI flower detection, prints detailed species info in Marathi & English,
and exports the annotated video with YOLO bounding boxes and HUD.
"""

import os
import sys
import cv2

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.flower_detector import FlowerDetector, FLOWER_DATABASE


def main():
    video_path = "test.mp4"
    if not os.path.exists(video_path):
        if os.path.exists("test (2).mp4"):
            video_path = "test (2).mp4"
        else:
            print("[ERROR] Neither 'test.mp4' nor 'test (2).mp4' found!")
            return

    os.makedirs("output_videos", exist_ok=True)
    os.makedirs("output_videos/flower_keyframes", exist_ok=True)
    output_video_path = "output_videos/annotated_flower_detection.mp4"

    print("=" * 70)
    print("🌸 DERMA-AI : FLOWER OBJECT DETECTION & BOTANICAL SPECIES IDENTIFIER")
    print("=" * 70)
    print(f"Input Video: {video_path}")
    print(f"Output Video: {output_video_path}\n")

    detector = FlowerDetector()

    def print_progress(ratio: float):
        pct = int(ratio * 100)
        bar = "#" * (pct // 5) + "-" * (20 - (pct // 5))
        sys.stdout.write(f"\rProcessing Frames: [{bar}] {pct}%")
        sys.stdout.flush()

    res = detector.process_flower_video(
        input_video_path=video_path,
        output_video_path=output_video_path,
        frame_stride=1,
        progress_callback=print_progress,
    )

    print("\n\n" + "=" * 70)
    print("✅ FLOWER DETECTION & CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"• Total Video Duration : {res['duration_sec']} seconds ({res['total_frames']} frames)")
    print(f"• Detected Species     : {res['detected_species_count']} Unique Flower Types")
    print(f"• Output Video Saved   : {output_video_path}\n")

    for idx, kf in enumerate(res["species_keyframes"], 1):
        m = kf["metadata"]
        key = kf["flower_key"]

        # Save keyframe image
        kf_filename = f"output_videos/flower_keyframes/species_{idx}_{key}.jpg"
        cv2.imwrite(kf_filename, kf["annotated_frame"])

        print(f"[{idx}] {m.english_name.upper()}  |  {m.marathi_name}")
        print(f"    • Scientific Name : {m.scientific_name}")
        print(f"    • Botanical Family: {m.family}")
        print(f"    • AI Confidence   : {int(kf['confidence'] * 100)}%")
        print(f"    • Appearance      : {m.color_description}")
        print(f"    • Medicinal Uses  : {m.medicinal_uses}")
        print(f"    • Cultural Value  : {m.cultural_significance}")
        print(f"    • Care Tips       : {m.care_tips}")
        print(f"    • Keyframe Image  : {kf_filename}")
        print("-" * 70)

    print(f"\n[SUCCESS] Completed! You can view '{output_video_path}' or open the Web UI.\n")


if __name__ == "__main__":
    main()

