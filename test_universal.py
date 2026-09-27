import cv2
import numpy as np

def universal_flower_detect(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Color definitions
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 75, 60), (13, 255, 255)),
        cv2.inRange(hsv, (165, 75, 60), (180, 255, 255))
    )
    mask_pink = cv2.inRange(hsv, (138, 35, 90), (165, 255, 255))
    mask_orange = cv2.inRange(hsv, (13, 90, 80), (24, 255, 255))
    mask_yellow = cv2.inRange(hsv, (22, 60, 90), (36, 255, 255))
    mask_violet = cv2.inRange(hsv, (110, 35, 70), (138, 255, 255))
    mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 45, 255))

    color_masks = [
        ("Red Blossom", mask_red, (50, 60, 240)),
        ("Pink Blossom", mask_pink, (235, 100, 240)),
        ("Orange Blossom", mask_orange, (0, 165, 255)),
        ("Yellow Blossom", mask_yellow, (0, 220, 255)),
        ("Violet Blossom", mask_violet, (255, 180, 50)),
        ("White Blossom", mask_white, (240, 240, 240)),
    ]

    all_raw_boxes = []
    all_scores = []
    all_labels = []
    all_colors = []

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for label, mask, color in color_masks:
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        conts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in conts:
            area = cv2.contourArea(c)
            if 300 < area < 300000:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw > w * 0.95 and bh > h * 0.95:
                    continue
                ar = float(bw) / float(bh)
                if 0.25 < ar < 4.0:
                    pad = 6
                    x1 = max(0, bx - pad)
                    y1 = max(0, by - pad)
                    x2 = min(w - 1, bx + bw + pad)
                    y2 = min(h - 1, by + bh + pad)
                    all_raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    conf = float(min(0.98, 0.88 + area / 80000.0))
                    all_scores.append(conf)
                    all_labels.append(label)
                    all_colors.append(color)

    detections = []
    if all_raw_boxes:
        indices = cv2.dnn.NMSBoxes(all_raw_boxes, all_scores, 0.5, 0.3)
        for i in indices:
            idx_val = i[0] if isinstance(i, (list, np.ndarray)) else i
            bx, by, bw, bh = all_raw_boxes[idx_val]
            detections.append({
                "label": all_labels[idx_val],
                "confidence": all_scores[idx_val],
                "bbox": (bx, by, bx + bw, by + bh),
                "color": all_colors[idx_val],
            })

    return detections

cap = cv2.VideoCapture("test.mp4")
for f_no in [20, 70, 110, 160, 210]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
    ret, frame = cap.read()
    if ret:
        dets = universal_flower_detect(frame)
        print(f"Frame {f_no}: Detected {len(dets)} flowers -> {list(set([d['label'] for d in dets]))}")
cap.release()

