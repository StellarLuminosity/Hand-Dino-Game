# pseudo-structure you can paste into a file
from pathlib import Path
import cv2, numpy as np

CLASSES = ["palm","fist","peace"]
VAL_DIR = Path("data/val")

def predict_one(img_bgr):
    # 1) preprocess
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # basic skin mask (tweak quickly per your lighting)
    mask = cv2.inRange(hsv, (0, 30, 60), (20, 170, 255))
    mask = cv2.medianBlur(mask, 7)

    # 2) largest contour
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return "peace"  # empty → neutral fallback
    c = max(cnts, key=cv2.contourArea)

    # 3) convex hull + defects
    hull = cv2.convexHull(c, returnPoints=False)
    if hull is None or len(hull) < 3:  # degenerate
        return "fist"

    defects = cv2.convexityDefects(c, hull)
    num_def = 0 if defects is None else defects.shape[0]

    # 4) tiny rule-of-thumb (adjust quickly):
    if num_def >= 3:  return "palm"   # open hand → jump
    if num_def <= 1:  return "fist"   # closed → duck
    return "peace"                    # otherwise neutral
    # See OpenCV convexity defects tutorial for rationale. 

def eval_folder():
    correct = total = 0
    per_class = {k:[0,0] for k in CLASSES}  # [correct, total]
    for cname in CLASSES:
        for p in (VAL_DIR/cname).glob("*"):
            img = cv2.imread(str(p))
            if img is None: continue
            pred = predict_one(img)
            correct += int(pred == cname); total += 1
            pc = per_class[cname]; pc[0]+=int(pred==cname); pc[1]+=1
    acc = correct/total if total else 0.0
    print(f"baseline val-acc: {acc:.3f}")
    for cname,(ok,tot) in per_class.items():
        print(f"{cname}: {ok}/{tot} = {ok/max(1,tot):.3f}")

if __name__ == "__main__":
    eval_folder()
