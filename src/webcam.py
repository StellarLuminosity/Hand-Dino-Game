import argparse
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import torch
from PIL import Image
from torchvision import transforms

import config

from .model import HandGestureCNN


def load_model():
    ckpt_path = os.path.join(config.model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. " "Run training (main.py) first."
        )

    ckpt = torch.load(ckpt_path, map_location=config.device)
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is None:
        raise KeyError("Checkpoint missing 'class_to_idx' – did train.py save it?")

    num_classes = len(class_to_idx)
    model = HandGestureCNN(num_classes=num_classes).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print("Loaded model from:", ckpt_path)
    print("Class mapping:", class_to_idx)
    return model, class_to_idx, idx_to_class


def build_transform():
    """
    Validation-style transform for webcam frames.
    """
    return transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std),
        ]
    )


def preprocess_frame(frame_bgr, roi):
    """
    Crop ROI from full BGR frame, convert to RGB PIL Image, and apply transform.
    """
    x1, y1, x2, y2 = roi
    h, w, _ = frame_bgr.shape

    # Clamp ROI to frame
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # BGR -> RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    transform = build_transform()
    tensor = transform(pil_img)
    tensor = tensor.unsqueeze(0)
    return tensor, crop


def majority_vote(history):
    """
    Given a deque of recent predictions, return the most common label.
    """
    if not history:
        return None
    counts = Counter(history)
    return counts.most_common(1)[0][0]


def gesture_to_action(gesture):
    """
    Map gesture label --> dino action string
    Returns one of {"idle", "jump", "duck"}
    """
    if gesture == "fist":
        return "jump"
    elif gesture == "peace":
        return "duck"
    elif gesture == "palm":
        return "idle"
    else:
        # fallback
        return "idle"


# def send_key_for_action(action, last_action, cooldown_frames, frame_since_action):
#     """
#     Send pyautogui keypresses based on action.
#     Uses a cooldown so that keys aren't spammed every frame
#     """
#     # Only trigger if action changed and cooldown has passed
#     if action == last_action and frame_since_action < cooldown_frames:
#         return last_action, frame_since_action + 1
#
#     if action == "jump":
#         pyautogui.press("space")
#         print("JUMP")
#         return action, 0
#     elif action == "duck":
#         pyautogui.keyDown("down")
#         time.sleep(0.05)
#         pyautogui.keyUp("down")
#         print("DUCK")
#         return action, 0
#     else:
#         # idle: do nothing
#         return last_action, frame_since_action + 1


def update_keys_for_action(action, last_jump_time):
    """
    Treat gestures as held keys:
      - 'jump' -> hold UP
      - 'duck' -> hold DOWN
      - 'idle' -> release both
    """
    now = time.time()
    in_jump_lock = (now - last_jump_time) < config.jump_lock_duration

    if action == "jump" and not in_jump_lock:
        pyautogui.press("space")
        print("[ACTION] JUMP")
        last_jump_time = now

    elif action == "duck" and not in_jump_lock:
        pyautogui.keyDown("down")
        time.sleep(0.05)
        pyautogui.keyUp("down")
        print("[ACTION] DUCK")

    # 'idle' or anything during jump lock --> do nothing
    return last_jump_time


def maybe_save_frame(crop_bgr, gesture, out_root):
    """
    Save current crop to data/custom_test/<gesture> with timestamp.
    """
    if crop_bgr is None or gesture is None:
        return

    class_dir = out_root / gesture
    class_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    fname = class_dir / f"{gesture}_{ts}.jpg"
    cv2.imwrite(str(fname), crop_bgr)
    print(f"Saved frame to {fname}")


def main():
    print("Starting Hand Dino webcam application...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera", type=int, default=0, help="Webcam index (default: 0)"
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of frames to use for majority vote",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=1,
        help="Cooldown in frames between repeated key actions",
    )
    parser.add_argument(
        "--save_custom",
        action="store_true",
        help="If set, pressing 's' will save ROI frames to data/custom_test/",
    )
    args = parser.parse_args()

    print(f"Camera index: {args.camera}")
    print(f"History size: {args.history}")
    print(f"Cooldown frames: {args.cooldown}")
    print(f"Save custom frames: {args.save_custom}")
    print(f"Device: {config.device}")

    print("Loading model...")
    device = config.device
    model, class_to_idx, idx_to_class = load_model()
    print(f"Model loaded successfully!")

    roi = None

    print(f"Attempting to open camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index.")

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened successfully! Resolution: {width}x{height}, FPS: {fps}")

    print("\nPress 'q' to quit.")
    if args.save_custom:
        print("Press 's' to save current ROI frame.")
    print("Starting main loop...\n")

    pred_history = deque(maxlen=args.history)
    last_action = "idle"
    frames_since_action = 0
    last_jump_time = 0.0
    frame_count = 0

    custom_root = Path("data/custom_test")

    print("Entering main processing loop...")
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        h, w, _ = frame.shape

        if roi is None:
            roi_width = int(0.25 * w)
            roi_height = int(0.35 * h)

            center_x = w // 2
            center_y = int(h * 0.65)  # a bit below center

            x1 = max(center_x - roi_width // 2, 0)
            y1 = max(center_y - roi_height // 2, 0)
            x2 = min(center_x + roi_width // 2, w)
            y2 = min(center_y + roi_height // 2, h)
            roi = (x1, y1, x2, y2)
            print(f"Initialized ROI: ({x1}, {y1}) to ({x2}, {y2})")

        # Draw ROI rectangle for user guidance
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Preprocess ROI for model
        result = preprocess_frame(frame, roi)
        if result is None:
            if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                print(f"Frame {frame_count}: Empty ROI crop, skipping inference")
            cv2.imshow("Hand Dino - CNN", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        tensor, crop_bgr = result
        cv2.imshow("ROI", crop_bgr)
        tensor = tensor.to(device)

        # Model inference
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            gesture = idx_to_class.get(pred_idx, None)

        # Debug: show probabilities on screen
        prob_text = " ".join(
            f"{idx_to_class[i]}:{probs[i]:.2f}" for i in range(len(idx_to_class))
        )
        cv2.putText(
            frame,
            prob_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        pred_history.append(gesture)
        smooth_gesture = majority_vote(pred_history)

        # Print prediction info periodically or when it changes
        if frame_count % 30 == 0 or smooth_gesture != (
            pred_history[-2] if len(pred_history) > 1 else None
        ):
            print(
                f"Frame {frame_count}: Raw={gesture}, Smoothed={smooth_gesture}, History={list(pred_history)}"
            )

        # Map to action and maybe send key
        action = gesture_to_action(smooth_gesture)
        if action != last_action:
            print(
                f"Gesture '{smooth_gesture}' -> Action '{action}' (was '{last_action}')"
            )

        if smooth_gesture == "fist":
            action = "jump"
        elif smooth_gesture == "peace":
            action = "duck"
        else:
            action = "idle"

        last_jump_time = update_keys_for_action(action, last_jump_time)

        # Overlay prediction on frame
        text = f"pred: {smooth_gesture} (raw: {gesture})"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Dino - CNN", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"\nQuit requested. Processed {frame_count} frames total.")
            break
        if key == ord("s") and args.save_custom:
            maybe_save_frame(crop_bgr, smooth_gesture, custom_root)
        elif key == ord("s") and not args.save_custom:
            print(
                "'s' pressed but --save_custom not enabled. Enable with --save_custom flag."
            )

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Exiting.")


if __name__ == "__main__":
    main()
