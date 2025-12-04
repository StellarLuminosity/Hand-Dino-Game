# 🦖 Dino Game!

Control the Chrome Dino game using real-time hand gesture recognition. Show a **fist** to jump, **peace sign** to duck, and **palm** to idle.

## 🎮 How It Works

A CNN trained on the HAGRID dataset recognizes three hand gestures from your webcam:
- **👊 Fist** → Jump (Space)
- **✌️ Peace** → Duck (Down arrow)
- **🖐️ Palm** → Idle (no action)

The model processes frames in real-time and sends keyboard inputs to control the game.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

This will:
- Download and preprocess the HAGRID dataset
- Train a CNN on palm/peace/fist gestures
- Evaluate on test set and save metrics

### 3. Play!

1. Open [Chrome Dino](chrome://dino) in your browser
2. Run the webcam inference:

```bash
python -m src.webcam
```

3. Position your hand in the green ROI box
4. Make gestures to control the dino!

## Features

- **Real-time inference** with webcam feed
- **Smooth predictions** using majority voting over recent frames
- **Jump lock** prevents accidental double-jumps
- **Visual feedback** showing predictions and confidence scores
- **OpenCV baseline** for comparison with CNN approach

## Model Architecture

Simple CNN with 3 conv blocks:
- Input: 64×64 RGB images
- Conv layers: 32 → 64 → 128 channels
- Fully connected: 256 → num_classes
- Trained with cross-entropy loss

## Configuration

Edit `config.py` to adjust:
- Training hyperparameters (epochs, batch size, learning rate)
- Image preprocessing (size, normalization)
- Jump lock duration
- Dataset split ratios

## Controls

- **`q`** - Quit the application
- **`s`** - Save current ROI frame (with `--save_custom` flag)

## 🛠️ Advanced Usage

```bash
# Use different camera
python -m src.webcam --camera 1

# Adjust prediction smoothing
python -m src.webcam --history 10

# Save custom test frames
python -m src.webcam --save_custom
```

Built with PyTorch, OpenCV, and way too much coffee ☕ :)

