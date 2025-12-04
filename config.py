import torch

random_seed = 42

# Dataset configuration
dataset_name = "innominate817/hagrid-sample-30k-384p"
annotations_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
data_dir = "data"
annotations_path = "data/hagrid_annotations"

target_classes = ["palm", "peace", "fist"]
split_ratios = {"train": 0.7, "test": 0.15, "val": 0.15}

img_extensions = {".jpg", ".jpeg", ".png"}
img_extensions_torch = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Training configuration
epochs = 10
batch_size = 64
learning_rate = 1e-3
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "checkpoints"
model_dir = "models"
cnn_metrics_output = "figs/cnn_metrics.json"
metrics_output = "figs/baseline_metrics.json"

# Image preprocessing
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
image_size = (64, 64)

# inference config
jump_lock_duration = 0.9  # in seconds
