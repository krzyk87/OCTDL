import random
import os
import numpy as np
import hydra
from omegaconf import DictConfig
from utils.func import *
from train import evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model

# Imports for Grad-CAM visualization
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
from torch.nn import functional as F

"""
Test script for evaluating a saved model on the test dataset.

This script loads a model from a specified checkpoint and evaluates it on the test dataset.
The model path can be specified in the config file or as a command-line argument.

Usage:
    python test.py train.checkpoint=path/to/model.pt

Examples:
    # Evaluate a model using the checkpoint specified in the config
    python test.py

    # Evaluate a specific model checkpoint
    python test.py train.checkpoint=runs/run_11/best_validation_weights.pt

    # Evaluate a model with a different config
    python test.py --config-name=OCT2018 train.checkpoint=runs/run_11/best_validation_weights.pt
"""


@hydra.main(version_base=None, config_path="./configs", config_name="CAVRI-H5_cleaned")
def main(cfg: DictConfig) -> None:
    # Get model path from config
    model_path = cfg.train.checkpoint

    # If model_path is relative, make it absolute using the original working directory
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(hydra.utils.get_original_cwd(), model_path)
        cfg.train.checkpoint = model_path

    if model_path is None:
        print_msg("Error: Model path not provided. Please specify a model path using 'train.checkpoint=path/to/model.pt' when running the script.", error=True)
        return

    # Check if model path exists
    if not os.path.exists(model_path):
        print_msg(f"Error: Model path {model_path} does not exist.", error=True)
        return

    print_msg(f"Testing model from: {model_path}")

    # Set random seed for reproducibility
    if cfg.base.random_seed != -1:
        seed = cfg.base.random_seed
        set_random_seed(seed, cfg.base.cudnn_deterministic)

    # Generate model and load weights
    model = generate_model(cfg)

    # Generate dataset
    _, test_dataset, _ = generate_dataset(cfg)

    # Create estimator for evaluation metrics
    estimator = Estimator(cfg.train.metrics, cfg.data.num_classes, cfg.train.criterion)

    # Evaluate model on test dataset
    print_msg("Evaluating model on test dataset...")
    evaluate(cfg, model, test_dataset, estimator)

    # Generate Grad-CAM visualizations for sample images using the loaded model
    try:
        generate_and_save_grad_cam(cfg, model)
    except Exception as e:
        print_msg(f"Grad-CAM generation skipped due to error: {e}", warning=True)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


# ===== Grad-CAM utilities (adapted from paper_docs/grad_cam_visualization.py) =====

def _get_dataset_name_from_cfg(cfg) -> str:
    data_path = str(cfg.base.data_path)
    # Use the last path component as dataset name (e.g., CAVRI-H5_cleaned)
    return os.path.basename(os.path.normpath(data_path))


def get_sample_images(dataset_name: str):
    """
    Get sample images from each class in the examples directory by matching
    class-name substrings in filenames (in capital letters), instead of
    relying on exact filenames.

    Returns a dict mapping class name to example image path.
    """
    examples_dir = os.path.join("paper_docs", "examples", dataset_name)

    if not os.path.exists(examples_dir):
        print_msg(f"Examples directory not found: {examples_dir}", warning=True)
        return {}

    class_names = ["CNV", "DME", "DRUSEN", "NORMAL", "VMT"]

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    try:
        all_files = [os.path.join(examples_dir, f) for f in os.listdir(examples_dir) if f.lower().endswith(exts)]
    except Exception:
        all_files = []

    samples = {}
    for cname in class_names:
        match = next((p for p in all_files if cname in os.path.basename(p)), None)
        if match is not None:
            samples[cname] = match
    return samples


def preprocess_image(image_path, cfg):
    """Preprocess image using cfg.data.input_size and cfg.data mean/std.
    Returns (input_tensor[1, C, H, W], image_for_display[H, W, 3] in [0,1]).
    """
    image = Image.open(image_path).convert('RGB')

    mean = cfg.data.mean
    std = cfg.data.std
    if isinstance(mean, str) or not hasattr(mean, '__iter__'):
        mean = [0.485, 0.456, 0.406]
    else:
        try:
            mean = [float(x) for x in mean]
        except Exception:
            mean = [0.485, 0.456, 0.406]
    if isinstance(std, str) or not hasattr(std, '__iter__'):
        std = [0.229, 0.224, 0.225]
    else:
        try:
            std = [float(x) for x in std]
        except Exception:
            std = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    input_tensor = preprocess(image).unsqueeze(0)
    image_for_display = np.array(image.resize((cfg.data.input_size, cfg.data.input_size))) / 255.0
    return input_tensor, image_for_display


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, input_tensor, target_category=None):
        out = self.model(input_tensor)
        self.model.zero_grad()
        if target_category is None:
            target_category = torch.argmax(out, dim=1)
        one_hot = torch.zeros_like(out)
        one_hot[0, target_category] = 1
        out.backward(gradient=one_hot, retain_graph=True)
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        self._remove_hooks()
        return cam.squeeze().cpu().numpy()


def show_cam_on_image(img, mask, use_rgb=True, colormap=cv2.COLORMAP_JET, alpha=0.8):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = alpha * np.float32(img) + (1 - alpha) * heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_target_layer(model, model_name: str):
    try:
        if "resnet" in model_name:
            return model.layer4[-1]
        elif "vgg" in model_name:
            return model.features[-1]
        elif "inception" in model_name:
            return model.Mixed_7c.branch_pool
        elif "mobilenetv3" in model_name:
            return model.blocks[-1]
        else:
            # Fallback: last Conv2d
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
    except Exception:
        pass
    raise ValueError(f"Could not find a suitable target layer for model: {model_name}")


def generate_and_save_grad_cam(cfg, model):
    """Generate a combined Grad-CAM plot for sample images and save it in the
    model weights directory (the directory of cfg.train.checkpoint).
    """
    # Determine dataset and sample images
    dataset_name = _get_dataset_name_from_cfg(cfg)
    samples = get_sample_images(dataset_name)
    if not samples:
        print_msg("No sample images found for Grad-CAM. Skipping.", warning=True)
        return

    model_name = str(cfg.train.network)
    try:
        target_layer = get_target_layer(model, model_name)
    except Exception as e:
        print_msg(f"Grad-CAM not supported for model '{model_name}': {e}", warning=True)
        return

    preferred_order = ["CNV", "DME", "DRUSEN", "NORMAL", "VMT"]
    class_names = [c for c in preferred_order if c in samples]
    class_names += [c for c in samples.keys() if c not in preferred_order]
    if len(class_names) == 0:
        print_msg("No valid sample images for Grad-CAM.", warning=True)
        return

    n_cols = len(class_names)
    fig_width = max(4 * n_cols, 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 4))
    if n_cols == 1:
        axes = [axes]

    device = cfg.base.device if hasattr(cfg.base, 'device') else 'cuda'
    model_device = device

    for i, cname in enumerate(class_names):
        img_path = samples[cname]
        input_tensor, img_disp = preprocess_image(img_path, cfg)
        input_tensor = input_tensor.to(model_device)
        cam = GradCAM(model=model, target_layer=target_layer)
        grayscale_cam = cam(input_tensor=input_tensor)
        vis = show_cam_on_image(img_disp, grayscale_cam, alpha=0.8)
        axes[i].imshow(vis)
        axes[i].set_title(f"{cname}")
        axes[i].axis('off')

    plt.tight_layout()

    # Save to the directory of the checkpoint
    ckpt_path = str(cfg.train.checkpoint)
    out_dir = os.path.dirname(ckpt_path) if ckpt_path else str(cfg.base.save_path)
    os.makedirs(out_dir, exist_ok=True)
    dataset_token = dataset_name
    out_name = f"grad_cam_{model_name}_{dataset_token}_combined.png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path)
    plt.close()
    print_msg(f"Saved Grad-CAM visualization to: {out_path}")


if __name__ == '__main__':
    main()
