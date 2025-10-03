"""
Generate Grad-CAM visualizations for trained models across different datasets.

This script finds trained model weights for a given model across all datasets (OCT2017, OCT2018, Ravelli),
loads the models, and generates Grad-CAM visualizations for sample images from each class.

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that uses the gradients of any target concept
flowing into the final convolutional layer to produce a coarse localization map highlighting important regions
in the image for predicting the concept.

Prerequisites:
    - Trained model weights in runs/MODEL_NAME/run_NUMBER_DATASET/final_weights.pt
    - Test images in ../DATA/DATASET/test directory with class subdirectories (CNV, DME, DRUSEN, NORMAL)
    - Configuration files in configs folder (OCT2017.yaml, OCT2018.yaml, OCT2023_Ravelli.yaml)

Usage:
    python grad_cam_visualization.py MODEL_NAME

Where MODEL_NAME can be one of: resnet50, vgg16, inception_v3, mobilenetv3_small, mobilenetv3_large

Example:
    python grad_cam_visualization.py resnet50

Output:
    The script will generate a combined Grad-CAM visualization for each dataset,
    showing one image from each of the four classes (CNV, DME, DRUSEN, NORMAL) in a single row.
    The visualizations are saved in the paper_docs/grad_cam directory with filenames in the format:
    MODEL_NAME_DATASET_combined.png (e.g., resnet50_OCT2017_combined.png)

Each output image contains:
    - Four Grad-CAM visualizations in one row, one for each class
    - Each visualization shows the original image with a semi-transparent Grad-CAM overlay

Note:
    If the script cannot find the test images, check that the data paths in the config files
    (configs/OCT2017.yaml, configs/OCT2018.yaml, configs/OCT2023_Ravelli.yaml) are correct.
    The expected data structure is:
    ../DATA/DATASET/test/CLASS/images
"""

import os
import sys
import glob
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.builder import generate_model
from omegaconf import OmegaConf


def find_model_weights(model_name):
    """
    Find trained model weights for a given model across all datasets.
    
    Args:
        model_name (str): Name of the model (e.g., resnet50, vgg16, etc.)
        
    Returns:
        dict: Dictionary mapping dataset names to model weight file paths
    """
    datasets = ["CAVRI-H5_cleaned"] # ["OCT2017", "OCT2018", "Ravelli"]
    weight_files = {}
    
    for dataset in datasets:
        # Find all run directories for this model and dataset
        pattern = os.path.join("runs", dataset, f"run_*_{model_name}", "final_weights.pt")
        files = glob.glob(pattern)
        
        if files:
            # Use the most recent weight file if there are multiple
            weight_files[dataset] = files[0]
    
    return weight_files


def load_config(dataset_name):
    """
    Load configuration for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., OCT2017, OCT2018, Ravelli)
        
    Returns:
        dict: Configuration dictionary
    """
    if dataset_name == "Ravelli":
        config_file = os.path.join("configs", "OCT2023_Ravelli.yaml")
    else:
        config_file = os.path.join("configs", f"{dataset_name}.yaml")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf for compatibility with the model loading code
    cfg = OmegaConf.create(config)
    return cfg


def get_sample_images(dataset_name):
    """
    Get sample images from each class in the examples directory by matching
    class-name substrings in filenames (in capital letters), instead of
    relying on exact filenames.

    Args:
        dataset_name (str): Name of the dataset (used to select examples subfolder)

    Returns:
        dict: Dictionary mapping class names to image file paths (one file per class if found)
    """
    examples_dir = os.path.join("paper_docs", "examples", dataset_name)

    # Check if the examples directory exists
    if not os.path.exists(examples_dir):
        print(f"Examples directory not found: {examples_dir}")
        print("Please ensure paper_docs/examples/<DATASET>/ contains example images per class.")
        return {}

    # Classes to look for (uppercase substrings expected in filenames)
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL", "VMT"]

    # Collect all candidate image files
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    try:
        all_files = [os.path.join(examples_dir, f) for f in os.listdir(examples_dir) if f.lower().endswith(exts)]
    except Exception:
        all_files = []

    # Build mapping by substring matching (case-sensitive for the class token)
    sample_images = {}
    for cname in class_names:
        match = next((p for p in all_files if cname in os.path.basename(p)), None)
        if match is not None:
            sample_images[cname] = match
        else:
            # Informative message; continue without failing hard
            print(f"No example image found for class '{cname}' in {examples_dir} (looked for filename containing '{cname}').")

    return sample_images


def preprocess_image(image_path, cfg):
    """
    Preprocess an image for model input.
    
    Args:
        image_path (str): Path to the image file
        cfg (dict): Configuration dictionary
        
    Returns:
        tuple: (preprocessed_image_tensor, normalized_image_for_display)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # ANSI color codes
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    # Ensure mean and std are numeric lists, not strings
    mean = cfg.data.mean
    std = cfg.data.std

    # Handle case where mean/std might be strings or other non-numeric types
    if isinstance(mean, str) or not hasattr(mean, '__iter__'):
        # Fallback to ImageNet defaults for RGB images
        mean = [0.485, 0.456, 0.406]
        print(f"{YELLOW}Warning: Using default ImageNet mean values due to invalid config: {cfg.data.mean}{RESET}")
    else:
        # Ensure it's a list of floats
        try:
            mean = [float(x) for x in mean]
        except (ValueError, TypeError):
            mean = [0.485, 0.456, 0.406]
            print(f"{YELLOW}Warning: Using default ImageNet mean values due to conversion error: {cfg.data.mean}{RESET}")

    if isinstance(std, str) or not hasattr(std, '__iter__'):
        # Fallback to ImageNet defaults for RGB images
        std = [0.229, 0.224, 0.225]
        print(f"{YELLOW}Warning: Using default ImageNet std values due to invalid config: {cfg.data.std}{RESET}")
    else:
        # Ensure it's a list of floats
        try:
            std = [float(x) for x in std]
        except (ValueError, TypeError):
            std = [0.229, 0.224, 0.225]
            print(f"{YELLOW}Warning: Using default ImageNet std values due to conversion error: {cfg.data.std}{RESET}")
    
    # Create preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Preprocess image for model input
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Create normalized image for display
    image_for_display = np.array(image.resize((cfg.data.input_size, cfg.data.input_size))) / 255.0
    
    return input_tensor, image_for_display


class GradCAM:
    """
    Implements Grad-CAM for visualizing CNN decisions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
        
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        # Register the hooks
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def __call__(self, input_tensor, target_category=None):
        # Forward pass
        model_output = self.model(input_tensor)
        
        # Clear gradients
        self.model.zero_grad()
        
        # Target for backprop
        if target_category is None:
            target_category = torch.argmax(model_output, dim=1)
            
        # One-hot encoding
        one_hot = torch.zeros_like(model_output)
        one_hot[0, target_category] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Remove hooks
        self.remove_hooks()
        
        return cam.squeeze().cpu().numpy()


def show_cam_on_image(img, mask, use_rgb=True, colormap=cv2.COLORMAP_JET, alpha=0.6):
    """
    Overlay the CAM mask on the image with adjustable transparency.
    
    Args:
        img (numpy.ndarray): Input image
        mask (numpy.ndarray): CAM mask
        use_rgb (bool): Whether to use RGB or BGR
        colormap: OpenCV colormap
        alpha (float): Transparency factor (0.0 to 1.0) - higher means more transparent overlay
        
    Returns:
        numpy.ndarray: Visualization
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    heatmap = np.float32(heatmap) / 255
    # Blend with transparency: alpha * img + (1-alpha) * heatmap
    cam = alpha * np.float32(img) + (1-alpha) * heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_target_layer(model, model_name):
    """
    Get the target layer for Grad-CAM visualization.
    
    Args:
        model: PyTorch model
        model_name (str): Name of the model
        
    Returns:
        layer: Target layer for Grad-CAM
    """
    # Different models have different layer names for the final convolutional layer
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "vgg" in model_name:
        return model.features[-1]
    elif "inception" in model_name:
        return model.Mixed_7c.branch_pool
    elif "mobilenetv3" in model_name:
        return model.blocks[-1]
    else:
        # Default to the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
    
    raise ValueError(f"Could not find a suitable target layer for model: {model_name}")


def generate_grad_cam(model_name, dataset_name, weight_file, sample_images, cfg):
    """
    Generate Grad-CAM visualizations for a model and dataset.
    
    Args:
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        weight_file (str): Path to the model weights file
        sample_images (dict): Dictionary mapping class names to image file paths
        cfg (dict): Configuration dictionary
        
    Returns:
        None
    """
    # Set up model
    cfg.train.network = model_name
    cfg.train.checkpoint = weight_file
    model = generate_model(cfg)
    model.eval()
    
    # Get target layer for Grad-CAM
    target_layer = get_target_layer(model, model_name)
    
    # Determine which classes we have based on provided sample_images
    preferred_order = ["CNV", "DME", "DRUSEN", "NORMAL", "VMT"]
    class_names = [c for c in preferred_order if c in sample_images]
    # Include any additional keys not in preferred_order (preserve dict order)
    class_names += [c for c in sample_images.keys() if c not in preferred_order]

    if len(class_names) == 0:
        print("No valid sample images provided. Cannot create combined visualization.")
        return

    # Create a single figure with as many images as we have samples
    n_cols = len(class_names)
    fig_width = max(4 * n_cols, 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 4))
    if n_cols == 1:
        axes = [axes]
    
    # Process each class
    for i, class_name in enumerate(class_names):
        image_path = sample_images[class_name]
        
        # Preprocess image
        input_tensor, image_for_display = preprocess_image(image_path, cfg)

        # Move input tensor to the same device as model
        input_tensor = input_tensor.to(cfg.base.device)

        # Initialize a new Grad-CAM instance for each image
        cam = GradCAM(model=model, target_layer=target_layer)
        
        # Generate Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor)
        
        # Overlay Grad-CAM on image with more transparency
        visualization = show_cam_on_image(image_for_display, grayscale_cam, alpha=0.8)
        
        # Add to the figure
        axes[i].imshow(visualization)
        axes[i].set_title(f"{class_name}")
        axes[i].axis('off')
    
    # Set title for the entire figure
    # fig.suptitle(f"{model_name} - {dataset_name} - Grad-CAM Visualizations", fontsize=16)
    
    # Save figure
    save_path = os.path.join("paper_docs", "grad_cam", f"{model_name}_{dataset_name}_combined.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved combined Grad-CAM visualization for {model_name} - {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for a model")
    parser.add_argument("--model_name", type=str, default="vgg16", help="Name of the model (e.g., resnet50, vgg16, inception_v3, mobilenetv3_small, mobilenetv3_large)")
    args = parser.parse_args()
    
    model_name = args.model_name
    
    # Find model weights
    weight_files = find_model_weights(model_name)
    
    if not weight_files:
        print(f"No model weights found for model: {model_name}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join("paper_docs", "grad_cam"), exist_ok=True)
    
    # Process each dataset
    for dataset_name, weight_file in weight_files.items():
        print(f"Processing {dataset_name} dataset with weights from {weight_file}")
        
        # Load config
        cfg = load_config(dataset_name)
        
        # Get sample images
        sample_images = get_sample_images(dataset_name)
        
        if not sample_images:
            print(f"No sample images found for dataset: {dataset_name}")
            continue
        
        # Generate Grad-CAM visualizations
        generate_grad_cam(model_name, dataset_name, weight_file, sample_images, cfg)


if __name__ == "__main__":
    main()