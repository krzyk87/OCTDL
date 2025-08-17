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
    datasets = ["OCT2017", "OCT2018", "Ravelli"]
    weight_files = {}
    
    for dataset in datasets:
        # Find all run directories for this model and dataset
        pattern = os.path.join("runs", model_name, f"run_*_{dataset}", "final_weights.pt")
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
    Get sample images from each class in the test set.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., OCT2017, OCT2018, Ravelli)
        
    Returns:
        dict: Dictionary mapping class names to image file paths
    """
    # Load config to get data path
    cfg = load_config(dataset_name)
    data_path = cfg.base.data_path
    
    # Get test directory
    test_dir = os.path.join(data_path, "test")
    
    # Check if the test directory exists
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        print(f"Please ensure the data path in the config file ({dataset_name}.yaml) is correct.")
        print(f"Current data path: {data_path}")
        print(f"Expected structure: {data_path}/test/[CNV,DME,DRUSEN,NORMAL]/images")
        return {}
    
    # Get one image from each class
    class_dirs = ["CNV", "DME", "DRUSEN", "NORMAL"]
    sample_images = {}
    
    for class_dir in class_dirs:
        class_path = os.path.join(test_dir, class_dir)
        if os.path.exists(class_path):
            image_files = glob.glob(os.path.join(class_path, "*.jpeg")) + glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
            if image_files:
                sample_images[class_dir] = image_files[0]
        else:
            print(f"Class directory not found: {class_path}")
    
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
    
    # Create preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std)
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
    
    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layer=target_layer)
    
    # Check if we have all four classes
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
    if not all(class_name in sample_images for class_name in class_names):
        missing = [c for c in class_names if c not in sample_images]
        print(f"Missing images for classes: {missing}. Cannot create combined visualization.")
        return
    
    # Create a single figure with 4 images in one row
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Process each class
    for i, class_name in enumerate(class_names):
        image_path = sample_images[class_name]
        
        # Preprocess image
        input_tensor, image_for_display = preprocess_image(image_path, cfg)

        # Move input tensor to the same device as model
        input_tensor = input_tensor.to(cfg.base.device)  # Add this line

        # Generate Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor)
        
        # Overlay Grad-CAM on image with more transparency
        visualization = show_cam_on_image(image_for_display, grayscale_cam, alpha=0.8)
        
        # Add to the figure
        axes[i].imshow(visualization)
        axes[i].set_title(f"{class_name}")
        axes[i].axis('off')
    
    # Set title for the entire figure
    fig.suptitle(f"{model_name} - {dataset_name} - Grad-CAM Visualizations", fontsize=16)
    
    # Save figure
    save_path = os.path.join("paper_docs", "grad_cam", f"{model_name}_{dataset_name}_combined.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved combined Grad-CAM visualization for {model_name} - {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for a model")
    parser.add_argument("model_name", type=str, help="Name of the model (e.g., resnet50, vgg16, inception_v3, mobilenetv3_small, mobilenetv3_large)")
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