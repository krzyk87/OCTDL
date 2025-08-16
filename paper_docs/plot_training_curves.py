"""
Plot training and validation curves (accuracy, loss) for a selected model architecture.

This script finds tensorboard log files for a given model across all datasets (OCT2017, OCT2018, Ravelli),
loads the log data, and plots training and validation metrics if available.

Usage:
    python plot_training_curves.py MODEL_NAME

Where MODEL_NAME can be one of: resnet50, vgg16, inception_v3, mobilenetv3_small, mobilenetv3_large

Example:
    python plot_training_curves.py resnet50

The script will generate a plot showing training and validation metrics for each dataset
and save it as MODEL_NAME_metrics.png in the current directory.
"""

import os
import argparse
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def find_log_files(model_name):
    """
    Find tensorboard log files for a given model across all datasets.
    
    Args:
        model_name (str): Name of the model (e.g., resnet50, vgg16, etc.)
        
    Returns:
        dict: Dictionary mapping dataset names to log file paths
    """
    datasets = ["OCT2017", "OCT2018", "Ravelli"]
    log_files = {}
    
    for dataset in datasets:
        # Find all run directories for this model and dataset
        pattern = os.path.join("runs", model_name, f"run_*_{dataset}", "log", "*")
        files = glob.glob(pattern)
        
        if files:
            # Use the most recent log file if there are multiple
            log_files[dataset] = files[0]
    
    return log_files

def load_tensorboard_data(log_file):
    """
    Load data from a tensorboard log file.
    
    Args:
        log_file (str): Path to the tensorboard log file
        
    Returns:
        dict: Dictionary containing the extracted metrics
    """
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    
    # Available tags
    tags = ea.Tags()['scalars']
    
    # Extract metrics of interest
    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        metrics[tag] = (steps, values)
    
    return metrics

def plot_metrics(model_name, datasets_metrics):
    """
    Plot training and validation metrics for a given model.
    
    Args:
        model_name (str): Name of the model
        datasets_metrics (dict): Dictionary mapping dataset names to metrics
    """
    # Create a figure with 2 rows (acc and loss) and 1 column per dataset
    n_datasets = len(datasets_metrics)
    if n_datasets == 0:
        print(f"No data found for model: {model_name}")
        return
    
    fig, axes = plt.subplots(2, n_datasets, figsize=(5*n_datasets, 10), squeeze=False)
    fig.suptitle(f"Training and Validation Metrics for {model_name}", fontsize=16)
    
    # Find min/max values across all datasets for consistent y-axis ranges
    min_acc, max_loss = 1.0, 0.0
    for dataset, metrics in datasets_metrics.items():
        for metric_name, (steps, values) in metrics.items():
            if "acc" in metric_name.lower():
                min_acc = min(min_acc, min(values) if values else 1.0)
            elif "loss" in metric_name.lower():
                max_loss = max(max_loss, max(values) if values else 0.0)
    
    # Ensure min_acc is not too close to 1.0 for better visualization
    min_acc = max(0.0, min_acc - 0.05)
    # Add a small margin to max_loss for better visualization
    max_loss = max_loss * 1.05
    
    for i, (dataset, metrics) in enumerate(datasets_metrics.items()):
        # Plot accuracy
        ax_acc = axes[0, i]
        ax_acc.set_title(f"{dataset} - Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(min_acc, 1.1)  # Set consistent y-axis range for accuracy
        ax_acc.grid(True, alpha=0.3)  # Add grid to accuracy plot

        # Plot loss
        ax_loss = axes[1, i]
        ax_loss.set_title(f"{dataset} - Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_ylim(0, max_loss)  # Set consistent y-axis range for loss
        ax_loss.grid(True, alpha=0.3)  # Add grid to loss plot

        # Plot training and validation metrics if available
        for metric_name, (steps, values) in metrics.items():
            if "acc" in metric_name.lower() and "val" not in metric_name.lower():
                ax_acc.plot(steps, values, label="Training Acc")
            elif "acc" in metric_name.lower() and "val" in metric_name.lower():
                ax_acc.plot(steps, values, label="Validation Acc")
            elif "loss" in metric_name.lower() and "val" not in metric_name.lower():
                ax_loss.plot(steps, values, label="Training Loss")
            elif "loss" in metric_name.lower() and "val" in metric_name.lower():
                ax_loss.plot(steps, values, label="Validation Loss")
        
        ax_acc.legend()
        ax_loss.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(f"{model_name}_metrics.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation curves for a model")
    parser.add_argument("model_name", type=str, help="Name of the model (e.g., resnet50, vgg16, inception_v3, mobilenetv3_small, mobilenetv3_large)")
    args = parser.parse_args()
    
    model_name = args.model_name
    
    # Find log files for the model
    log_files = find_log_files(model_name)
    
    if not log_files:
        print(f"No log files found for model: {model_name}")
        return
    
    # Load data and plot metrics
    datasets_metrics = {}
    for dataset, log_file in log_files.items():
        print(f"Loading data for {dataset} from {log_file}")
        metrics = load_tensorboard_data(log_file)
        datasets_metrics[dataset] = metrics
    
    # Plot the metrics
    plot_metrics(model_name, datasets_metrics)

if __name__ == "__main__":
    main()