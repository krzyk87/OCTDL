import random
import numpy as np
import hydra
from omegaconf import DictConfig
from utils.func import *
from train import evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model

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


@hydra.main(version_base=None, config_path="./configs", config_name="CAVRI-H5")
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


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


if __name__ == '__main__':
    main()
