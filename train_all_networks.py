#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from datetime import datetime

# Keep this list in sync with test_all_networks.py
NETWORKS = [
    "vgg16",
    "inception_v3",
    "resnet50",
    "mobilenetv3_large_100",
    "mobilenetv3_small_100",
    "vit_base_patch16_224",
    "tf_efficientnetv2_s",
    "resnet18",
]

DEFAULT_CONFIG_NAME = "CAVRI-H5_cleaned.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Run main.py training for multiple networks.")
    p.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Config YAML file name in ./configs to use (default: OCTDL.yaml)",
    )
    p.add_argument(
        "--networks",
        nargs="*",
        default=NETWORKS,
        help="Optional subset of networks to train (default: all)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(project_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Prepare Hydra-compatible config name (without .yaml extension)
    cfg_name_for_hydra = os.path.splitext(os.path.basename(args.config_name))[0]
    configs_dir = os.path.join(project_root, "configs")
    config_path = os.path.join(configs_dir, args.config_name)
    if not os.path.exists(config_path):
        print(f"[run_all_networks] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for net in args.networks:
        out_path = os.path.join(runs_dir, f"output_{net}.txt")
        print(
            f"[run_all_networks] Starting training for network={net} with config={cfg_name_for_hydra}. Output -> {out_path}"
        )
        with open(out_path, "w", buffering=1) as f:
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Config: {args.config_name}\n")
            f.write(f"Network: {net}\n\n")
            f.flush()
            # Use Hydra override syntax to set the network and selected config for this run
            cmd = [
                "python",
                os.path.join(project_root, "main.py"),
                "--config-name",
                cfg_name_for_hydra,
                f"train.network={net}",
            ]
            # Stream both stdout and stderr to the same file
            try:
                subprocess.run(cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                f.write(f"\n[run_all_networks] Run failed for {net} with return code {e.returncode}.\n")
                print(f"[run_all_networks] Run failed for {net} (see {out_path}).")
            else:
                f.write("\n[run_all_networks] Run finished successfully.\n")
                print(f"[run_all_networks] Completed network={net}.")


if __name__ == "__main__":
    main()
