#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime

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


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(project_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for net in NETWORKS:
        out_path = os.path.join(runs_dir, f"output_{net}.txt")
        print(f"[run_all_networks] Starting training for network={net}. Output -> {out_path}")
        with open(out_path, "w", buffering=1) as f:
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Network: {net}\n\n")
            f.flush()
            # Use Hydra override syntax to set the network for this run
            cmd = [
                "python",
                os.path.join(project_root, "main.py"),
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
