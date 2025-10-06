#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from datetime import datetime
from typing import List, Optional

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None  # We'll fallback to simple string substitution if not available

# Keep this in sync with run_all_networks.py
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
    p = argparse.ArgumentParser(description="Run test.py for multiple networks and selected weights.")
    p.add_argument(
        "--weights",
        choices=["best", "final"],
        default="best",
        help="Which weights file to evaluate: best -> best_validation_weights.pt, final -> final_weights.pt",
    )
    p.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Config YAML file name in ./configs to use (default: CAVRI-H5_cleaned.yaml)",
    )
    p.add_argument(
        "--networks",
        nargs="*",
        default=NETWORKS,
        help="Optional subset of networks to evaluate (default: all)",
    )
    return p.parse_args()


def load_save_path_template(config_path: str) -> Optional[str]:
    """Return the raw template string for base.save_path from the YAML file.
    We intentionally avoid resolving OmegaConf interpolations here to preserve
    placeholders like ${train.network} so they can be substituted per network.
    """
    # Parse the YAML file textually to avoid resolving ${...} interpolations.
    try:
        with open(config_path, "r") as f:
            for line in f:
                s = line.strip()
                if s.startswith("save_path:"):
                    # after colon
                    val = s.split(":", 1)[1].strip()
                    # strip quotes if any
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    return val
    except Exception:
        return None
    return None


def resolve_save_path(template: str, network: str) -> str:
    # Replace common interpolation patterns
    # Support both ${train.network} and ${ train.network } patterns
    return (
        template.replace("${train.network}", network)
        .replace("${ train.network }", network)
    )


def find_checkpoint(project_root: str, network: str, weights_kind: str, save_path_template: str) -> Optional[str]:
    """Return absolute path to the checkpoint file if found, else None.
    weights_kind: 'best' or 'final'
    """
    runs_root = os.path.join(project_root, "runs")
    # 1) First try the direct path using template
    candidate_dir_rel = resolve_save_path(save_path_template, network)
    # Ensure relative paths are based on project root
    candidate_dir = candidate_dir_rel
    if not os.path.isabs(candidate_dir):
        candidate_dir = os.path.normpath(os.path.join(project_root, candidate_dir_rel))

    filename = "best_validation_weights.pt" if weights_kind == "best" else "final_weights.pt"
    direct_ckpt = os.path.join(candidate_dir, filename)
    if os.path.exists(direct_ckpt):
        return direct_ckpt

    # 2) Fallback: search under runs/ for directories that look like the template root
    # Handle both hyphen and underscore dataset folder naming differences
    # e.g., runs/CAVRI-H5_cleaned/ or runs/CAVRI_H5_cleaned/
    possible_bases: List[str] = []
    # If template starts with ./runs/<something>/, reuse that something
    try:
        after_runs = candidate_dir_rel.split("runs" + os.sep, 1)[1]
        first_component = after_runs.split(os.sep, 1)[0]
        if first_component:
            possible_bases.append(first_component)
    except Exception:
        pass

    # Add known variants for robustness
    possible_bases = list(dict.fromkeys(possible_bases + [
        "CAVRI-H5_cleaned",
        "CAVRI_H5_cleaned",
    ]))

    candidates: List[str] = []
    for base in possible_bases:
        base_dir = os.path.join(runs_root, base)
        if not os.path.isdir(base_dir):
            continue
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            # Prefer dirs that include the network in their name (for new template scheme)
            if network in name:
                ck = os.path.join(full, filename)
                if os.path.exists(ck):
                    candidates.append(ck)
            else:
                # Also consider generic dirs (e.g., run_batch64_A, run_batch64_A_1, ...)
                # We'll pick the most recent one that has the checkpoint file
                ck = os.path.join(full, filename)
                if os.path.exists(ck):
                    candidates.append(ck)

    if candidates:
        # Pick the most recently modified checkpoint
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    return None


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(project_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    configs_dir = os.path.join(project_root, "configs")
    config_path = os.path.join(configs_dir, args.config_name)
    if not os.path.exists(config_path):
        print(f"[test_all_networks] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    save_path_template = load_save_path_template(config_path)
    if not save_path_template:
        print("[test_all_networks] Could not read base.save_path from config. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Prepare Hydra-compatible config name (without .yaml extension)
    cfg_name_for_hydra = os.path.splitext(os.path.basename(args.config_name))[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_filename = "best_validation_weights.pt" if args.weights == "best" else "final_weights.pt"

    for net in args.networks:
        out_path = os.path.join(runs_dir, f"test_output_{net}_{args.weights}.txt")
        print(f"[test_all_networks] Starting test for network={net} using {weights_filename} with config={cfg_name_for_hydra}. Output -> {out_path}")

        ckpt_path = find_checkpoint(project_root, net, args.weights, save_path_template)
        with open(out_path, "w", buffering=1) as f:
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Config: {args.config_name}\n")
            f.write(f"Network: {net}\n")
            f.write(f"Weights: {weights_filename}\n\n")
            if not ckpt_path:
                msg = f"[test_all_networks] Could not locate checkpoint for {net} ({weights_filename}). Skipping.\n"
                f.write(msg)
                print(msg.strip())
                continue

            f.write(f"Checkpoint: {ckpt_path}\n\n")
            f.flush()

            cmd = [
                "python",
                os.path.join(project_root, "test.py"),
                "--config-name", cfg_name_for_hydra,
                f"train.network={net}",
                f"train.checkpoint={ckpt_path}",
            ]
            try:
                subprocess.run(cmd, cwd=project_root, stdout=f, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                f.write(f"\n[test_all_networks] Test failed for {net} with return code {e.returncode}.\n")
                print(f"[test_all_networks] Test failed for {net} (see {out_path}).")
            else:
                f.write("\n[test_all_networks] Test finished successfully.\n")
                print(f"[test_all_networks] Completed network={net} ({args.weights}).")


if __name__ == "__main__":
    main()
