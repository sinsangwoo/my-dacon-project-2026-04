import os
import numpy as np
import pandas as pd
import json
import subprocess
from src.config import Config

# Setup run_id (using existing successful run for components)
RUN_ID = "run_20260421_125821"
os.environ['RUN_ID'] = RUN_ID

# Define experiment configurations
experiments = {
    "A": {
        "STAGE1": False, "STAGE2": False, "STAGE3": False, "STAGE4": False, "STAGE5": False
    },
    "B": {
        "STAGE1": True, "STAGE2": False, "STAGE3": False, "STAGE4": False, "STAGE5": False
    },
    "C": {
        "STAGE1": True, "STAGE2": True, "STAGE3": False, "STAGE4": False, "STAGE5": False
    },
    "D": {
        "STAGE1": True, "STAGE2": True, "STAGE3": True, "STAGE4": False, "STAGE5": False
    },
    "E": {
        "STAGE1": True, "STAGE2": True, "STAGE3": True, "STAGE4": True, "STAGE5": True
    }
}

results = []

def run_pipeline(version, cfg):
    print(f"\n--- [ABLATON] Version {version} ---")
    
    # Update Config dynamically
    # Note: We need to pass these as environment variables or modify src/config.py 
    # since main.py imports it. For simplicity, we'll use env vars and modify main.py 
    # to read them if present.
    
    env = os.environ.copy()
    env["ENABLE_STAGE1_SHIELD"] = str(cfg["STAGE1"])
    env["ENABLE_STAGE2_RATIO"] = str(cfg["STAGE2"])
    env["ENABLE_STAGE3_SOFTENING"] = str(cfg["STAGE3"])
    env["ENABLE_STAGE4_DEFENSE"] = str(cfg["STAGE4"])
    env["ENABLE_STAGE5_PRUNING"] = str(cfg["STAGE5"])
    
    # Rerun Phase 2 (Build Raw) and Phase 4 (Build Full) to apply Stage 1, 2, 5 feature changes
    # Then Phase 7 (Inference) and 8 (Submission)
    phases = ["2_build_raw", "4_build_full", "7_inference", "8_submission"]
    
    for phase in phases:
        cmd = ["python", "main.py", "--phase", phase, "--mode", "full"]
        subprocess.run(cmd, env=env, check=True)
    
    # Collect Metrics
    pred_path = f"outputs/{RUN_ID}/predictions/final_submission.npy"
    preds = np.load(pred_path)
    
    # OOF MAE from metrics.json (Phase 7 logs it)
    # Actually, MAE is logged in logs/{RUN_ID}/current_mae.txt
    mae = 0.0
    mae_file = f"logs/{RUN_ID}/current_mae.txt"
    if os.path.exists(mae_file):
        with open(mae_file, "r") as f:
            mae = float(f.read().strip())
            
    stats = {
        "version": version,
        "MAE": mae,
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "p99": float(np.quantile(preds, 0.99))
    }
    return stats

# Modify main.py to read ablation flags from environment variables
def patch_main():
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    patch = """
    # [ABLATION_OVERRIDE]
    if os.getenv("ENABLE_STAGE1_SHIELD"): Config.ENABLE_STAGE1_SHIELD = os.getenv("ENABLE_STAGE1_SHIELD") == "True"
    if os.getenv("ENABLE_STAGE2_RATIO"): Config.ENABLE_STAGE2_RATIO = os.getenv("ENABLE_STAGE2_RATIO") == "True"
    if os.getenv("ENABLE_STAGE3_SOFTENING"): Config.ENABLE_STAGE3_SOFTENING = os.getenv("ENABLE_STAGE3_SOFTENING") == "True"
    if os.getenv("ENABLE_STAGE4_DEFENSE"): Config.ENABLE_STAGE4_DEFENSE = os.getenv("ENABLE_STAGE4_DEFENSE") == "True"
    if os.getenv("ENABLE_STAGE5_PRUNING"): Config.ENABLE_STAGE5_PRUNING = os.getenv("ENABLE_STAGE5_PRUNING") == "True"
"""
    if "[ABLATION_OVERRIDE]" not in content:
        content = content.replace("Config.recompute_paths()", "Config.recompute_paths()" + patch)
        with open("main.py", "w", encoding="utf-8") as f:
            f.write(content)

patch_main()

for version, cfg in experiments.items():
    res = run_pipeline(version, cfg)
    results.append(res)

# Output Table
print("\n" + "="*50)
print(f"| version | MAE | mean | std | p99 |")
print("-" * 50)
for r in results:
    print(f"| {r['version']} | {r['MAE']:.4f} | {r['mean']:.4f} | {r['std']:.4f} | {r['p99']:.4f} |")
print("="*50)
