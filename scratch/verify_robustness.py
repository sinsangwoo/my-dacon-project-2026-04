import sys
import os
sys.path.append(os.getcwd())
from src.intelligence import ExperimentIntelligence
import json

def simulate_robust():
    sim_registry = 'metadata/robust_registry.json'
    if os.path.exists(sim_registry):
        os.remove(sim_registry)
    
    intel = ExperimentIntelligence(registry_path=sim_registry)
    
    # 1. Run 1: Solid Performer
    print("Simulating Run 1 (Solid Performer)...")
    os.makedirs('metadata/run_001', exist_ok=True)
    metrics_1 = {
        "mean_mae": 8.5,
        "worst_mae": 9.2,
        "extreme_mae": 15.0,
        "extreme_f1": 0.25, # Above 0.20 base
        "variance_ratio": 0.8,
        "extreme_quantile": 0.9,
        "extreme_threshold_val": 30.5,
        "features": ["feat_a"]
    }
    intel.run_risk_focused_pipeline("run_001", metrics_1)
    
    # 2. Run 2: Recall Hacker
    print("\nSimulating Run 2 (Recall Hacker - Should be rejected by F1 floor)...")
    os.makedirs('metadata/run_002', exist_ok=True)
    metrics_2 = {
        "mean_mae": 8.4,
        "worst_mae": 9.0,
        "extreme_mae": 16.0,
        "extreme_f1": 0.12, # Low F1!
        "variance_ratio": 0.81,
        "extreme_quantile": 0.9,
        "extreme_threshold_val": 30.5,
        "features": ["feat_b"]
    }
    intel.run_risk_focused_pipeline("run_002", metrics_2)
    
    # 3. Stability Test
    print("\nInjecting stable history to trigger STRICT mode...")
    with open(sim_registry, 'r') as f:
        reg = json.load(f)
    
    for i in range(10): # Last 10 runs
        reg["runs"].append({
            "run_id": f"stab_{i}",
            "status": "PASSED",
            "mean_mae": 8.6,
            "worst_mae": 9.3,
            "extreme_mae": 15.5,
            "extreme_f1": 0.28,
            "variance_ratio": 0.8 # Constant 0.8 -> std = 0
        })
    with open(sim_registry, 'w') as f:
        json.dump(reg, f)
        
    intel = ExperimentIntelligence(registry_path=sim_registry) # Reload
    
    # 4. Run 3: Moderate but F1 too low for STRICT mode
    print("\nSimulating Run 3 (Moderate F1 in STRICT mode - Should be rejected)...")
    os.makedirs('metadata/run_003', exist_ok=True)
    metrics_3 = {
        "mean_mae": 8.3,
        "worst_mae": 8.9,
        "extreme_mae": 14.5,
        "extreme_f1": 0.26, # 0.26 is > 0.20 baseline, but STRICT floor is 0.30
        "variance_ratio": 0.8,
        "extreme_quantile": 0.9,
        "extreme_threshold_val": 30.5,
        "features": ["feat_c"]
    }
    intel.run_risk_focused_pipeline("run_003", metrics_3)

    # 5. Dynamic Pareto Cut Test
    print("\nDynamic Pareto Cut Test...")
    reg["runs"].append({
        "run_id": "outlier",
        "status": "PASSED",
        "mean_mae": 8.4, 
        "worst_mae": 9.1,
        "extreme_mae": 50.0, # GARBAGE extreme
        "extreme_f1": 0.35, 
        "variance_ratio": 0.8,
        "extreme_quantile": 0.9,
        "extreme_threshold_val": 30.5,
        "features": ["feat_trash"]
    })
    with open(sim_registry, 'w') as f:
        json.dump(reg, f)
    
    intel = ExperimentIntelligence(registry_path=sim_registry)
    metrics_4 = {
        "mean_mae": 8.45,
        "worst_mae": 9.15,
        "extreme_mae": 15.1,
        "extreme_f1": 0.31,
        "variance_ratio": 0.8,
        "extreme_quantile": 0.9,
        "extreme_threshold_val": 30.5,
        "features": ["feat_d"]
    }
    intel.run_risk_focused_pipeline("run_004", metrics_4)
    
    with open('metadata/run_004/pareto_runs.json', 'r') as f:
        pareto = json.load(f)
        print(f"\nFinal Pareto Runs count (should have pruned outlier): {len(pareto)}")
        ids = [p['run_id'] for p in pareto]
        print(f"Pareto IDs: {ids}")
        if "outlier" not in ids:
            print("✓ SUCCESS: Outlier pruned by dynamic cut.")
        else:
            print("✗ FAILURE: Outlier still in Pareto frontier.")

if __name__ == "__main__":
    simulate_robust()
