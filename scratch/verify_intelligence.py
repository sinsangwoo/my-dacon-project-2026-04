import sys
import os
sys.path.append(os.getcwd())
from src.intelligence import ExperimentIntelligence
import json
import shutil

def simulate():
    # Setup fresh simulation
    sim_registry = 'metadata/test_registry.json'
    if os.path.exists(sim_registry):
        os.remove(sim_registry)
    
    intel = ExperimentIntelligence(registry_path=sim_registry)
    
    # 1. Run 1: Base Run
    print("Simulating Run 1...")
    os.makedirs('metadata/run_001', exist_ok=True)
    metrics_1 = {
        "mean_mae": 8.5,
        "worst_mae": 9.2,
        "extreme_mae": 15.0,
        "extreme_recall": 0.20,
        "variance_ratio": 0.8,
        "features": ["feat_a", "feat_b"]
    }
    intel.run_risk_focused_pipeline("run_001", metrics_1)
    
    # Check Registry
    with open(sim_registry, 'r') as f:
        reg = json.load(f)
        print(f"Current Best: {reg['current_best_run_id']}")
    
    # 2. Run 2: Worse and Low Recall
    print("\nSimulating Run 2 (Should be rejected due to low recall)...")
    os.makedirs('metadata/run_002', exist_ok=True)
    metrics_2 = {
        "mean_mae": 9.0,
        "worst_mae": 10.0,
        "extreme_mae": 20.0,
        "extreme_recall": 0.10, # Below 0.15 threshold
        "variance_ratio": 0.85,
        "features": ["feat_a", "feat_c"]
    }
    intel.run_risk_focused_pipeline("run_002", metrics_2)
    
    # 3. Run 3: Better Mean but Worse Extreme (Pareto test)
    print("\nSimulating Run 3 (Pareto check)...")
    os.makedirs('metadata/run_003', exist_ok=True)
    metrics_3 = {
        "mean_mae": 8.0, # Better than run_001
        "worst_mae": 9.5, # Worse than run_001
        "extreme_mae": 18.0, # Worse than run_001
        "extreme_recall": 0.18,
        "variance_ratio": 0.9,
        "features": ["feat_a", "feat_d"]
    }
    intel.run_risk_focused_pipeline("run_003", metrics_3)
    
    # 4. Check Pareto Runs
    with open('metadata/run_003/pareto_runs.json', 'r') as f:
        pareto = json.load(f)
        print(f"Pareto count: {len(pareto)}")
        for p in pareto:
            print(f" - {p['run_id']} (Mean: {p['mean_mae']}, Extreme: {p['extreme_mae']})")

    # 5. Delta Analysis Check
    with open('metadata/run_003/run_comparison.json', 'r') as f:
        comparison = json.load(f)
        print("\nDelta Analysis (Run 3 vs Best):")
        print(f" Mean MAE Delta: {comparison['deltas']['mean_mae']:.4f}")
        print(f" Added features: {comparison['feature_changes']['added']}")

    # 6. Knowledge Log Check
    if os.path.exists('metadata/knowledge_log.json'):
        with open('metadata/knowledge_log.json', 'r') as f:
            log = json.load(f)
            print("\nKnowledge Log:")
            for k in log:
                print(f" - {k['action']} -> {k['effect']} (Confidence: {k['confidence']})")

if __name__ == "__main__":
    simulate()
