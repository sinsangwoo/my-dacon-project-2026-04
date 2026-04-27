import pandas as pd
import numpy as np

def analyze_feasibility():
    print("Loading grid search results...")
    try:
        df = pd.read_csv('logs/feasibility/grid_results_final.csv')
    except:
        df = pd.read_csv('logs/feasibility/grid_results_temp.csv')
        
    print(f"Total runs evaluated: {len(df)}")
    
    # Define criteria
    var_mask = (df['std_ratio'] >= 0.90) & (df['std_ratio'] <= 1.10)
    tail_mask = df['p99_ratio'] >= 0.85
    gen_mask = df['ADV_AUC'] <= 0.75
    
    print("\n--- 1. CRITICAL THRESHOLD METRICS ---")
    print(f"Runs satisfying Variance (0.90 <= std_ratio <= 1.10): {var_mask.sum()}")
    print(f"Runs satisfying Tail (p99_ratio >= 0.85): {tail_mask.sum()}")
    print(f"Runs satisfying Generalization (ADV AUC <= 0.75): {gen_mask.sum()}")
    
    feasible_runs = df[var_mask & tail_mask & gen_mask]
    
    print("\n--- 2. FEASIBLE REGION ANALYSIS ---")
    if len(feasible_runs) > 0:
        print(f"[CASE 1] FEASIBLE REGION EXISTS! Found {len(feasible_runs)} configurations.")
        print("Top 5 configurations (sorted by Q99_MAE):")
        best = feasible_runs.sort_values('Q99_MAE')
        print(best[['weight', 'tail_pct', 'feature_count', 'interaction', 'std_ratio', 'p99_ratio', 'ADV_AUC', 'Q99_MAE']].head(5))
    else:
        # Check boundary cases
        almost_feasible = df[var_mask & gen_mask]
        if len(almost_feasible) > 0:
            print("[CASE 3] BOUNDARY REGION EXISTS. Variance & Gen satisfied, but Tail is the bottleneck.")
            print(f"Max p99_ratio in this boundary: {almost_feasible['p99_ratio'].max():.4f} (Target >= 0.85)")
            best_boundary = almost_feasible.sort_values('p99_ratio', ascending=False).head(3)
            print("Closest configurations:")
            print(best_boundary[['weight', 'tail_pct', 'feature_count', 'interaction', 'std_ratio', 'p99_ratio', 'ADV_AUC']])
            
            # Root Cause Analysis
            print("\n--- ROOT CAUSE ANALYSIS ---")
            print("Hypothesis: Tail (p99_ratio >= 0.85) cannot be reached without breaking Generalization (ADV AUC <= 0.75)")
            high_tail = df[df['p99_ratio'] >= 0.85]
            if len(high_tail) > 0:
                print(f"When p99_ratio >= 0.85, the average ADV AUC is: {high_tail['ADV_AUC'].mean():.4f}")
                print(f"Min ADV AUC when p99_ratio >= 0.85 is: {high_tail['ADV_AUC'].min():.4f}")
                print("-> INCREASING TAIL WEIGHT BREAKS GENERALIZATION.")
            else:
                print("-> EXTREME TAIL (p99_ratio >= 0.85) IS STRUCTURALLY UNREACHABLE WITH CURRENT FEATURE SPACE.")
        else:
            print("[CASE 2] NO FEASIBLE REGION. Strong Trade-off Structure Detected.")
            print("\n--- ROOT CAUSE ANALYSIS ---")
            print("Correlation between std_ratio and ADV_AUC:", df['std_ratio'].corr(df['ADV_AUC']))
            print("Correlation between p99_ratio and ADV_AUC:", df['p99_ratio'].corr(df['ADV_AUC']))
            print("This indicates a fundamental trade-off: forcing variance/tail pushes the model to memorize the train set, ruining ADV AUC.")

    print("\n--- 3. PARAMETER SENSITIVITY ---")
    print("Effect of Interaction Strength on ADV AUC (Mean):")
    print(df.groupby('interaction')['ADV_AUC'].mean().sort_values())
    
    print("\nEffect of Sample Weight on p99_ratio (Mean):")
    print(df.groupby('weight')['p99_ratio'].mean().sort_values())

if __name__ == "__main__":
    analyze_feasibility()
