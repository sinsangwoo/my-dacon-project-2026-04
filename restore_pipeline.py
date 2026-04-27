"""v16.0 Pipeline Restoration Tool.

Restores a broken run by executing all phases sequentially.
"""
import subprocess
import sys
import os

# Import phases from main to ensure SSOT
sys.path.append(os.getcwd())
from main import VALID_PHASES

def run_phase(phase, run_id=None):
    cmd = [sys.executable, "main.py", "--phase", phase, "--mode", "full"]
    if run_id:
        cmd += ["--run-id", run_id]
        
    print(f"\n--- Running Phase: {phase} ---")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error in Phase {phase}: Pipeline stopped.")
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, help="Existing RUN_ID to continue")
    args = parser.parse_args()
    
    print(f"Starting Pipeline Restoration | RUN_ID: {args.run_id or 'NEW'}")
    
    for phase in VALID_PHASES:
        if not run_phase(phase, args.run_id):
            print(f"Pipeline stopped at phase: {phase}")
            sys.exit(1)
            
    print("\n--- Pipeline Restoration SUCCESS! ---")

if __name__ == "__main__":
    main()
