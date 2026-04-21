import subprocess
import sys

phases = [
    "1_data_check",
    "2_build_raw",
    "3_train_raw",
    "4_build_full",
    "5_train_final",
    "6_retrain",
    "7_inference",
    "8_submission",
    "9_intelligence"
]

def run_phase(phase):
    cmd = [sys.executable, "main.py", "--phase", phase, "--mode", "full"]
    print(f"\n--- Running Phase: {phase} ---")
    # Don't capture output so we can see it in real-time
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error in Phase {phase}: Pipeline stopped.")
        return False
    return True

def main():
    for phase in phases:
        if not run_phase(phase):
            print(f"Pipeline stopped at phase: {phase}")
            sys.exit(1)
    print("\n--- Pipeline Restoration SUCCESS! ---")

if __name__ == "__main__":
    main()
