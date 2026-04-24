import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentIntelligence:
    def __init__(self, registry_path='metadata/experiment_registry.json'):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        
    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "runs": [], 
            "current_best_run_id": None, 
            "config": {
                "risk_threshold": 2.0
            }
        }

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        # Create backup before overwrite
        if os.path.exists(self.registry_path):
            import shutil
            shutil.copy2(self.registry_path, self.registry_path + ".bak")
            
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def log_experiment_audit(self, run_id, metrics, adv_auc, risk_score, fold_stats=None):
        """
        PURPOSE: [ANTI-CV-ILLUSION AUDIT LOG]
        Records all validation dimensions for forensic analysis.
        """
        risk_threshold = self.registry.get("config", {}).get("risk_threshold", 2.0)
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "adv_auc": adv_auc,
            "risk_score": risk_score,
            "fold_stats": fold_stats,
            "status": "APPROVED" if risk_score < risk_threshold else "RISKY"
        }

        # [MISSION: ARTIFACT ISOLATION] Save run-local report PRIMARY
        report_path = f"logs/{run_id}/validation_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(run_data, f, indent=2)
            
        # Global Registry Update (Aggregator)
        self.registry["runs"].append(run_data)
        
        # Check for new best
        if run_data["status"] == "APPROVED":
            current_best_id = self.registry.get("current_best_run_id")
            if current_best_id:
                prev_best = next((r for r in self.registry["runs"] if r["run_id"] == current_best_id), None)
                if prev_best and metrics["mae"] < prev_best["metrics"]["mae"]:
                    self.registry["current_best_run_id"] = run_id
                    logger.info(f"[NEW_BEST] Run {run_id} is the new best!")
            else:
                self.registry["current_best_run_id"] = run_id
        
        try:
            self._save_registry()
        except Exception as e:
            logger.error(f"[REGISTRY_FAIL] Could not update global registry: {e}")
            
        return run_data

    def generate_intelligence_summary(self, run_id):
        """Phase 9: Final Intelligence Report."""
        run = next((r for r in self.registry["runs"] if r["run_id"] == run_id), None)
        if not run:
            logger.error(f"Run {run_id} not found in registry.")
            return
            
        m = run["metrics"]
        
        report = [
            f"\n{'='*60}",
            f" ANTI-CV-ILLUSION INTELLIGENCE REPORT | {run_id}",
            f"{'='*60}",
            f" STATUS: {run['status']}",
            f" RISK SCORE: {run['risk_score']:.4f}",
            f" ADV AUC: {run['adv_auc']:.4f}",
            f"{'-'*60}",
            f" [GLOBAL PERFORMANCE]",
            f"  MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | MedAE: {m['med_ae']:.4f}",
            f"{'-'*60}",
            f" [DISTRIBUTION RATIOS]",
            f"  Mean Ratio: {m['mean_ratio']:.4f}",
            f"  Std Ratio:  {m['std_ratio']:.4f}",
            f"  P99 Ratio:  {m['p99_ratio']:.4f}",
            f"{'-'*60}",
            f" [TAIL PERFORMANCE]",
            f"  Q90-99 MAE: {m['Q90_99_mae']:.4f}",
            f"  Q99-100 MAE: {m['Q99_100_mae']:.4f}",
            f"  Top 1% Error: {m['mae_top_1']:.4f}",
            f"{'='*60}\n"
        ]
        
        summary_text = "\n".join(report)
        print(summary_text)
        
        with open(f"logs/{run_id}/intelligence_summary.txt", "w") as f:
            f.write(summary_text)
        
        if run['status'] == "RISKY":
            logger.warning("!!! [SUBMISSION_BLOCKED] High Risk Score detected. Verify distribution drift before submitting. !!!")
