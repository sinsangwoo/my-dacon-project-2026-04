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
            "stability_index": 0.0, 
            "config": {
                "history_window": 100, 
                "baseline_f1_threshold": 0.20,
                "dynamic_cut_buffer": 0.05 # 5% relative buffer
            }
        }

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def calculate_stability_index(self, window=10):
        """Calculate variance stability based on recent runs. (Lower is more stable)"""
        runs = [r for r in self.registry.get("runs", []) if r.get("status") == "PASSED"]
        if len(runs) < 3:
            return 1.0 # Default highly unstable for beginning
        
        recent_variances = [r.get("variance_ratio", 0.0) for r in runs[-window:]]
        stability = float(np.std(recent_variances)) # Closer to 0 means higher stability
        return stability

    def get_adaptive_params(self, stability_index):
        """Determine if we should be in Strict or Loose mode based on stability.
        PHASE 2 & 3: Removed extreme_f1 constraint and variance_ratio hard reject.
        """
        if stability_index < 0.05:
            mode = "STRICT"
        elif stability_index < 0.1:
            mode = "MODERATE"
        else:
            mode = "LOOSE"

        return {
            "mode": mode,
            "log_variance_ratio_only": True
        }

    def filtering_stage(self, current_run, params):
        """PHASE 2 & 3: NO hard constraints. Only log variance ratio."""
        rejections = []
        v_ratio = current_run.get("variance_ratio", 0.0)
        logger.info(f"[LOG_ONLY] Variance ratio: {v_ratio:.4f} (no reject threshold)")
        return rejections

    def find_pareto_frontier(self, all_runs):
        """
        PURPOSE: Identify non-dominated runs across multiple objectives.
        INPUT: all_runs (List[dict]) - Must comply with Config.METRIC_SCHEMA.
        OUTPUT: List[dict] - Subset of runs on the Pareto frontier.
        FAILURE: Raises KeyError if any run is missing a required metric.
        """
        if not all_runs:
            return []
            
        # [PHASE 4: INTELLIGENCE INPUT HARDENING]
        # Direct access only, no .get() usage to hide contract violations.
        metrics = np.array([
            [r["mean_mae"], r["worst_mae"], r["extreme_mae"]] for r in all_runs
        ])
        is_efficient = np.ones(metrics.shape[0], dtype=bool)
        for i, m in enumerate(metrics):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(metrics[is_efficient] < m, axis=1) 
                is_efficient[i] = True
        return [all_runs[i] for i, eff in enumerate(is_efficient) if eff]

    def apply_dynamic_cut(self, pareto_runs):
        """
        PURPOSE: Prune Pareto frontier based on a relative performance buffer.
        INPUT: pareto_runs (List[dict]) - Non-dominated runs.
        OUTPUT: List[dict] - Filtered runs within the dynamic cut buffer.
        FAILURE: Returns empty list if pareto_runs is empty.
        """
        if not pareto_runs:
            return []
        best_mean = min(r["mean_mae"] for r in pareto_runs)
        best_extreme = min(r["extreme_mae"] for r in pareto_runs)
        buffer = self.registry["config"].get("dynamic_cut_buffer", 0.05)
        filtered = []
        for r in pareto_runs:
            if (r["mean_mae"] <= best_mean * (1 + buffer)) or \
               (r["extreme_mae"] <= best_extreme * (1 + buffer)):
                filtered.append(r)
            else:
                logger.info(f"Pruning run {r['run_id']} from frontier (Relative gap > {buffer*100}%)")
        return filtered

    def calculate_risk_score(self, run):
        """
        PURPOSE: [PHASE 4] Compute a single selection score for model ranking.
        INPUT: run (dict) - Must comply with Config.METRIC_SCHEMA.
        OUTPUT: float (LOWER is better).
        FAILURE: Raises KeyError if any required metric is missing.
        """
        # [PHASE 5: DEFAULT VALUE PROHIBITION]
        # Accessing keys directly to ensure structural integrity.
        score = (
            run["mean_mae"] * 0.5 +
            run["extreme_mae"] * 0.3 +
            run["worst_mae"] * 0.2
        )
        return float(score)

    def analyze_delta(self, current, previous_best):
        """
        PURPOSE: Compare current run against the historical best.
        INPUT: current (dict), previous_best (dict)
        OUTPUT: dict containing deltas and feature changes.
        FAILURE: Returns minimal info if previous_best is None.
        """
        if not previous_best:
            return {"status": "first_best", "deltas": {}}
        deltas = {
            "mean_mae": current["mean_mae"] - previous_best["mean_mae"],
            "worst_mae": current["worst_mae"] - previous_best["worst_mae"],
            "extreme_mae": current["extreme_mae"] - previous_best["extreme_mae"],
            "variance_ratio": current["variance_ratio"] - previous_best["variance_ratio"]
        }
        cur_features = set(current.get("features", []))
        prev_features = set(previous_best.get("features", []))
        return {
            "deltas": deltas,
            "feature_changes": {"added": list(cur_features - prev_features), "removed": list(prev_features - cur_features)}
        }

    def run_risk_focused_pipeline(self, current_run_id, metrics):
        """
        PURPOSE: [PHASE 4: INTELLIGENCE INPUT HARDENING]
        Main entry point for model ranking and knowledge extraction.
        
        INPUT:
        - current_run_id: str
        - metrics: dict - MUST match Config.METRIC_SCHEMA exactly.
        
        OUTPUT:
        - bool: True if current run is the new historical best.
        
        FAILURE:
        - Raises AssertionError if metrics do not match the schema.
        - Raises KeyError if any logic attempts to access missing keys.
        """
        # [PHASE 3: ASSERTION FIREWALL]
        from src.config import Config
        for key in Config.METRIC_SCHEMA:
            assert key in metrics, f"[METRIC_MISSING_FATAL] Intelligence requires '{key}' but it was missing!"

        logger.info(f"--- [FAIR_INTELLIGENCE: {current_run_id}] ---")
        stability = self.calculate_stability_index()
        params = self.get_adaptive_params(stability)
        logger.info(f"[ADAPTIVE] Mode: {params['mode']} | StabilityIdx: {stability:.4f}")
        current_run_data = {"run_id": current_run_id, "timestamp": datetime.now().isoformat(), **metrics}
        rejections = self.filtering_stage(current_run_data, params)
        if rejections:
            logger.warning(f"[REJECTED] {current_run_id} failed hard constraints: {rejections}")
            current_run_data["status"] = "REJECTED"
            current_run_data["rejection_reason"] = rejections
            self.registry["runs"].append(current_run_data)
            self._save_registry()
            # Still try to create folder for rejected runs comparison
            os.makedirs(f"metadata/{current_run_id}", exist_ok=True)
            with open(f"metadata/{current_run_id}/filtered_runs.json", 'w') as f:
                json.dump({"run_id": current_run_id, "reasons": rejections}, f, indent=2)
            return False

        # Pareto Stage
        valid_runs = [r for r in self.registry["runs"] if r.get("status") == "PASSED"]
        valid_runs.append(current_run_data)
        raw_pareto = self.find_pareto_frontier(valid_runs)
        pareto_runs = self.apply_dynamic_cut(raw_pareto)
        
        # Ensure output directory exists before writing
        os.makedirs(f"metadata/{current_run_id}", exist_ok=True)
        with open(f"metadata/{current_run_id}/pareto_runs.json", 'w') as f:
            json.dump(pareto_runs, f, indent=2)
            
        # Best Selection
        scored_pareto = []
        for p in pareto_runs:
            p_copy = p.copy(); p_copy["risk_score"] = self.calculate_risk_score(p)
            scored_pareto.append(p_copy)
        best_run_data = min(scored_pareto, key=lambda x: x["risk_score"])
        is_new_best = best_run_data["run_id"] == current_run_id
        if is_new_best:
            logger.info(f"[WINNER] {current_run_id} is the NEW BEST run!")
            self.registry["current_best_run_id"] = current_run_id
        
        # Delta Analysis
        prev_best_id = self.registry.get("current_best_run_id")
        prev_best_run = next((r for r in self.registry["runs"] if r["run_id"] == prev_best_id), None)
        delta_info = self.analyze_delta(current_run_data, prev_best_run)
        with open(f"metadata/{current_run_id}/run_comparison.json", 'w') as f:
            json.dump(delta_info, f, indent=2)
        with open(f"metadata/{current_run_id}/best_run.json", 'w') as f:
            json.dump(best_run_data, f, indent=2)
            
        # Maintenance
        current_run_data["status"] = "PASSED"
        current_run_data["is_best"] = is_new_best
        self.registry["runs"].append(current_run_data)
        self.registry["stability_index"] = stability
        if len(self.registry["runs"]) > self.registry["config"]["history_window"]:
            archived = self.registry["runs"].pop(0)
            archive_dir = 'metadata/archive'; os.makedirs(archive_dir, exist_ok=True)
            with open(f"{archive_dir}/{archived['run_id']}.json", 'w') as f: json.dump(archived, f, indent=2)
        self._save_registry()
        logger.info(f"--- [/FAIR_INTELLIGENCE: {current_run_id}] ---")
        return is_new_best
