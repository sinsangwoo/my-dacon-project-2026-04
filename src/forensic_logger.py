import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .utils import get_logger, save_json

class ForensicLogger:
    """Forensic Logging System for Model Failure Analysis.
    
    Structure:
    1. GENERALIZATION (OOF, LB, Gap)
    2. SCENARIO LEVEL (Worst/Best 10)
    3. DRIFT (KS-stat, Mean/Std)
    4. MODEL PERFORMANCE (Cat/LGB, Delay segments)
    5. DYNAMICS (Variance ratio)
    6. FAILURE CASE (Top N error samples)
    7. FEATURE HEALTH (Count, Corr, Condition number)
    8. EXPERIMENT SUMMARY (Config, Run info)
    """
    
    def __init__(self, run_id=None, mode='full', log_dir='logs', summary_dir='metadata'):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mode = mode # 'full' or 'lite'
        self.log_dir = log_dir
        self.summary_dir = summary_dir
        self.audit_path = os.path.join(summary_dir, "audit_history.json")
        
        # Initialize directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        
        self.sections = {
            "GENERALIZATION": {},
            "SCENARIO_LEVEL": {},
            "DRIFT": {},
            "MODEL_PERFORMANCE": {},
            "DYNAMICS": {},
            "FAILURE_CASE": {},
            "FEATURE_HEALTH": {},
            "EXPERIMENT_SUMMARY": {}
        }
        
        self.logger = get_logger(f"forensic_{run_id}" if run_id else "forensic")
        
        # Load existing sections if they exist in volatile log_dir
        self._load_existing_sections()

    def _load_existing_sections(self):
        """Load sections from previous phases in the same run (Non-critical)."""
        for section in self.sections:
            path = os.path.join(self.log_dir, f"forensic_{section.lower()}.json")
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        self.sections[section].update(json.load(f))
                except Exception as e:
                    self.logger.warning(f"[FORENSIC_LOAD_ERR] Could not load section {section}: {str(e)}")
        
    def add_metric(self, section, key, value):
        if section in self.sections:
            self.sections[section][key] = value
            
    def save_section(self, section):
        """Save a specific section to JSON and TXT (Non-critical)."""
        if section not in self.sections:
            return
            
        try:
            data = self.sections[section]
            base_path = os.path.join(self.log_dir, f"forensic_{section.lower()}")
            
            # Save JSON
            save_json(data, f"{base_path}.json", indent=4)
                
            # Save TXT (Summary view) - Pretty format
            with open(f"{base_path}.txt", "w") as f:
                f.write(f"==================================================\n")
                f.write(f"   [FORENSIC: {section}]\n")
                f.write(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"==================================================\n\n")
                
                if section == "SCENARIO_LEVEL":
                    for group in ["worst_10", "best_10"]:
                        f.write(f"--- [{group.upper()}] ---\n")
                        df_view = pd.DataFrame(data.get(group, []))
                        if not df_view.empty:
                            f.write(df_view.to_string(index=False) + "\n\n")
                elif section == "FAILURE_CASE":
                    f.write(f"--- [TOP ERROR SAMPLES] ---\n")
                    df_view = pd.DataFrame(data.get("top_error_samples", []))
                    if not df_view.empty:
                        # Show first 15 columns for readability
                        f.write(df_view.iloc[:, :15].to_string(index=False) + "\n\n")
                elif section == "DRIFT":
                    for key, top_drift in data.items():
                        f.write(f"--- [{key.upper()}] ---\n")
                        df_view = pd.DataFrame(top_drift)
                        if not df_view.empty:
                            f.write(df_view.to_string(index=False) + "\n\n")
                elif section == "DYNAMICS":
                    f.write(f"--- [SCENARIO DYNAMICS] ---\n")
                    df_view = pd.DataFrame(data.get("scenario_dynamics", []))
                    if not df_view.empty:
                        # Show only a sample or top/bottom
                        f.write("Worst 10 by ratio (Low ratio = Sequence failure risk):\n")
                        f.write(df_view.sort_values("ratio").head(10).to_string(index=False) + "\n\n")
                else:
                    for k, v in data.items():
                        if isinstance(v, dict):
                            f.write(f"[{k}]\n")
                            for sub_k, sub_v in v.items():
                                f.write(f"  - {sub_k}: {sub_v}\n")
                        else:
                            f.write(f"{k}: {v}\n")
                f.write("\n")
        except Exception as e:
            self.logger.warning(f"[FORENSIC_SAVE_ERR] Non-critical failure saving section {section}: {str(e)}")
                
    def save_all(self):
        """Save all sections and update permanent audit history."""
        for section in self.sections:
            if self.mode == 'lite' and section not in ["GENERALIZATION", "EXPERIMENT_SUMMARY"]:
                continue
            self.save_section(section)
            
        self._update_audit_history()
        
    def _update_audit_history(self):
        """Update permanent audit history file (Non-critical)."""
        history = []
        try:
            if os.path.exists(self.audit_path):
                with open(self.audit_path, "r") as f:
                    history = json.load(f)
        except Exception as e:
            self.logger.warning(f"[AUDIT_LOAD_ERR] Could not load audit history: {str(e)}")
            history = []
                
        # Create summary for this run
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "metrics": {
                "oof_mae": self.sections["GENERALIZATION"].get("oof_mae"),
                "best_mae": self.sections["GENERALIZATION"].get("best_mae"),
            },
            "summary": self.sections["EXPERIMENT_SUMMARY"]
        }
        
        history.append(summary)
        
        # Keep only last 100 runs to avoid file bloat
        if len(history) > 100:
            history = history[-100:]
            
        try:
            save_json(history, self.audit_path, indent=4)
        except Exception as e:
            self.logger.warning(f"[AUDIT_SAVE_ERR] Could not save audit history: {str(e)}")

    # --- Section Specific Helpers ---
    
    def log_generalization(self, oof_mae, lb_mae=None, unseen_mae=None):
        gap = (oof_mae - lb_mae) if lb_mae else None
        self.add_metric("GENERALIZATION", "oof_mae", oof_mae)
        self.add_metric("GENERALIZATION", "lb_mae", lb_mae)
        self.add_metric("GENERALIZATION", "oof_lb_gap", gap)
        self.add_metric("GENERALIZATION", "unseen_scenario_mae", unseen_mae)
        
    def log_scenario_analysis(self, scenario_df):
        """
        scenario_df expected columns: scenario_id, mae, count, target_mean, target_std, pred_mean, pred_std
        """
        if self.mode == 'lite': return
        
        # Sort and get worst/best 10
        worst_10 = scenario_df.sort_values("mae", ascending=False).head(10).to_dict(orient="records")
        best_10 = scenario_df.sort_values("mae", ascending=True).head(10).to_dict(orient="records")
        
        self.add_metric("SCENARIO_LEVEL", "worst_10", worst_10)
        self.add_metric("SCENARIO_LEVEL", "best_10", best_10)
        
    def log_drift(self, train_df, test_df, features, top_n=20):
        """Perform drift analysis between train and test."""
        if self.mode == 'lite': return
        from scipy.stats import ks_2samp
        
        drift_results = []
        for col in features:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            
            # Simple stats
            tr_mean, tr_std = train_df[col].mean(), train_df[col].std()
            te_mean, te_std = test_df[col].mean(), test_df[col].std()
            
            # KS test
            ks_stat, _ = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
            
            drift_results.append({
                "feature": col,
                "ks_stat": float(ks_stat),
                "train_mean": float(tr_mean),
                "train_std": float(tr_std),
                "test_mean": float(te_mean),
                "test_std": float(te_std)
            })
            
        top_drift = sorted(drift_results, key=lambda x: x['ks_stat'], reverse=True)[:top_n]
        self.add_metric("DRIFT", f"top_{top_n}_drift_features", top_drift)
        
    def log_model_performance(self, y_true, oof_lgb, oof_cat=None):
        """Log model-specific performance and delay segment MAE."""
        from sklearn.metrics import mean_absolute_error
        lgb_mae = mean_absolute_error(y_true, oof_lgb)
        model_metrics = {"lgb_mae": float(lgb_mae)}
        
        if oof_cat is not None and not (oof_cat == 0).all():
            cat_mae = mean_absolute_error(y_true, oof_cat)
            model_metrics["cat_mae"] = float(cat_mae)
            
        # Delay segments
        delay_metrics = {}
        # Define segments: low (<10), mid (10-30), high (>30)
        segments = {
            "low_delay": y_true < 10,
            "mid_delay": (y_true >= 10) & (y_true <= 30),
            "high_delay": y_true > 30
        }
        
        for name, mask in segments.items():
            if mask.any():
                mae = mean_absolute_error(y_true[mask], oof_lgb[mask])
                delay_metrics[f"{name}_mae"] = float(mae)
                delay_metrics[f"{name}_count"] = int(mask.sum())
        
        self.add_metric("MODEL_PERFORMANCE", "model_specific", model_metrics)
        self.add_metric("MODEL_PERFORMANCE", "delay_segments", delay_metrics)
        
    def log_dynamics(self, scenario_ids, y_true, y_pred):
        """Analyze target vs prediction variance ratio per scenario."""
        if self.mode == 'lite': return
        
        df = pd.DataFrame({
            "scenario_id": scenario_ids,
            "y_true": y_true,
            "y_pred": y_pred
        })
        
        # Calculate per-scenario variance ratio
        scenario_dynamics = []
        for sid, group in df.groupby("scenario_id"):
            std_t = group["y_true"].std()
            std_p = group["y_pred"].std()
            ratio = std_p / (std_t + 1e-6)
            scenario_dynamics.append({
                "scenario_id": sid,
                "target_std": float(std_t),
                "pred_std": float(std_p),
                "ratio": float(ratio)
            })
            
        self.add_metric("DYNAMICS", "scenario_dynamics", scenario_dynamics)

    def log_feature_schema(self, schema):
        """[PHASE 7] Log the global feature schema."""
        self.logger.info(f"[FEATURE_SCHEMA] Raw: {len(schema['raw_features'])} | Embed: {len(schema['embed_features'])} | Total: {len(schema['all_features'])}")
        self.add_metric("FEATURE_HEALTH", "schema_counts", {
            "raw": len(schema['raw_features']),
            "embed": len(schema['embed_features']),
            "total": len(schema['all_features'])
        })
        # Save first and last 5 features for verification
        self.add_metric("FEATURE_HEALTH", "schema_head", schema['all_features'][:5])
        self.add_metric("FEATURE_HEALTH", "schema_tail", schema['all_features'][-5:])

    def log_index_alignment(self, df, schema):
        """[PHASE 7] Verify index alignment between DF and Schema."""
        self.logger.info("[INDEX_ALIGNMENT_CHECK] Verifying column indices...")
        df_cols = list(df.columns)
        schema_cols = schema['all_features']
        
        match = (df_cols == schema_cols)
        self.add_metric("FEATURE_HEALTH", "index_alignment_match", match)
        
        if not match:
            self.logger.error("[INDEX_ALIGNMENT_CHECK] Column mismatch detected!")
            # Find first mismatch
            for i, (c1, c2) in enumerate(zip(df_cols, schema_cols)):
                if c1 != c2:
                    self.logger.error(f"Mismatch at index {i}: DF={c1}, Schema={c2}")
                    break
        else:
            self.logger.info("[INDEX_ALIGNMENT_CHECK] Alignment verified successfully.")
        
    def log_failure_cases(self, scenario_ids, y_true, y_pred, feature_df, top_n=50):
        """Extract samples with highest error."""
        if self.mode == 'lite': return
        
        errors = np.abs(y_true - y_pred)
        df = pd.DataFrame({
            "scenario_id": scenario_ids,
            "true": y_true,
            "pred": y_pred,
            "error": errors
        })
        
        # Add key features
        top_idx = df.sort_values("error", ascending=False).head(top_n).index
        samples = df.loc[top_idx].copy()
        
        # Merge with features (first 10 features for context)
        sample_features = feature_df.iloc[top_idx, :10]
        samples = pd.concat([samples, sample_features], axis=1)
        
        self.add_metric("FAILURE_CASE", "top_error_samples", samples.to_dict(orient="records"))
        
    def log_feature_health(self, features, removed_features=None, train_df=None):
        """Feature health metrics: count, max correlation, condition number."""
        health_info = {
            "feature_count": len(features),
            "removed_features": removed_features or []
        }
        
        if train_df is not None and len(features) > 1:
            # ONLY use numerical features for correlation/SVD
            num_features = train_df[features].select_dtypes(include=[np.number]).columns.tolist()
            if not num_features:
                 self.add_metric("FEATURE_HEALTH", "metrics", health_info)
                 return

            # Correlation max (sampled if too large)
            sample_size = min(len(train_df), 10000)
            corr_matrix = train_df[num_features].sample(sample_size, random_state=42).corr().abs()
            # Use copy to avoid read-only issues
            corr_values = corr_matrix.values.copy()
            np.fill_diagonal(corr_values, 0)
            health_info["correlation_max"] = float(np.max(corr_values))
            
            # Condition number (for numerical stability)
            try:
                # SVD for condition number (more robust than np.linalg.cond)
                s = np.linalg.svd(train_df[num_features].sample(sample_size, random_state=42).fillna(0).values, compute_uv=False)
                health_info["condition_number"] = float(s[0] / s[-1]) if s[-1] != 0 else float('inf')
            except:
                health_info["condition_number"] = None
                
        self.add_metric("FEATURE_HEALTH", "metrics", health_info)
        
    def log_experiment_summary(self, config_hash, major_changes=""):
        self.add_metric("EXPERIMENT_SUMMARY", "config_hash", config_hash)
        self.add_metric("EXPERIMENT_SUMMARY", "major_changes", major_changes)
