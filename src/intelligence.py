import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import shutil
from .config import Config
from .utils import save_json

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
            shutil.copy2(self.registry_path, self.registry_path + ".bak")
            
        save_json(self.registry, self.registry_path)

    def log_experiment_audit(self, run_id, metrics, adv_auc, risk_results, fold_stats=None, metadata=None):
        """
        PURPOSE: [ANTI-CV-ILLUSION AUDIT LOG]
        Records all validation dimensions for forensic analysis.
        """
        risk_threshold = self.registry.get("config", {}).get("risk_threshold", 5.0) # Increased threshold for new scale
        
        # Extract total score for status check
        total_risk = risk_results['total'] if isinstance(risk_results, dict) else risk_results
        
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "adv_auc": adv_auc,
            "risk_score": total_risk,
            "risk_breakdown": risk_results['breakdown'] if isinstance(risk_results, dict) else {},
            "fold_stats": fold_stats,
            "metadata": metadata or {},
            "status": "APPROVED" if total_risk < risk_threshold else "RISKY"
        }

        # [MISSION: ARTIFACT ISOLATION] Save run-local report PRIMARY
        report_path = f"logs/{run_id}/validation_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        save_json(run_data, report_path)
            
        # [DETERMINISTIC_REGISTRY_FIX] Update existing run or append
        existing_idx = next((i for i, r in enumerate(self.registry["runs"]) if r["run_id"] == run_id), None)
        if existing_idx is not None:
            self.registry["runs"][existing_idx] = run_data
        else:
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

    def calculate_top_10_risks(self, run_data):
        """
        PURPOSE: [PHASE 9: FORENSIC RISK AUDIT]
        Calibrates risks based on the 2-Stage architecture's performance.
        Reflects: Fixed issues (Capping) vs. Environmental constraints (Drift).
        """
        m = run_data["metrics"]
        # Extract fold MAEs robustly
        raw_fs = run_data.get("fold_stats", [])
        fs = []
        for s in raw_fs:
            if isinstance(s, dict): fs.append(s.get('mae', 0.0))
            else: fs.append(float(s))
        adv_auc = run_data.get("adv_auc", 0.5)
        
        cv_mean = np.mean(fs) if fs else 1e-9
        cv_std = np.std(fs) if fs else 0.0
        
        risks = {}
        
        # 1. 환경적 분포 불일치 (External Drift)
        # AUC 0.99는 환경적 상수이므로, 0.7 이상부터 비례하여 상승
        risks["환경적 분포 불일치"] = min(100, max(0, (adv_auc - 0.7) / 0.3 * 100))
        
        # 2. 타겟 예측 바이어스 (Global Bias)
        # MeanRatio 0.72는 전체적인 Under-prediction을 의미함
        mean_ratio = m.get("mean_ratio", 1.0)
        risks["전체 예측 바이어스"] = min(100, abs(1.0 - mean_ratio) / 0.3 * 100)
        
        # 3. 예측 분산 압축 (Variance Compression)
        # StdRatio 0.73은 여전히 모델이 조심스럽게(Cautious) 예측하고 있음을 의미
        std_ratio = m.get("std_ratio", 1.0)
        risks["예측 분산 압축"] = min(100, max(0, (1.0 - std_ratio) / 0.5 * 100))
        
        # 4. 검증셋 신뢰도 (CV Reliability)
        # CV Std/Mean이 0.2를 넘어가면 LB와 어긋날 확률이 매우 높음
        cv_instability = (cv_std / cv_mean) if cv_mean > 0 else 0
        risks["검증셋 신뢰도 위기"] = min(100, cv_instability / 0.2 * 100)
        
        # 5. 꼬리 영역 잔여 오차 (Tail Residual Risk)
        # P99 Ratio가 0.61로 개선되었으나 여전히 1.0에 미달함
        p99_ratio = m.get("p99_ratio", 0.3)
        risks["꼬리 영역 잔여 오차"] = min(100, max(0, (1.0 - p99_ratio) / 0.7 * 100))
        
        # 6. 모델 구조적 한계 (Capping Risk) - [RESOLVED INDICATOR]
        # 2-Stage 이후 P99 Ratio가 0.5를 넘으면 이 리스크는 사실상 소멸된 것으로 간주
        risks["물리적 예측 한계(Cap)"] = min(100, max(0, (0.5 - p99_ratio) / 0.2 * 100))
        
        # 7. 피처 변별력 상실 (Feature Blindness)
        # Top 1% 에러 기여도가 너무 높으면(10배 이상) 피처가 부족하다는 증거
        error_contribution = m.get("mae_top_1", 0.0) / (m.get("mae", 1.0) + 1e-9)
        risks["피처 변별력 부족"] = min(100, max(0, (error_contribution - 5) / 10 * 100))
        
        # 8. 학습/테스트 불연속성 (Temporal Leakage)
        # AUC가 0.95를 넘으면 시계열적 구조가 완전히 무너졌음을 의미
        risks["시계열 구조 불연속성"] = min(100, max(0, (adv_auc - 0.9) / 0.1 * 100))
        
        # 9. 시스템 자원 임계치 (Hardware Risk)
        peak_mem = run_data.get("metadata", {}).get("peak_memory_mb", 4096)
        risks["자원 병목 리스크"] = min(100, (peak_mem / 16384) * 100)
        
        # 10. 일반화 성능 괴리 (Generalization Gap)
        # CV와 LB의 구조적 차이 (Adversarial Risk 기반)
        risks["일반화 성능 괴리"] = min(100, run_data.get("risk_score", 0.0) / 10.0 * 100)
        
        return risks

    def generate_intelligence_summary(self, run_id):
        """Phase 9: Final Intelligence Report - FORENSIC DIAGNOSIS EDITION."""
        # [SSOT_FIX] Local import removed
        run = next((r for r in self.registry["runs"] if r["run_id"] == run_id), None)
        if not run:
            logger.error(f"Run {run_id} not found in registry.")
            return
            
        m = run["metrics"]
        metadata = run.get("metadata", {})
        fs_raw = run.get("fold_stats", [])
        fs = [s.get('mae', 0.0) if isinstance(s, dict) else float(s) for s in fs_raw]
            
        cv_mean = np.mean(fs) if fs else 0.0
        cv_std = np.std(fs) if fs else 0.0
        
        # [v4.0] Execution Intelligence
        fp_cost = m.get("fp_cost", 0.0)
        gain_capture = m.get("gain_capture", 0.0)
        
        # Strategic Verdict logic
        if fp_cost > 1.0:
            execution_verdict = "OVER-AGGRESSIVE (High FP Damage)"
            advice = "Increase damping (p^3 or higher) or sharpen classification threshold."
        elif gain_capture < 0.05:
            execution_verdict = "TOO CAUTIOUS (Signal Missing)"
            advice = "Decrease damping (p^1.5) or check feature importance for tail signal."
        else:
            execution_verdict = "OPTIMIZED (Balanced Risk/Reward)"
            advice = "Structure is stable. Focus on new feature discovery or hyper-parameter tuning."

        # Forensic Status
        p99_status = "SUCCESS" if m.get('p99_ratio', 0) > 0.5 else "CRITICAL"
        
        report = [
            f"\n{'='*75}",
            f" [ATTACK ZONE] FORENSIC DIAGNOSIS REPORT | {run_id}",
            f"{'='*75}",
            f" [STRATEGIC VERDICT]",
            f"  Status:    {execution_verdict}",
            f"  Advice:    {advice}",
            f"{'-'*75}",
            f" [EXECUTION AUDIT]",
            f"  1. FP Damage (Risk):     {fp_cost:.4f} MAE penalty",
            f"  2. Gain Capture (Reward): {gain_capture*100:.2f}% of possible signal recovered",
            f"  3. P99 Sensitivity:      {m.get('p99_ratio', 0):.4f} ({p99_status})",
            f"{'-'*75}",
            f" [STRUCTURAL INTEGRITY AUDIT]",
            f"  1. Feature Alignment:    {metadata.get('num_features', 'N/A')} features (Determinism: {'PASS' if metadata.get('feature_alignment', True) else 'FAIL'})",
            f"  2. Drift Quarantine:     {metadata.get('pruned_count', 0)} dropped ({metadata.get('prune_rate', 0):.1%})",
            f"  3. OOF Validity:         {metadata.get('oof_valid_rate', 0):.1%} samples validated (expanding-window constraint)",
            f"  4. PCA Stability:        {metadata.get('pca_input_count', 0)}/30 inputs used for reconstruction",
            f"{'-'*75}",
            f" [AI 모델 경진대회 성능 저하 10대 위험 요인]",
            f"  순위 | 위험 요인 (Risk Factor)    | 지수(%) | 상태 (Status)",
            f"  -----|----------------------------|---------|----------"
        ]
        
        top_risks = self.calculate_top_10_risks(run)
        sorted_risks = sorted(top_risks.items(), key=lambda x: x[1], reverse=True)
        for i, (name, val) in enumerate(sorted_risks, 1):
            status = "CRITICAL" if val > 80 else "WARNING" if val > 40 else "HEALTHY"
            report.append(f"  #{i:<2} | {name:<26} | {val:>6.1f}% | {status}")
            
        report.extend([
            f"{'-'*75}",
            f" [PERFORMANCE SNAPSHOT]",
            f"  MAE:   {m['mae']:.4f} (CV: {cv_mean:.4f} ±{cv_std:.4f})",
            f"  Tail MAE (Q99-100): {m.get('Q99_100_mae', 0.0):.4f}",
            f"  Mean Ratio: {m.get('mean_ratio', 0):.4f} (Under-prediction check)",
            f"  Std Ratio:  {m.get('std_ratio', 0):.4f} (Variance recovery check)",
            f"{'='*75}\n"
        ])
        
        summary_text = "\n".join(report)
        print(summary_text)
        
        # Save artifacts
        for path in [f"{Config.LOG_DIR}/intelligence_summary.txt", f"{Config.SUMMARY_DIR}/intelligence_summary.txt"]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding='utf-8') as f:
                f.write(summary_text)
            
        if run['status'] == "RISKY":
            logger.warning("!!! [HIGH RISK] Structural instability detected. See report. !!!")
