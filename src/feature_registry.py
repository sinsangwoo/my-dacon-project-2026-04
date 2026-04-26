"""
[CONTEXT — DO NOT REMOVE]

src/feature_registry.py — Feature Drop Registry & Pruning Manifest

WHY THIS FILE EXISTS:
- The previous pipeline mutated FEATURE_SCHEMA indirectly by computing drop sets
  and applying them locally, but never recorded WHY each feature was dropped,
  WHAT threshold was used, or HOW that threshold was derived.
- This caused "silent knowledge loss": once a feature was dropped, no downstream
  phase could reconstruct the decision logic, making debugging and reproducibility
  impossible.

DESIGN PRINCIPLES:
1. FEATURE_SCHEMA is NEVER mutated — it remains the canonical static manifest.
2. This registry is the ONLY authorized record of runtime drop decisions.
3. ALL thresholds stored here MUST include their derivation logic string.
4. The registry is serializable to JSON for cross-phase persistence.
5. PruningManifest wraps the registry for passing from train→test processing,
   ensuring test uses identical thresholds computed on train (no leakage).

[END CONTEXT]
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from .utils import save_json

logger = logging.getLogger(__name__)


@dataclass
class DropRecord:
    """
    Immutable record of a single feature-drop decision.

    Fields:
        feature    : Name of the dropped feature.
        reason     : Category of drop (nan_drop | corr_drop | variance_drop | schema_extra).
        stat_value : The observed statistic that triggered the drop (e.g., NaN ratio).
        threshold  : The derived threshold that was exceeded.
        derivation : Human-readable string explaining HOW the threshold was computed.
    """
    feature: str
    reason: str
    stat_value: float
    threshold: float
    derivation: str


class FeatureDropRegistry:
    """
    Authoritative, append-only audit log of every feature dropped during pipeline execution.

    [WHY_THIS_CHANGE]
    Problem:
        Drop decisions were scattered across build_base_features() with no central record.
        Future debugging required re-running the full pipeline to see why a feature vanished.
    Root Cause:
        No separation between "what was dropped" and "why it was dropped".
    Decision:
        Centralize all drop records in this registry with full statistical provenance.
    Why this approach (not alternatives):
        - Mutating FEATURE_SCHEMA: violates immutability contract.
        - Plain sets: lose threshold/derivation metadata.
        - This registry: preserves full audit trail, serializable, cross-phase compatible.
    Expected Impact:
        Every dropped feature is traceable to its exact statistic and derived threshold.
    """

    def __init__(self):
        self._records: List[DropRecord] = []
        self._derived_thresholds: Dict[str, Dict[str, Any]] = {}

    def record_drop(
        self,
        feature: str,
        reason: str,
        stat_value: float,
        threshold: float,
        derivation: str
    ):
        """
        Append a single drop record. This is the ONLY way to register a feature drop.

        Parameters
        ----------
        feature    : Column name being dropped.
        reason     : One of 'nan_drop', 'corr_drop', 'variance_drop', 'schema_extra'.
        stat_value : Actual measured value (e.g., NaN ratio = 0.21).
        threshold  : The derived threshold value that was exceeded.
        derivation : String description of how threshold was computed
                     (e.g., "P85 of NaN ratio distribution over 425 features = 0.17").
        """
        rec = DropRecord(
            feature=feature,
            reason=reason,
            stat_value=float(stat_value),
            threshold=float(threshold),
            derivation=derivation
        )
        self._records.append(rec)

    def record_threshold(self, name: str, value: float, derivation: str, supporting_stats: dict):
        """
        Register a derived threshold so it can be inspected and logged.

        Parameters
        ----------
        name             : Threshold identifier (e.g., 'nan_threshold', 'corr_threshold').
        value            : The derived numeric value.
        derivation       : Free-text derivation log (e.g., "Q3 + 1.5*IQR of upper-tri correlations").
        supporting_stats : Dict of supporting statistics (e.g., {'Q1': 0.12, 'Q3': 0.54, 'IQR': 0.42}).
        """
        self._derived_thresholds[name] = {
            "value": float(value),
            "derivation": derivation,
            "supporting_stats": supporting_stats
        }
        logger.info(
            f"[REGISTRY] Threshold '{name}' derived = {value:.6f} | "
            f"Method: {derivation} | Stats: {supporting_stats}"
        )

    def get_dropped_features(self, reason: Optional[str] = None) -> List[str]:
        """Return list of all dropped feature names, optionally filtered by reason."""
        if reason:
            return [r.feature for r in self._records if r.reason == reason]
        return [r.feature for r in self._records]

    def get_dropped_set(self, reason: Optional[str] = None):
        """Return set of dropped feature names for fast membership testing."""
        return set(self.get_dropped_features(reason))

    def get_threshold(self, name: str) -> Optional[float]:
        """Return the numeric value of a previously derived threshold."""
        entry = self._derived_thresholds.get(name)
        return entry["value"] if entry else None

    def summary(self) -> Dict[str, int]:
        """Return counts of dropped features per reason."""
        counts = {}
        for rec in self._records:
            counts[rec.reason] = counts.get(rec.reason, 0) + 1
        return counts

    def log_summary(self):
        """Emit a structured summary to logger."""
        s = self.summary()
        total = sum(s.values())
        logger.info("[REGISTRY_SUMMARY] ================================")
        logger.info(f"[REGISTRY_SUMMARY] Total features dropped: {total}")
        for reason, count in s.items():
            thresh_entry = None
            for name, entry in self._derived_thresholds.items():
                if reason.replace('_drop', '') in name:
                    thresh_entry = entry
                    break
            thresh_str = f" | threshold={thresh_entry['value']:.6f} ({thresh_entry['derivation']})" if thresh_entry else ""
            logger.info(f"[REGISTRY_SUMMARY]   {reason}: {count} features{thresh_str}")
        logger.info("[REGISTRY_SUMMARY] Derived Thresholds:")
        for name, entry in self._derived_thresholds.items():
            logger.info(
                f"[REGISTRY_SUMMARY]   {name} = {entry['value']:.6f} "
                f"| {entry['derivation']} | stats={entry['supporting_stats']}"
            )
        logger.info("[REGISTRY_SUMMARY] ================================")

    def to_dict(self) -> dict:
        """Serialize registry to a plain dictionary (JSON-serializable)."""
        return {
            "drop_records": [asdict(r) for r in self._records],
            "derived_thresholds": self._derived_thresholds,
            "summary": self.summary()
        }

    def save(self, path: str):
        """Persist registry to a JSON file for cross-phase consumption."""
        save_json(self.to_dict(), path)
        logger.info(f"[REGISTRY] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureDropRegistry":
        """Reconstruct a registry from a previously saved JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reg = cls()
        for rec_dict in data.get("drop_records", []):
            reg._records.append(DropRecord(**rec_dict))
        reg._derived_thresholds = data.get("derived_thresholds", {})
        logger.info(
            f"[REGISTRY] Loaded from {path} | "
            f"{len(reg._records)} drop records | "
            f"{len(reg._derived_thresholds)} thresholds"
        )
        return reg


@dataclass
class PruningManifest:
    """
    Immutable snapshot of all pruning decisions made on the TRAIN set.

    [WHY_THIS_CHANGE]
    Problem:
        build_base_features() was called independently for train and test.
        Each call recomputed thresholds on the passed dataframe.
        For test data, this means thresholds were computed on TEST distribution — DATA LEAKAGE.
    Root Cause:
        Stateless function design; no mechanism to pass train-derived decisions to test processing.
    Decision:
        Capture all train-time pruning decisions in a PruningManifest and apply them
        unchanged when processing the test set.
    Why this approach (not alternatives):
        - Recomputing on test: leakage.
        - Global mutable state: fragile, not thread-safe.
        - PruningManifest: immutable, serializable, explicit contract.
    Expected Impact:
        Train and test always go through identical pruning with zero leakage from test distribution.

    Fields:
        nan_threshold   : Derived NaN drop threshold (computed on train NaN distribution).
        corr_threshold  : Derived correlation pruning threshold (cluster-based, computed on train).
        var_threshold   : Derived variance floor threshold (P5 of train variance distribution).
        cols_to_drop_nan: Feature names dropped due to high NaN on train.
        cols_to_drop_corr: Feature names dropped due to high correlation on train.
        cols_to_drop_var: Feature names dropped due to near-zero variance on train.
        derivation_log  : Dict mapping threshold names to their derivation descriptions.
        regime_boundaries: NaN regime boundaries (low/medium/high) from train distribution.
    """
    nan_threshold: float
    corr_threshold: float
    var_threshold: float
    cols_to_drop_nan: List[str]
    cols_to_drop_corr: List[str]
    cols_to_drop_var: List[str]
    derivation_log: Dict[str, str] = field(default_factory=dict)
    regime_boundaries: Dict[str, float] = field(default_factory=dict)
    # [TASK 5 — CAUSAL FALLBACK LEAKAGE FIX]
    # Stores per-column means computed on TRAIN data only.
    # Used as fallback in add_time_series_features() instead of df[col].mean()
    # which would leak test distribution when processing test data.
    train_col_means: Dict[str, float] = field(default_factory=dict)
    # [TASK 11 — EXTREME QUANTILE CONSISTENCY]
    # Stores P95 quantiles computed on TRAIN data only.
    # Used in add_extreme_detection_features() to ensure test uses train thresholds.
    extreme_quantiles: Dict[str, float] = field(default_factory=dict)

    def all_dropped(self) -> List[str]:
        """Return flat list of all features to drop (union of all reasons)."""
        return list(set(self.cols_to_drop_nan) | set(self.cols_to_drop_corr) | set(self.cols_to_drop_var))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        save_json(self.to_dict(), path)
        logger.info(f"[PRUNING_MANIFEST] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PruningManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls(
            nan_threshold=data["nan_threshold"],
            corr_threshold=data["corr_threshold"],
            var_threshold=data["var_threshold"],
            cols_to_drop_nan=data["cols_to_drop_nan"],
            cols_to_drop_corr=data["cols_to_drop_corr"],
            cols_to_drop_var=data["cols_to_drop_var"],
            derivation_log=data.get("derivation_log", {}),
            regime_boundaries=data.get("regime_boundaries", {}),
            train_col_means=data.get("train_col_means", {}),
            extreme_quantiles=data.get("extreme_quantiles", {})
        )
        logger.info(
            f"[PRUNING_MANIFEST] Loaded from {path} | "
            f"nan_drop={len(manifest.cols_to_drop_nan)} | "
            f"corr_drop={len(manifest.cols_to_drop_corr)} | "
            f"var_drop={len(manifest.cols_to_drop_var)}"
        )
        return manifest
