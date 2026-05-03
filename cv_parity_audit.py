"""
[CV PARITY AUDIT]
목적: proxy holdout 전략의 세 가지 핵심 질문에 수치로 답한다.

Q1. proxy holdout vs train vs test의 KS distance 비교
Q2. proxy holdout vs test adversarial AUC
Q3. proxy holdout MAE vs LB MAE correlation

실행 방법:
  python cv_parity_audit.py --run-id run_20260430_120906

  LB MAE를 알고 있을 때 (Q3 활성화):
  python cv_parity_audit.py --run-id run_20260430_120906 --lb-maes "9.44,9.12,8.87"

결과는 logs/{run_id}/cv_parity_audit_result.json 에 저장된다.
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cv_parity_audit")

# ──────────────────────────────────────────────────────────
# HELPER: scenario ordering (trainer.py와 동일한 로직 — 변경 금지)
# ──────────────────────────────────────────────────────────
def get_scenario_order(df):
    temp = df[["scenario_id", "ID"]].copy()
    temp["id_num"] = temp["ID"].str.extract(r"(\d+)").astype(int)
    return temp.groupby("scenario_id")["id_num"].min().sort_values().index.tolist()


# ──────────────────────────────────────────────────────────
# STEP A: proxy holdout 시나리오 선택
# 방법: 시나리오 수준 적대적 분류기로 test 유사도 점수 부여
# ──────────────────────────────────────────────────────────
def identify_proxy_scenarios(train_df, test_df, feature_cols, holdout_pct=0.20, seed=42):
    """
    각 훈련 시나리오에 'test 유사도 점수'를 부여하고
    상위 holdout_pct를 proxy holdout으로 선택한다.

    [WHY 시나리오 수준 집계]
    시나리오 내 행은 높은 자기상관을 가지므로 행 단위 적대적 분류는
    시나리오 내 패턴을 과적합한다. 시나리오 단위 집계가 더 신뢰할 수 있다.
    """
    logger.info("[STEP A] 시나리오 수준 적대적 분류 시작...")

    agg_funcs = ["mean", "std", "max", "min"]
    train_agg = train_df.groupby("scenario_id")[feature_cols].agg(agg_funcs)
    train_agg.columns = [f"{c}_{f}" for c, f in train_agg.columns]
    train_agg = train_agg.fillna(-999)

    test_agg = test_df.groupby("scenario_id")[feature_cols].agg(agg_funcs)
    test_agg.columns = [f"{c}_{f}" for c, f in test_agg.columns]
    test_agg = test_agg.fillna(-999)

    common = [c for c in train_agg.columns if c in test_agg.columns]
    X_tr = train_agg[common].values
    X_te = test_agg[common].values

    X = np.vstack([X_tr, X_te])
    y = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])

    clf = LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                         random_state=seed, verbose=-1, n_jobs=-1)
    clf.fit(X, y)

    scores = clf.predict_proba(X_tr)[:, 1]
    scenario_ids = list(train_agg.index)

    score_df = pd.DataFrame({"scenario_id": scenario_ids, "test_sim_score": scores})
    score_df = score_df.sort_values("test_sim_score", ascending=False)

    n_holdout = max(1, int(len(score_df) * holdout_pct))
    proxy_set = set(score_df.head(n_holdout)["scenario_id"])

    logger.info(f"[STEP A] proxy holdout 선택: {n_holdout}/{len(score_df)} 시나리오 "
                f"(상위 {holdout_pct:.0%}, 평균 유사도 점수: {score_df.head(n_holdout)['test_sim_score'].mean():.4f})")
    return proxy_set, score_df


# ──────────────────────────────────────────────────────────
# Q1: KS Distance 비교
# train vs test / proxy vs test / remaining_train vs test
# ──────────────────────────────────────────────────────────
def compute_ks_comparison(train_df, test_df, proxy_set, feature_cols):
    """
    세 그룹(전체 train, proxy holdout, 나머지 train)과 test 간
    피처별 KS 통계량을 계산해 비교한다.
    """
    logger.info("[Q1] KS Distance 비교 계산 중...")

    proxy_df = train_df[train_df["scenario_id"].isin(proxy_set)]
    remain_df = train_df[~train_df["scenario_id"].isin(proxy_set)]

    records = []
    for col in feature_cols:
        if col not in test_df.columns:
            continue
        ks_all,   _ = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        ks_proxy, _ = ks_2samp(proxy_df[col].dropna(), test_df[col].dropna())
        ks_remain,_ = ks_2samp(remain_df[col].dropna(), test_df[col].dropna())
        records.append({
            "feature": col,
            "ks_all_train_vs_test":  round(ks_all,   4),
            "ks_proxy_vs_test":      round(ks_proxy,  4),
            "ks_remaining_vs_test":  round(ks_remain, 4),
        })

    df = pd.DataFrame(records)
    summary = {
        "mean_ks_all_train_vs_test":  float(df["ks_all_train_vs_test"].mean()),
        "mean_ks_proxy_vs_test":      float(df["ks_proxy_vs_test"].mean()),
        "mean_ks_remaining_vs_test":  float(df["ks_remaining_vs_test"].mean()),
        "proxy_improvement_vs_all":   float(df["ks_all_train_vs_test"].mean()
                                             - df["ks_proxy_vs_test"].mean()),
        "proxy_improvement_vs_remaining": float(df["ks_remaining_vs_test"].mean()
                                                 - df["ks_proxy_vs_test"].mean()),
        "n_features_checked": len(df),
        "worst_3_proxy_features": df.nlargest(3, "ks_proxy_vs_test")[
            ["feature", "ks_proxy_vs_test"]].to_dict(orient="records"),
        "best_3_proxy_features": df.nsmallest(3, "ks_proxy_vs_test")[
            ["feature", "ks_proxy_vs_test"]].to_dict(orient="records"),
    }

    # 판정
    if summary["proxy_improvement_vs_all"] > 0.01:
        verdict = "PROXY_VALID: proxy holdout이 전체 train보다 test에 가깝다."
    elif summary["mean_ks_proxy_vs_test"] < 0.10:
        verdict = "PROXY_MARGINAL: proxy holdout의 KS 개선은 작지만 절대값이 낮다."
    else:
        verdict = ("PROXY_FAILED: proxy holdout이 test를 충분히 근사하지 못한다. "
                   "훈련·테스트 모집단이 근본적으로 다를 가능성이 높다.")

    summary["verdict"] = verdict

    logger.info(f"[Q1] 전체 train vs test:   mean KS = {summary['mean_ks_all_train_vs_test']:.4f}")
    logger.info(f"[Q1] proxy holdout vs test: mean KS = {summary['mean_ks_proxy_vs_test']:.4f}")
    logger.info(f"[Q1] remaining train vs test: mean KS = {summary['mean_ks_remaining_vs_test']:.4f}")
    logger.info(f"[Q1] proxy 개선량 (vs all train): {summary['proxy_improvement_vs_all']:+.4f}")
    logger.info(f"[Q1] 판정: {verdict}")

    return summary, df


# ──────────────────────────────────────────────────────────
# Q2: proxy holdout vs test Adversarial AUC
# ──────────────────────────────────────────────────────────
def compute_adversarial_auc(train_df, test_df, proxy_set, feature_cols, seed=42):
    """
    세 비교를 수행한다:
      (a) 전체 train vs test → 현재 파이프라인의 기준값
      (b) proxy holdout vs test → 새 전략의 검증 도메인
      (c) remaining train vs test → 나머지 학습 데이터

    AUC 기준:
      < 0.60 → 구분 불가 (test와 동일 분포)
      0.60–0.75 → 중간 분포 차이
      > 0.75 → 심각한 분포 차이 (현재 파이프라인 상태)
    """
    logger.info("[Q2] Adversarial AUC 계산 중...")

    proxy_df   = train_df[train_df["scenario_id"].isin(proxy_set)]
    remain_df  = train_df[~train_df["scenario_id"].isin(proxy_set)]

    safe_cols = [c for c in feature_cols if c in test_df.columns]

    def _adv_auc(df_a, df_b, label, n_sample=5000):
        n_a = min(n_sample, len(df_a))
        n_b = min(n_sample, len(df_b))
        X_a = df_a[safe_cols].sample(n_a, random_state=seed).fillna(-999).values
        X_b = df_b[safe_cols].sample(n_b, random_state=seed).fillna(-999).values
        X   = np.vstack([X_a, X_b])
        y   = np.hstack([np.zeros(len(X_a)), np.ones(len(X_b))])

        skf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            clf = LGBMClassifier(n_estimators=100, max_depth=3,
                                 learning_rate=0.1, random_state=seed, verbose=-1, n_jobs=-1)
            clf.fit(X[tr_idx], y[tr_idx])
            preds = clf.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], preds))

        avg = float(np.mean(aucs))
        status = "INDISTINCT" if avg < 0.60 else ("MODERATE" if avg < 0.75 else "SEVERE_SHIFT")
        logger.info(f"[Q2] {label}: AUC = {avg:.4f} [{status}]")
        return avg, status

    auc_all,   status_all   = _adv_auc(train_df, test_df, "전체 train vs test")
    auc_proxy, status_proxy = _adv_auc(proxy_df,  test_df, "proxy holdout vs test")
    auc_remain,status_remain= _adv_auc(remain_df, test_df, "remaining train vs test")

    improvement = auc_all - auc_proxy

    if status_proxy == "INDISTINCT":
        verdict = "PROXY_EXCELLENT: proxy holdout이 test와 구분 불가 수준 → CV ≈ LB 기대"
    elif status_proxy == "MODERATE":
        verdict = "PROXY_USEFUL: 중간 수준 drift. proxy MAE는 LB의 방향성 지표로 유효."
    else:
        verdict = ("PROXY_INSUFFICIENT: proxy holdout도 여전히 test와 심각한 분포 차이. "
                   "어떤 훈련 서브셋도 test를 근사할 수 없음 → 데이터 수집 단계 문제.")

    result = {
        "auc_all_train_vs_test":    round(auc_all, 4),
        "auc_proxy_vs_test":        round(auc_proxy, 4),
        "auc_remaining_vs_test":    round(auc_remain, 4),
        "status_all":               status_all,
        "status_proxy":             status_proxy,
        "status_remaining":         status_remain,
        "auc_improvement_vs_all":   round(improvement, 4),
        "verdict":                  verdict,
    }
    logger.info(f"[Q2] AUC 개선량 (전체→proxy): {improvement:+.4f}")
    logger.info(f"[Q2] 판정: {verdict}")
    return result


# ──────────────────────────────────────────────────────────
# Q3: proxy holdout MAE vs LB MAE correlation
# ──────────────────────────────────────────────────────────
def compute_proxy_lb_correlation(run_registry, proxy_mae_map, lb_mae_map):
    """
    여러 실행(run)에 걸쳐 proxy holdout MAE와 LB MAE 간 상관계수를 계산한다.

    [입력]
    proxy_mae_map: {run_id: proxy_holdout_mae}  — 이 스크립트가 각 run에 대해 계산
    lb_mae_map:    {run_id: lb_mae}             — 사용자가 리더보드에서 확인한 값

    [비교 기준: 현재 OOF MAE와의 상관]
    현재 CV-LB 상관 ≈ 0.05 (near-zero) 로 보고됨.
    proxy holdout MAE가 이를 초과하면 전략이 유효하다.

    [단일 run인 경우]
    상관계수 계산에는 최소 3개 데이터 포인트가 필요하다.
    단일 run이면 절대값 비교(proxy MAE vs LB MAE 차이)만 제공한다.
    """
    logger.info("[Q3] proxy holdout MAE vs LB MAE 상관 분석...")

    # 공통 run_id만 사용
    common_runs = [r for r in proxy_mae_map if r in lb_mae_map]

    if len(common_runs) < 2:
        logger.warning(f"[Q3] 공통 run이 {len(common_runs)}개뿐 — 상관계수 계산 불가.")
        if len(common_runs) == 1:
            rid = common_runs[0]
            proxy_mae = proxy_mae_map[rid]
            lb_mae    = lb_mae_map[rid]
            abs_gap   = abs(proxy_mae - lb_mae)
            pct_gap   = abs_gap / (lb_mae + 1e-9) * 100

            # OOF MAE도 함께 비교
            oof_mae   = run_registry.get(rid, {}).get("oof_mae", None)
            oof_gap   = abs(oof_mae - lb_mae) if oof_mae else None

            result = {
                "n_runs_used": 1,
                "run_id": rid,
                "proxy_holdout_mae": round(proxy_mae, 4),
                "lb_mae": round(lb_mae, 4),
                "abs_gap_proxy_vs_lb": round(abs_gap, 4),
                "pct_gap_proxy_vs_lb": round(pct_gap, 2),
                "oof_mae": round(oof_mae, 4) if oof_mae else None,
                "abs_gap_oof_vs_lb": round(oof_gap, 4) if oof_gap else None,
                "verdict": (
                    "PROXY_CLOSER_TO_LB" if (oof_gap and abs_gap < oof_gap)
                    else "OOF_CLOSER_TO_LB" if (oof_gap and abs_gap >= oof_gap)
                    else "ONLY_PROXY_AVAILABLE"
                ),
                "note": "상관계수 계산에는 최소 2개 run의 LB MAE가 필요합니다."
            }
        else:
            result = {
                "n_runs_used": 0,
                "note": "LB MAE 입력 없음. --lb-maes 파라미터로 제공하세요.",
                "verdict": "NO_LB_DATA"
            }
        return result

    proxy_maes = [proxy_mae_map[r] for r in common_runs]
    lb_maes    = [lb_mae_map[r]    for r in common_runs]
    oof_maes   = [run_registry.get(r, {}).get("oof_mae", None) for r in common_runs]

    # 상관계수
    r_proxy, p_proxy = pearsonr(proxy_maes, lb_maes)
    rho_proxy, p_rho = spearmanr(proxy_maes, lb_maes)

    r_oof, p_oof = (pearsonr([m for m in oof_maes if m is not None],
                              [lb_maes[i] for i, m in enumerate(oof_maes) if m is not None])
                    if any(m is not None for m in oof_maes) else (None, None))

    result = {
        "n_runs_used": len(common_runs),
        "run_ids": common_runs,
        "proxy_maes": [round(m, 4) for m in proxy_maes],
        "lb_maes":    [round(m, 4) for m in lb_maes],
        "pearson_r_proxy_vs_lb":  round(r_proxy, 4),
        "p_value_pearson":        round(p_proxy, 4),
        "spearman_rho_proxy_vs_lb": round(rho_proxy, 4),
        "p_value_spearman":       round(p_rho, 4),
        "pearson_r_oof_vs_lb":    round(r_oof, 4) if r_oof else None,
        "improvement_vs_oof_r":   round(r_proxy - r_oof, 4) if r_oof else None,
        "verdict": (
            "PROXY_SUPERIOR" if (r_oof and r_proxy > r_oof + 0.1)
            else "PROXY_MARGINAL" if (r_oof and r_proxy > r_oof)
            else "PROXY_NO_IMPROVEMENT" if r_oof
            else "OOF_CORRELATION_UNAVAILABLE"
        ),
    }

    logger.info(f"[Q3] Proxy MAE vs LB MAE Pearson r = {r_proxy:.4f} (p={p_proxy:.4f})")
    logger.info(f"[Q3] Proxy MAE vs LB MAE Spearman ρ = {rho_proxy:.4f} (p={p_rho:.4f})")
    if r_oof:
        logger.info(f"[Q3] OOF MAE vs LB MAE Pearson r = {r_oof:.4f}")
        logger.info(f"[Q3] 개선량 (proxy - oof): {r_proxy - r_oof:+.4f}")
    return result


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",     type=str, default="run_20260430_120906",
                        help="분석할 run ID")
    parser.add_argument("--holdout-pct",type=float, default=0.20,
                        help="proxy holdout 시나리오 비율 (기본 0.20)")
    parser.add_argument("--lb-maes",   type=str, default=None,
                        help="LB MAE 값들 (쉼표 구분, run-id 순서와 일치). "
                             "예: '9.44,9.12,8.87'")
    parser.add_argument("--lb-run-ids",type=str, default=None,
                        help="LB MAE에 대응하는 run_id 목록 (쉼표 구분). "
                             "없으면 --run-id 단독 사용.")
    args = parser.parse_args()

    # ── 경로 설정
    processed_dir = f"outputs/{args.run_id}/processed"
    log_dir       = f"logs/{args.run_id}"
    os.makedirs(log_dir, exist_ok=True)

    # ── 데이터 로드
    logger.info(f"[LOAD] {processed_dir} 에서 데이터 로드 중...")
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    test_df  = pd.read_pickle(f"{processed_dir}/test_base.pkl")
    y_train  = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    oof_preds= np.load(f"outputs/{args.run_id}/predictions/oof_stable.npy", allow_pickle=True)

    logger.info(f"[LOAD] train: {train_df.shape}, test: {test_df.shape}")

    # ── 피처 컬럼 결정 (ID 컬럼 및 타겟 제외, 수치형만)
    id_cols    = ["ID", "scenario_id", "layout_id", "avg_delay_minutes_next_30m"]
    feature_cols = [
        c for c in train_df.columns
        if c not in id_cols and c in test_df.columns
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    logger.info(f"[LOAD] 사용 피처 수: {len(feature_cols)}")

    # ── STEP A: proxy 시나리오 선택
    proxy_set, score_df = identify_proxy_scenarios(
        train_df, test_df, feature_cols, holdout_pct=args.holdout_pct
    )

    # ── Q1: KS 비교
    q1_result, ks_df = compute_ks_comparison(train_df, test_df, proxy_set, feature_cols)

    # ── Q2: Adversarial AUC
    q2_result = compute_adversarial_auc(train_df, test_df, proxy_set, feature_cols)

    # ── proxy holdout MAE 계산 (현재 run의 OOF를 활용)
    proxy_idx = train_df[train_df["scenario_id"].isin(proxy_set)].index
    # OOF 인덱스 정렬: train_df.index 기준
    train_index_arr = train_df.index.values
    proxy_mask = np.isin(train_index_arr, proxy_idx)

    if proxy_mask.sum() > 0 and len(oof_preds) == len(train_df):
        proxy_mae = float(mean_absolute_error(y_train[proxy_mask], oof_preds[proxy_mask]))
        remain_mae= float(mean_absolute_error(y_train[~proxy_mask], oof_preds[~proxy_mask]))
        overall_mae = float(mean_absolute_error(y_train, oof_preds))
        logger.info(f"[MAE] 전체 OOF MAE:       {overall_mae:.4f}")
        logger.info(f"[MAE] proxy holdout MAE:  {proxy_mae:.4f}")
        logger.info(f"[MAE] remaining train MAE: {remain_mae:.4f}")
    else:
        proxy_mae = remain_mae = overall_mae = None
        logger.warning("[MAE] OOF shape 불일치 또는 proxy 없음 — MAE 계산 스킵")

    # ── Q3: LB MAE 상관 분석
    # 레지스트리에서 OOF MAE 목록 수집
    registry_path = "metadata/experiment_registry.json"
    run_oof_registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, encoding="utf-8") as f:
            reg = json.load(f)
        for run in reg.get("runs", []):
            rid = run.get("run_id")
            m   = run.get("metrics", {})
            oof_m = m.get("mae") or run.get("mean_mae")
            if rid and oof_m:
                run_oof_registry[rid] = {"oof_mae": oof_m}

    proxy_mae_map = {}
    lb_mae_map    = {}

    if proxy_mae:
        proxy_mae_map[args.run_id] = proxy_mae

    if args.lb_maes:
        lb_values  = [float(v) for v in args.lb_maes.split(",")]
        if args.lb_run_ids:
            lb_run_ids = args.lb_run_ids.split(",")
        else:
            lb_run_ids = [args.run_id] * len(lb_values)
        for rid, val in zip(lb_run_ids, lb_values):
            lb_mae_map[rid] = val
            # proxy_mae_map에도 없으면 추가 (다른 run의 경우 None 처리)
            if rid not in proxy_mae_map:
                proxy_mae_map[rid] = None

    q3_result = compute_proxy_lb_correlation(run_oof_registry, proxy_mae_map, lb_mae_map)

    # ── 결과 저장
    full_result = {
        "run_id": args.run_id,
        "holdout_pct": args.holdout_pct,
        "n_proxy_scenarios": len(proxy_set),
        "n_total_scenarios": train_df["scenario_id"].nunique(),
        "n_features_used": len(feature_cols),
        "proxy_holdout_mae":  round(proxy_mae,   4) if proxy_mae  else None,
        "remaining_train_mae":round(remain_mae,  4) if remain_mae else None,
        "overall_oof_mae":    round(overall_mae, 4) if overall_mae else None,
        "Q1_ks_comparison":  q1_result,
        "Q2_adversarial_auc": q2_result,
        "Q3_lb_correlation":  q3_result,
    }

    out_path = f"{log_dir}/cv_parity_audit_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)

    # ── 최종 요약 출력
    print("\n" + "=" * 65)
    print(f"  [CV PARITY AUDIT 결과 요약]  run: {args.run_id}")
    print("=" * 65)
    print(f"\n[Q1] KS Distance (test 거리, 낮을수록 test에 가까움)")
    print(f"  전체 train vs test  : {q1_result['mean_ks_all_train_vs_test']:.4f}")
    print(f"  proxy holdout vs test: {q1_result['mean_ks_proxy_vs_test']:.4f}")
    print(f"  remaining vs test   : {q1_result['mean_ks_remaining_vs_test']:.4f}")
    print(f"  proxy 개선량        : {q1_result['proxy_improvement_vs_all']:+.4f}")
    print(f"  판정: {q1_result['verdict']}")

    print(f"\n[Q2] Adversarial AUC (낮을수록 test와 동일 분포)")
    print(f"  전체 train vs test  : {q2_result['auc_all_train_vs_test']:.4f}  [{q2_result['status_all']}]")
    print(f"  proxy holdout vs test: {q2_result['auc_proxy_vs_test']:.4f}  [{q2_result['status_proxy']}]")
    print(f"  remaining vs test   : {q2_result['auc_remaining_vs_test']:.4f}  [{q2_result['status_remaining']}]")
    print(f"  AUC 개선량          : {q2_result['auc_improvement_vs_all']:+.4f}")
    print(f"  판정: {q2_result['verdict']}")

    print(f"\n[MAE 분포]")
    print(f"  전체 OOF MAE        : {overall_mae:.4f}" if overall_mae else "  전체 OOF MAE: N/A")
    print(f"  proxy holdout MAE   : {proxy_mae:.4f}" if proxy_mae else "  proxy holdout MAE: N/A")
    print(f"  remaining train MAE : {remain_mae:.4f}" if remain_mae else "  remaining train MAE: N/A")

    print(f"\n[Q3] proxy holdout MAE vs LB MAE 상관")
    if q3_result.get("pearson_r_proxy_vs_lb") is not None:
        print(f"  Pearson r  : {q3_result['pearson_r_proxy_vs_lb']:.4f}")
        print(f"  Spearman ρ : {q3_result['spearman_rho_proxy_vs_lb']:.4f}")
        print(f"  OOF vs LB r: {q3_result.get('pearson_r_oof_vs_lb', 'N/A')}")
    else:
        print(f"  {q3_result.get('note', 'LB MAE 데이터 필요')}")
    print(f"  판정: {q3_result['verdict']}")

    print(f"\n결과 저장됨 → {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
