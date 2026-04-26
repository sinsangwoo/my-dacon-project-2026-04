# FULL FORENSIC ANALYSIS REPORT
## 1. FILE STRUCTURE

```text
logs\run_20260426_021454\1_data_check.log                2282 bytes  2026-04-26 02:15:29
logs\run_20260426_021454\2.5_drift_audit.log             1068 bytes  2026-04-26 02:16:38
logs\run_20260426_021454\2_build_base.log                9620 bytes  2026-04-26 02:16:17
logs\run_20260426_021454\3_train_raw.log                 2401 bytes  2026-04-26 02:17:14
logs\run_20260426_021454\5_train_leakage_free.log       36603 bytes  2026-04-26 03:01:34
logs\run_20260426_021454\7_inference.log                 6315 bytes  2026-04-26 03:09:10
logs\run_20260426_021454\8_submission.log                 898 bytes  2026-04-26 03:09:29
logs\run_20260426_021454\9_intelligence.log              1264 bytes  2026-04-26 03:09:45
logs\run_20260426_021454\intelligence_summary.txt         725 bytes  2026-04-26 03:09:45
logs\run_20260426_021454\pipeline.log                    2044 bytes  2026-04-26 03:09:45
logs\run_20260426_021454\validation_report.json          1205 bytes  2026-04-26 03:01:34
outputs\run_20260426_021454\logs\1_data_check.log        2210 bytes  2026-04-26 02:15:29
outputs\run_20260426_021454\logs\2.5_drift_audit.log        484 bytes  2026-04-26 02:16:38
outputs\run_20260426_021454\logs\2_build_base.log         480 bytes  2026-04-26 02:16:17
outputs\run_20260426_021454\logs\3_train_raw.log          372 bytes  2026-04-26 02:17:14
outputs\run_20260426_021454\logs\5_train_leakage_free.log        505 bytes  2026-04-26 03:01:34
outputs\run_20260426_021454\logs\7_inference.log          773 bytes  2026-04-26 03:09:10
outputs\run_20260426_021454\logs\8_submission.log         611 bytes  2026-04-26 03:09:29
outputs\run_20260426_021454\logs\9_intelligence.log        383 bytes  2026-04-26 03:09:45
outputs\run_20260426_021454\logs\FILE_IO.log             6862 bytes  2026-04-26 03:09:29
outputs\run_20260426_021454\models\lgbm\model_fold_0.pkl     353375 bytes  2026-04-26 02:26:33
outputs\run_20260426_021454\models\lgbm\model_fold_1.pkl     356793 bytes  2026-04-26 02:35:26
outputs\run_20260426_021454\models\lgbm\model_fold_2.pkl     356644 bytes  2026-04-26 02:44:02
outputs\run_20260426_021454\models\lgbm\model_fold_3.pkl     356539 bytes  2026-04-26 02:52:28
outputs\run_20260426_021454\models\lgbm\model_fold_4.pkl     356496 bytes  2026-04-26 03:01:09
outputs\run_20260426_021454\models\reconstructors\features_fold_0.pkl       5128 bytes  2026-04-26 02:26:08
outputs\run_20260426_021454\models\reconstructors\features_fold_1.pkl       5128 bytes  2026-04-26 02:35:03
outputs\run_20260426_021454\models\reconstructors\features_fold_2.pkl       5128 bytes  2026-04-26 02:43:39
outputs\run_20260426_021454\models\reconstructors\features_fold_3.pkl       5128 bytes  2026-04-26 02:51:56
outputs\run_20260426_021454\models\reconstructors\features_fold_4.pkl       5128 bytes  2026-04-26 03:00:39
outputs\run_20260426_021454\models\reconstructors\norm_scaler_fold_0.pkl       7755 bytes  2026-04-26 02:26:08
outputs\run_20260426_021454\models\reconstructors\norm_scaler_fold_1.pkl       7755 bytes  2026-04-26 02:35:03
outputs\run_20260426_021454\models\reconstructors\norm_scaler_fold_2.pkl       7755 bytes  2026-04-26 02:43:39
outputs\run_20260426_021454\models\reconstructors\norm_scaler_fold_3.pkl       7755 bytes  2026-04-26 02:51:56
outputs\run_20260426_021454\models\reconstructors\norm_scaler_fold_4.pkl       7755 bytes  2026-04-26 03:00:39
outputs\run_20260426_021454\models\reconstructors\recon_fold_0.pkl   64010336 bytes  2026-04-26 02:26:08
outputs\run_20260426_021454\models\reconstructors\recon_fold_1.pkl   64010336 bytes  2026-04-26 02:35:03
outputs\run_20260426_021454\models\reconstructors\recon_fold_2.pkl   64010336 bytes  2026-04-26 02:43:39
outputs\run_20260426_021454\models\reconstructors\recon_fold_3.pkl   64010336 bytes  2026-04-26 02:51:56
outputs\run_20260426_021454\models\reconstructors\recon_fold_4.pkl   64010336 bytes  2026-04-26 03:00:39
outputs\run_20260426_021454\models\reconstructors\scaler_fold_0.pkl      14765 bytes  2026-04-26 02:26:08
outputs\run_20260426_021454\models\reconstructors\scaler_fold_1.pkl      14765 bytes  2026-04-26 02:35:03
outputs\run_20260426_021454\models\reconstructors\scaler_fold_2.pkl      14765 bytes  2026-04-26 02:43:39
outputs\run_20260426_021454\models\reconstructors\scaler_fold_3.pkl      14765 bytes  2026-04-26 02:51:56
outputs\run_20260426_021454\models\reconstructors\scaler_fold_4.pkl      14765 bytes  2026-04-26 03:00:39
outputs\run_20260426_021454\predictions\final_submission.npy     400128 bytes  2026-04-26 03:09:10
outputs\run_20260426_021454\predictions\oof_stable.npy    2000128 bytes  2026-04-26 03:01:34
outputs\run_20260426_021454\predictions\test_stable.npy     400128 bytes  2026-04-26 03:01:34
outputs\run_20260426_021454\processed\oof_raw.npy     2000128 bytes  2026-04-26 02:17:14
outputs\run_20260426_021454\processed\pruning_manifest.json       8292 bytes  2026-04-26 02:16:15
outputs\run_20260426_021454\processed\residuals_raw.npy    2000128 bytes  2026-04-26 02:17:14
outputs\run_20260426_021454\processed\scenario_id.npy    1294059 bytes  2026-04-26 02:16:17
outputs\run_20260426_021454\processed\stability_manifest.json       4603 bytes  2026-04-26 02:16:38
outputs\run_20260426_021454\processed\test_base.pkl   42975490 bytes  2026-04-26 02:16:17
outputs\run_20260426_021454\processed\train_base.pkl  215095500 bytes  2026-04-26 02:16:16
outputs\run_20260426_021454\processed\train_stats.json         89 bytes  2026-04-26 02:16:17
outputs\run_20260426_021454\processed\y_train.npy     1000128 bytes  2026-04-26 02:16:17
outputs\run_20260426_021454\submission\submission.csv     932580 bytes  2026-04-26 03:09:29
outputs\run_20260426_021454\summary\config_snapshot.json       3673 bytes  2026-04-26 03:09:45
outputs\run_20260426_021454\summary\feature_drop_registry.json      65868 bytes  2026-04-26 02:16:15
outputs\run_20260426_021454\summary\distribution\drift_audit_raw.csv      26875 bytes  2026-04-26 02:16:38
[MISSING] c:\Github_public\my_dacon_project\my-dacon-project-2026-04\metadata\run_20260426_021454
src\audit_pipeline.py                                    7842 bytes  2026-04-25 12:53:53
src\config.py                                            5491 bytes  2026-04-25 21:13:49
src\data_loader.py                                      72014 bytes  2026-04-26 02:13:00
src\distribution.py                                      6492 bytes  2026-04-25 17:25:04
src\explosion_inference.py                               1708 bytes  2026-04-23 00:32:39
src\feature_registry.py                                 12379 bytes  2026-04-26 00:54:47
src\forensic_logger.py                                  15986 bytes  2026-04-18 13:06:51
src\intelligence.py                                      4595 bytes  2026-04-23 11:53:28
src\schema.py                                            6511 bytes  2026-04-26 00:55:12
src\trainer.py                                          21817 bytes  2026-04-26 01:24:45
src\utils.py                                            39548 bytes  2026-04-25 21:09:36
src\__pycache__\audit_pipeline.cpython-313.pyc           8405 bytes  2026-04-25 01:48:27
src\__pycache__\config.cpython-313.pyc                   4074 bytes  2026-04-25 21:13:57
src\__pycache__\data_loader.cpython-313.pyc             64882 bytes  2026-04-26 02:15:09
src\__pycache__\distribution.cpython-313.pyc             8358 bytes  2026-04-25 17:40:24
src\__pycache__\explosion_inference.cpython-313.pyc       2430 bytes  2026-04-23 00:33:31
src\__pycache__\feature_registry.cpython-313.pyc        16167 bytes  2026-04-26 01:28:43
src\__pycache__\forensic_logger.cpython-313.pyc         20591 bytes  2026-04-18 13:07:01
src\__pycache__\intelligence.cpython-313.pyc             7155 bytes  2026-04-23 11:53:55
src\__pycache__\schema.cpython-313.pyc                   3210 bytes  2026-04-26 01:28:14
src\__pycache__\trainer.cpython-313.pyc                 22085 bytes  2026-04-26 01:28:43
src\__pycache__\utils.cpython-313.pyc                   55911 bytes  2026-04-25 21:10:51
main.py                                                 14816 bytes  2026-04-25 17:39:11
pipeline.sh                                              4777 bytes  2026-04-26 01:58:58
```

## 2. RAW LOG DUMP

### 5_train_leakage_free.log
```text
--- OTHER METRICS ---
2026-04-26 02:17:32,935 - DriftShieldScaler - INFO - [DRIFT_SHIELD] Fitting on 142 features...
2026-04-26 02:17:33,362 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] raw explained variance: 0.7992 < 0.8. Switching to Mode B for this view.
2026-04-26 02:17:33,746 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] rank explained variance: 0.7838 < 0.8. Switching to Mode B for this view.
2026-04-26 02:17:33,899 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] local explained variance: 0.7992 < 0.8. Switching to Mode B for this view.
2026-04-26 02:17:41,656 - DriftShieldScaler - INFO - [DRIFT_SHIELD] Fitting on 142 features...
2026-04-26 02:17:44,510 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] raw explained variance: 0.7987 < 0.8. Switching to Mode B for this view.
2026-04-26 02:17:45,870 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] rank explained variance: 0.7844 < 0.8. Switching to Mode B for this view.
2026-04-26 02:17:46,518 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] local explained variance: 0.7987 < 0.8. Switching to Mode B for this view.
2026-04-26 02:26:34,940 - DriftShieldScaler - INFO - [DRIFT_SHIELD] Fitting on 142 features...
2026-04-26 02:26:37,864 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] raw explained variance: 0.7996 < 0.8. Switching to Mode B for this view.
2026-04-26 02:26:39,088 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] rank explained variance: 0.7845 < 0.8. Switching to Mode B for this view.
2026-04-26 02:26:39,730 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] local explained variance: 0.7996 < 0.8. Switching to Mode B for this view.
2026-04-26 02:35:27,583 - DriftShieldScaler - INFO - [DRIFT_SHIELD] Fitting on 142 features...
2026-04-26 02:35:30,291 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] raw explained variance: 0.7989 < 0.8. Switching to Mode B for this view.
2026-04-26 02:35:32,008 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] rank explained variance: 0.7845 < 0.8. Switching to Mode B for this view.
2026-04-26 02:35:32,808 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] local explained variance: 0.7989 < 0.8. Switching to Mode B for this view.
2026-04-26 02:44:03,490 - DriftShieldScaler - INFO - [DRIFT_SHIELD] Fitting on 142 features...
2026-04-26 02:44:07,520 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] raw explained variance: 0.7980 < 0.8. Switching to Mode B for this view.
2026-04-26 02:44:09,247 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] rank explained variance: 0.7844 < 0.8. Switching to Mode B for this view.
2026-04-26 02:44:10,010 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] local explained variance: 0.7980 < 0.8. Switching to Mode B for this view.
2026-04-26 02:52:28,966 - DriftShieldScaler - INFO - [DRIFT_SHIELD] Fitting on 142 features...
2026-04-26 02:52:31,234 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] raw explained variance: 0.7992 < 0.8. Switching to Mode B for this view.
2026-04-26 02:52:32,476 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] rank explained variance: 0.7844 < 0.8. Switching to Mode B for this view.
2026-04-26 02:52:33,072 - src.data_loader - WARNING - [PCA_LOW_VARIANCE] local explained variance: 0.7992 < 0.8. Switching to Mode B for this view.

--- VARIANCE COMPRESSION BY FEATURE ---
Feature: order_inflow_15m_rate_1
2026-04-26 02:17:33,035 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3930 !!!
2026-04-26 02:17:42,452 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3740 !!!
2026-04-26 02:23:20,383 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3753 !!!
2026-04-26 02:24:44,747 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3359 !!!
2026-04-26 02:26:36,045 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3819 !!!
2026-04-26 02:32:22,075 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3909 !!!
2026-04-26 02:33:44,953 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3444 !!!
2026-04-26 02:35:28,222 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3760 !!!
2026-04-26 02:40:59,774 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3724 !!!
2026-04-26 02:42:19,905 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3368 !!!
2026-04-26 02:44:04,378 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3694 !!!
2026-04-26 02:49:26,060 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3748 !!!
2026-04-26 02:50:44,153 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3325 !!!
2026-04-26 02:52:29,568 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3762 !!!
2026-04-26 02:58:06,213 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3640 !!!
2026-04-26 02:59:20,778 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3355 !!!

Feature: unique_sku_15m_rate_1
2026-04-26 02:17:33,036 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3629 !!!
2026-04-26 02:17:42,458 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3866 !!!
2026-04-26 02:23:20,386 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3910 !!!
2026-04-26 02:24:44,750 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3516 !!!
2026-04-26 02:26:36,050 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3785 !!!
2026-04-26 02:32:22,078 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3778 !!!
2026-04-26 02:33:44,955 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3433 !!!
2026-04-26 02:35:28,228 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3856 !!!
2026-04-26 02:40:59,777 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3785 !!!
2026-04-26 02:42:19,907 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3486 !!!
2026-04-26 02:44:04,384 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3789 !!!
2026-04-26 02:49:26,062 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3838 !!!
2026-04-26 02:50:44,156 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3447 !!!
2026-04-26 02:52:29,572 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3802 !!!
2026-04-26 02:58:06,215 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3786 !!!
2026-04-26 02:59:20,780 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3447 !!!

Feature: task_reassign_15m_diff_1
2026-04-26 02:17:33,042 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5685 !!!
2026-04-26 02:17:42,500 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5477 !!!
2026-04-26 02:23:20,398 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5435 !!!
2026-04-26 02:26:36,091 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5494 !!!
2026-04-26 02:32:22,101 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5565 !!!
2026-04-26 02:35:28,292 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5470 !!!
2026-04-26 02:40:59,788 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5554 !!!
2026-04-26 02:44:04,424 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5550 !!!
2026-04-26 02:49:26,078 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5504 !!!
2026-04-26 02:52:29,618 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5503 !!!
2026-04-26 02:58:06,227 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_diff_1 ratio=0.5434 !!!

Feature: task_reassign_15m_rate_1
2026-04-26 02:17:33,043 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4280 !!!
2026-04-26 02:17:42,502 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4460 !!!
2026-04-26 02:23:20,399 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4508 !!!
2026-04-26 02:24:44,770 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5252 !!!
2026-04-26 02:26:36,093 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4540 !!!
2026-04-26 02:32:22,102 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4594 !!!
2026-04-26 02:33:44,969 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5347 !!!
2026-04-26 02:35:28,294 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4447 !!!
2026-04-26 02:40:59,789 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4417 !!!
2026-04-26 02:42:19,919 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5218 !!!
2026-04-26 02:44:04,426 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4564 !!!
2026-04-26 02:49:26,079 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4551 !!!
2026-04-26 02:50:44,170 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5360 !!!
2026-04-26 02:52:29,620 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4524 !!!
2026-04-26 02:58:06,228 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.4465 !!!
2026-04-26 02:59:20,793 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5301 !!!

Feature: charge_queue_length_diff_1
2026-04-26 02:17:33,046 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5639 !!!
2026-04-26 02:17:42,516 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5667 !!!
2026-04-26 02:23:20,403 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5665 !!!
2026-04-26 02:26:36,107 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5663 !!!
2026-04-26 02:32:22,107 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5591 !!!
2026-04-26 02:35:28,310 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5688 !!!
2026-04-26 02:40:59,793 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5646 !!!
2026-04-26 02:44:04,442 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5703 !!!
2026-04-26 02:49:26,084 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5764 !!!
2026-04-26 02:52:29,634 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5648 !!!
2026-04-26 02:58:06,233 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] charge_queue_length_diff_1 ratio=0.5703 !!!

Feature: replenishment_overlap_rate_1
2026-04-26 02:17:33,050 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1915 !!!
2026-04-26 02:17:42,546 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1655 !!!
2026-04-26 02:23:20,410 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1635 !!!
2026-04-26 02:24:44,784 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1732 !!!
2026-04-26 02:26:36,131 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1549 !!!
2026-04-26 02:32:22,116 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1569 !!!
2026-04-26 02:33:44,982 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1629 !!!
2026-04-26 02:35:28,342 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1547 !!!
2026-04-26 02:40:59,802 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1535 !!!
2026-04-26 02:42:19,931 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1620 !!!
2026-04-26 02:44:04,471 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1474 !!!
2026-04-26 02:49:26,093 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1479 !!!
2026-04-26 02:50:44,182 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1548 !!!
2026-04-26 02:52:29,660 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1540 !!!
2026-04-26 02:58:06,241 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1544 !!!
2026-04-26 02:59:20,806 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1617 !!!

Feature: pack_utilization_rate_1
2026-04-26 02:17:33,051 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4586 !!!
2026-04-26 02:17:42,553 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4721 !!!
2026-04-26 02:23:20,412 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4747 !!!
2026-04-26 02:24:44,787 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4985 !!!
2026-04-26 02:26:36,136 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4792 !!!
2026-04-26 02:32:22,118 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4832 !!!
2026-04-26 02:33:44,984 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5061 !!!
2026-04-26 02:35:28,350 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4758 !!!
2026-04-26 02:40:59,805 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4757 !!!
2026-04-26 02:42:19,933 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5016 !!!
2026-04-26 02:44:04,478 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4746 !!!
2026-04-26 02:49:26,096 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4747 !!!
2026-04-26 02:50:44,185 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5004 !!!
2026-04-26 02:52:29,665 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4830 !!!
2026-04-26 02:58:06,243 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4763 !!!
2026-04-26 02:59:20,808 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5079 !!!

Feature: air_quality_idx_rate_1
2026-04-26 02:17:33,057 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1317 !!!
2026-04-26 02:17:42,579 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1614 !!!
2026-04-26 02:23:20,421 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1605 !!!
2026-04-26 02:24:44,796 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1662 !!!
2026-04-26 02:26:36,161 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1679 !!!
2026-04-26 02:32:22,128 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1663 !!!
2026-04-26 02:33:44,992 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1728 !!!
2026-04-26 02:35:28,379 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1645 !!!
2026-04-26 02:40:59,812 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1708 !!!
2026-04-26 02:42:19,940 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1709 !!!
2026-04-26 02:44:04,508 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1654 !!!
2026-04-26 02:49:26,106 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1612 !!!
2026-04-26 02:50:44,194 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1697 !!!
2026-04-26 02:52:29,696 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1649 !!!
2026-04-26 02:58:06,250 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1653 !!!
2026-04-26 02:59:20,816 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1701 !!!

Feature: robot_idle_rate_1
2026-04-26 02:17:42,487 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_idle_rate_1 ratio=0.5899 !!!
2026-04-26 02:23:20,394 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_idle_rate_1 ratio=0.5912 !!!
2026-04-26 02:26:36,078 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_idle_rate_1 ratio=0.5962 !!!
2026-04-26 02:32:22,096 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_idle_rate_1 ratio=0.5928 !!!
2026-04-26 02:49:26,072 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_idle_rate_1 ratio=0.5987 !!!

Feature: robot_utilization_rate_1
2026-04-26 02:24:44,766 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5908 !!!
2026-04-26 02:33:44,966 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5950 !!!
2026-04-26 02:42:19,916 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5939 !!!
2026-04-26 02:50:44,166 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5879 !!!
2026-04-26 02:59:20,790 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5884 !!!

Feature: task_reassign_15m
2026-04-26 02:26:36,009 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m ratio=0.5860 !!!
2026-04-26 02:32:22,065 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m ratio=0.5963 !!!
2026-04-26 02:35:28,187 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m ratio=0.5829 !!!
2026-04-26 02:40:59,766 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m ratio=0.5996 !!!
2026-04-26 02:52:29,545 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m ratio=0.5846 !!!
2026-04-26 02:58:06,204 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m ratio=0.5748 !!!

```

### 7_inference.log
```text
--- OTHER METRICS ---


--- VARIANCE COMPRESSION BY FEATURE ---
Feature: order_inflow_15m_rate_1
2026-04-26 03:01:54,267 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3359 !!!
2026-04-26 03:03:26,042 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3444 !!!
2026-04-26 03:04:52,146 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3368 !!!
2026-04-26 03:06:17,563 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3325 !!!
2026-04-26 03:07:43,663 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] order_inflow_15m_rate_1 ratio=0.3355 !!!

Feature: unique_sku_15m_rate_1
2026-04-26 03:01:54,269 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3516 !!!
2026-04-26 03:03:26,044 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3433 !!!
2026-04-26 03:04:52,148 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3486 !!!
2026-04-26 03:06:17,566 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3447 !!!
2026-04-26 03:07:43,666 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] unique_sku_15m_rate_1 ratio=0.3447 !!!

Feature: robot_utilization_rate_1
2026-04-26 03:01:54,277 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5908 !!!
2026-04-26 03:03:26,055 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5950 !!!
2026-04-26 03:04:52,158 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5939 !!!
2026-04-26 03:06:17,578 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5879 !!!
2026-04-26 03:07:43,682 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] robot_utilization_rate_1 ratio=0.5884 !!!

Feature: task_reassign_15m_rate_1
2026-04-26 03:01:54,280 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5252 !!!
2026-04-26 03:03:26,060 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5347 !!!
2026-04-26 03:04:52,161 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5218 !!!
2026-04-26 03:06:17,582 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5360 !!!
2026-04-26 03:07:43,687 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] task_reassign_15m_rate_1 ratio=0.5301 !!!

Feature: replenishment_overlap_rate_1
2026-04-26 03:01:54,289 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1732 !!!
2026-04-26 03:03:26,072 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1629 !!!
2026-04-26 03:04:52,172 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1620 !!!
2026-04-26 03:06:17,595 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1548 !!!
2026-04-26 03:07:43,704 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] replenishment_overlap_rate_1 ratio=0.1617 !!!

Feature: pack_utilization_rate_1
2026-04-26 03:01:54,291 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.4985 !!!
2026-04-26 03:03:26,075 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5061 !!!
2026-04-26 03:04:52,175 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5016 !!!
2026-04-26 03:06:17,598 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5004 !!!
2026-04-26 03:07:43,707 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] pack_utilization_rate_1 ratio=0.5079 !!!

Feature: air_quality_idx_rate_1
2026-04-26 03:01:54,298 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1662 !!!
2026-04-26 03:03:26,083 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1728 !!!
2026-04-26 03:04:52,183 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1709 !!!
2026-04-26 03:06:17,607 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1697 !!!
2026-04-26 03:07:43,716 - DriftShieldScaler - ERROR - !!! [VARIANCE_COMPRESSION] air_quality_idx_rate_1 ratio=0.1701 !!!

```

## 3. CODE SNIPPETS (NO OMIT)

### Variance Computation Trace (src/utils.py)
```python
    def transform(self, df, feature_cols):
        if not self.stats:
            self.logger.warning("[DRIFT_SHIELD] Transform called before fit! Returning raw.")
            return df

        df = df.copy()
        for col in feature_cols:
            if col not in df.columns or col not in self.stats: continue

            s = self.stats[col]
            x = df[col].values.astype(np.float32)

            # [PHASE 6: EXTREME VALUE PRESERVATION]
            # Replace Inf with large finite values
            x = np.nan_to_num(x, nan=s['mean'], posinf=s['p99'], neginf=s['p1'])

            # 1. Clip (Preserve original scale but bound outliers)
            clipped_mask = (x > s['p99']) | (x < s['p1'])
            n_clipped = np.sum(clipped_mask)
            clip_pct = n_clipped / len(x)

            self.clipping_ratios[col] = clip_pct

            if clip_pct > 0.05:
                self.logger.warning(f"[CLIPPING_MONITOR] High clipping: {col}={clip_pct:.2%}")

            # [WHY_THIS_DESIGN] Outlier Clipping
            # Observed Data Behavior: Heavy-tailed distribution in delay-related features.
            # Why P1/P99: Captures 98% of variance while suppressing extreme sensor noise
            #   that can destabilize Gradient Boosting and PCA reconstruction.
            # Sensitivity: P95 is too aggressive (loses real spikes); P99.9 preserves too much noise.
            x = np.clip(x, s['p1'], s['p99'])

            # [PHASE 5: VARIANCE RESTORATION]
            # We NO LONGER normalize here. Normalization is a separate responsibility.
            # We NO LONGER soft-clip (log suppression) here.

            # [PHASE 5: ASSERT REAL VARIANCE]
            std_after = np.std(x) + 1e-9
            # Since we no longer normalize, std_after should be close to std_before (s['std'])
            ratio = std_after / (s['std'] + 1e-9)

            if ratio < 0.6:
                self.logger.error(f"!!! [VARIANCE_COMPRESSION] {col} ratio={ratio:.4f} !!!")

            df[col] = x
        # [AXIS3_FIX] 이중 return 제거. 두 번째 return df는 데드 코드이며,
        # 향후 두 return 사이에 로직 삽입 시 의도치 않은 흐름이 발생할 수 있다.
        return df

    def save(self, path):
```

## 4. NUMERICAL TABLES

### Feature-Level Forensic

```text
Feature: order_inflow_15m_rate_1
Metric      | Raw (Train) | After DriftShield
------------|-------------|------------------
Mean        |     -0.5268 |           -0.3284
Std         |      4.0297 |            1.5124
Variance    |     16.2385 |            2.2874
Min         |   -221.0000 |          -10.8350
Max         |      1.0000 |            0.9258

Calculated Ratio = 1.5124 / 4.0297 = 0.3753
P1 = -10.8350, P99 = 0.9258
----------------------------------------
Feature: replenishment_overlap_rate_1
Metric      | Raw (Train) | After DriftShield
------------|-------------|------------------
Mean        |     -0.8379 |           -0.3134
Std         |     15.1322 |            2.3441
Variance    |    228.9841 |            5.4949
Min         |  -2900.5422 |          -15.6781
Max         |      1.0000 |            1.0000

Calculated Ratio = 2.3441 / 15.1322 = 0.1549
P1 = -15.6781, P99 = 1.0000
----------------------------------------
Feature: robot_utilization_rate_1
Metric      | Raw (Train) | After DriftShield
------------|-------------|------------------
Mean        |     -0.1141 |           -0.0862
Std         |      0.7789 |            0.4879
Variance    |      0.6067 |            0.2381
Min         |    -21.0570 |           -2.4500
Max         |      1.0000 |            0.7272

Calculated Ratio = 0.4879 / 0.7789 = 0.6264
P1 = -2.4500, P99 = 0.7272
----------------------------------------
```

## 5. PIPELINE FLOW TRACE

### Phase 5 (Train)
```python

        # 1. Global Scaling & Initialization
        global_scaler = DriftShieldScaler()
        # [TASK 12 — FOLD STABILITY] Use local RNG for deterministic sampling
        # [CODE_EVIDENCE] Previously: np.random.choice() used global RNG state
        # [FAILURE_MODE_PREVENTED] Non-deterministic feature selection across runs
        rng = np.random.RandomState(Config.SEED)
        sample_idx = rng.choice(n_train, min(n_train, 20000), replace=False)
        sample_df = self.df_train.iloc[sample_idx]
        sample_y = self.y[sample_idx]

        raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in sample_df.columns]
        global_scaler.fit(sample_df, raw_cols)

        from sklearn.preprocessing import StandardScaler
        global_norm_scaler = StandardScaler()
        sample_df_drifted = global_scaler.transform(sample_df, raw_cols)
        global_norm_scaler.fit(sample_df_drifted[raw_cols])

        global_reconstructor = SuperchargedPCAReconstructor(input_dim=len(raw_cols))
        sample_df_scaled = sample_df_drifted.copy()
        sample_df_scaled[raw_cols] = global_norm_scaler.transform(sample_df_drifted[raw_cols])
```

### Phase 7 (Inference)
```python
                with open(f'{Config.MODELS_PATH}/reconstructors/recon_fold_{fold}.pkl', 'rb') as f:
                    reconstructor = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/scaler_fold_{fold}.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/features_fold_{fold}.pkl', 'rb') as f:
                    fold_features = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/norm_scaler_fold_{fold}.pkl', 'rb') as f:
                    norm_scaler = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/lgbm/model_fold_{fold}.pkl', 'rb') as f:
                    model = pickle.load(f)

                # Apply features strictly via pre-computed cache
                from src.data_loader import apply_latent_features
                # [PHASE 2: UNIFIED SCALING] Scale test data before latent population using fold-specific schema
                raw_cols = list(norm_scaler.feature_names_in_)
                test_base_drifted = scaler.transform(test_base, raw_cols)
                test_base_scaled = test_base_drifted.copy()
                test_base_scaled[raw_cols] = norm_scaler.transform(test_base_drifted[raw_cols])

                test_df_full = apply_latent_features(test_base_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
                X_test_f = test_df_full[fold_features].values.astype(np.float32)

```

## 6. CONSISTENCY CHECK RESULT

```text
Best CV MAE: 11.472744941711426
Are same parameters used? (Checking saved scaler stats vs test time)
Found 5 saved DriftShieldScaler pickles.
Fold 0 Scaler param for order_inflow_15m: {'p1': 1.0, 'p99': 310.0, 'mean': 94.62772369384766, 'std': 77.06011199951172}
```

