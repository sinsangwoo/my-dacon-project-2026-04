import os
def patch_main():
    path = 'main.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Phase 3 Guard
    # Context: lgb_mae, _ = trainer.train_kfolds(features, seeds=Config.SEEDS)
    # We need to find the specific one in Phase 3.
    target_3 = 'lgb_mae, _ = trainer.train_kfolds(features, seeds=Config.SEEDS)'
    patch_3 = """if Config.DRY_RUN:
                logger.info("[DRY_RUN] Skipping Base Training (LGBM). Saving mock predictions.")
                lgb_mae = 9.99
                trainer.oof_preds = {"lgb": np.random.rand(len(X_train))}
                trainer.test_preds = {"lgb": np.random.rand(len(X_test))}
            else:
                lgb_mae, _ = trainer.train_kfolds(features, seeds=Config.SEEDS)"""
    
    if target_3 in content:
        content = content.replace(target_3, patch_3)
        print("Patched Phase 3")

    # Phase 4 Guard
    target_4 = 'meta_mae, _ = trainer.train_kfolds(features, seeds=Config.SEEDS[:1])'
    patch_4 = """if Config.DRY_RUN:
                logger.info("[DRY_RUN] Skipping Stacking Meta-Training. Saving mock predictions.")
                trainer.test_preds = {"meta": np.random.rand(len(X_test))}
                meta_mae = 9.99
            else:
                meta_mae, _ = trainer.train_kfolds(features, seeds=Config.SEEDS[:1])"""
    if target_4 in content:
        content = content.replace(target_4, patch_4)
        print("Patched Phase 4")

    # Phase 6 Guard
    target_6 = '_, cat_mae = trainer.train_kfolds(features, seeds=Config.SEEDS, train_df=X_combined)'
    patch_6 = """if Config.DRY_RUN:
                logger.info("[DRY_RUN] Skipping Final Retrain (CatBoost). Saving mock predictions.")
                cat_mae = 9.99
                trainer.test_preds = {"cat": np.random.rand(len(X_test_reduced))}
                trainer.extreme_analysis = {"cat": {"f1": 0.5, "extreme_mae": 10.0}}
            else:
                _, cat_mae = trainer.train_kfolds(features, seeds=Config.SEEDS, train_df=X_combined)"""
    if target_6 in content:
        content = content.replace(target_6, patch_6)
        print("Patched Phase 6")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("DONE: main.py patched successfully.")

if __name__ == "__main__":
    patch_main()
