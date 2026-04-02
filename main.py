import pandas as pd
from src.config import Config
from src.utils import seed_everything, get_logger
from src.data_loader import load_data, get_features
from src.trainer import Trainer

def main():
    """Main execution function for the Smart Warehouse delay prediction baseline."""
    logger = get_logger()
    logger.info("── Starting Baseline Pipeline ──")
    
    # 1. Initialization
    seed_everything(Config.SEED)
    
    # 2. Loading Data
    logger.info("Loading Data...")
    try:
        train, test = load_data()
    except FileNotFoundError:
        logger.error(f"Data directory not found. Please ensure `data/train.csv` and `data/test.csv` are present.")
        return

    # 3. Feature Preparation
    feature_cols = get_features(train, test)
    logger.info(f"Number of Features identified: {len(feature_cols)}")

    # 4. Training
    logger.info("Beginning 5-Fold Training...")
    trainer = Trainer(train, test, feature_cols)
    test_preds = trainer.train_kfolds()

    # 5. Exporting Results
    logger.info("Generating Submission File...")
    submission = pd.DataFrame({'ID': test['ID'], Config.TARGET: test_preds})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission successfully saved to: {Config.SUBMISSION_PATH}")
    logger.info("── Pipeline Execution Finished ──")

if __name__ == "__main__":
    main()
