# train_cf_model.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)

DATA_PATH = 'data/u.data' # Path to MovieLens 100k u.data file
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'cf_model.pkl')

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file not found at {DATA_PATH}. Please download MovieLens 100k and place u.data in the 'data' directory.")
        return

    # Load data
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(DATA_PATH, reader=reader)

    # We train on the full dataset for prediction purposes in the API
    # Alternatively, use cross-validation if evaluating performance is the goal
    trainset = data.build_full_trainset()

    # Use SVD algorithm
    logging.info("Training SVD model...")
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42) # Example parameters
    algo.fit(trainset)
    logging.info("Training complete.")


    # Ensure the models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the trained model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(algo, f)
    logging.info(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()