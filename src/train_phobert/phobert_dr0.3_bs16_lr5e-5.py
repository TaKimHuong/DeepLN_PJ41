import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from models.phobert_model import run_phobert_experiment, load_valid_experiment_result, lr_to_tag, RESULTS_DIR

if __name__ == "__main__":

    lr = 5e-05
    dr = 0.3
    bs = 16

    run_name = f"phobert_dr{dr}_bs{bs}_lr{lr_to_tag(lr)}"

    train_path = "data/data_01/processed/train.csv"
    val_path = "data/data_01/processed/val.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    exp_res_path = os.path.join(RESULTS_DIR, run_name, "experiment_results.json")

    if os.path.exists(exp_res_path):
        print(f">>> {run_name} already trained.")
    else:
        run_phobert_experiment(
            dr, bs, lr,
            train_df['tokenized_message'], train_df['label'],
            val_df['tokenized_message'], val_df['label']
        )