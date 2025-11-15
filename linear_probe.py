import os, json, torch
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from glob import glob
import random
import time

def plot_label_histograms(targets, out_path):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(targets["B"], bins=50)
    plt.title("B")

    plt.subplot(2, 2, 2)
    plt.hist(targets["B_all"], bins=50)
    plt.title("B_all")

    plt.subplot(2, 2, 3)
    plt.hist(targets["L"], bins=50)
    plt.title("L")

    plt.subplot(2, 2, 4)
    plt.hist(targets["R"], bins=50)
    plt.title("R")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_layer_data(layer_dir: str):
    """Load and concatenate all hidden states + labels for a given layer."""
    meta_path = os.path.join(layer_dir, "meta_index.jsonl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta_index.jsonl in {layer_dir}")

    records = [json.loads(line) for line in open(meta_path)]
    X_list, B_list,Ball_list, R_list, L_list = [], [], [], [], []

    sampled_records = random.sample(records, 5)
    for rec in sampled_records:
        t0 = time.time()
        data = torch.load(rec["path"], map_location="cpu", weights_only=False)
        t1 = time.time()
        print(f"Loaded {rec['path']} in {t1 - t0:.4f}s")
        h = torch.as_tensor(data["hidden_states"]).reshape(-1, data["hidden_states"].shape[-1])
        X_list.append(h)
        B_list.append(torch.as_tensor(data["B"]).reshape(-1, 1))
        Ball_list.append(torch.as_tensor(data["B_all"]).reshape(-1, 1))
        R_list.append(torch.as_tensor(data["R"]).reshape(-1, 1))
        L_list.append(torch.as_tensor(data["L"]).reshape(-1, 1))

    X = torch.cat(X_list, dim=0).numpy()
    yB = torch.cat(B_list, dim=0).numpy()
    yBall = torch.cat(Ball_list, dim=0).numpy()
    yR = torch.cat(R_list, dim=0).numpy()
    yL = torch.cat(L_list, dim=0).numpy()
    print("Finish loading")

    return X, {"B": yB, "B_all": yBall, "R": yR, "L": yL}


def train_ridge_probe(X_train, y_train, X_test, y_test):
    """Standardize using train only, fit RidgeCV, evaluate on held-out test."""
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    Xtr = x_scaler.transform(X_train)
    Xte = x_scaler.transform(X_test)
    ytr = y_scaler.transform(y_train)
    yte = y_scaler.transform(y_test)

    alphas = np.logspace(-4, 3, 10)
    model = RidgeCV(alphas=alphas)  
    model.fit(Xtr, ytr.ravel())

    y_pred = model.predict(Xte)
    return {
        "alpha": model.alpha_,
        "r2": float(r2_score(yte, y_pred)),
        "mae": float(mean_absolute_error(yte, y_pred)),
    }


def main(layer_dir: str):
    print(f"=== Training probes for {layer_dir} ===")
    X, targets = load_layer_data(layer_dir)

    layer_name = os.path.basename(layer_dir.rstrip("/"))
    output_path = f"{layer_name}_visualization.png"

    plot_label_histograms(targets, output_path)
    print(f"Saved label histograms → {output_path}")

    N = X.shape[0]
    idx = np.random.permutation(N)

    # ---- limit dataset ----
    TRAIN_MAX = 40000
    TEST_MAX  = 4000

    # ensure we don't overflow
    train_end = min(TRAIN_MAX, N)
    test_end  = min(TRAIN_MAX + TEST_MAX, N)

    train_idx = idx[:train_end]
    test_idx  = idx[train_end:test_end]

    X_train = X[train_idx]
    X_test  = X[test_idx]

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    results = {}
    for name, y in targets.items():
        y_train = y[train_idx]
        y_test  = y[test_idx]

        stats = train_ridge_probe(X_train, y_train, X_test, y_test)
        results[name] = stats
        print(f"{name:>5} | α={stats['alpha']:.2e}  R²={stats['r2']:.4f}  MAE={stats['mae']:.4f}")

    summary_path = os.path.join(layer_dir, "meta_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary → {summary_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer_dir",
        required=True,
        help="Path like probe_data/llada_instruct/gsm8k/layer08",
    )
    args = parser.parse_args()

    main(args.layer_dir)



