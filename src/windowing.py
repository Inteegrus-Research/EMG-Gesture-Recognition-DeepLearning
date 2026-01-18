import os
import numpy as np
from src.data_loader import load_all_trials

WINDOW_SIZE = 256
STRIDE = 128
SAVE_DIR = os.path.join("data", "processed")

def window_trials(X, y, g, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    X: (N_trials, T, C)
    returns windows (M, window_size, C), labels (M,), groups (M,)
    """
    Xw, yw, gw = [], [], []
    for trial, label, group in zip(X, y, g):
        T = trial.shape[0]
        if T < window_size:
            continue
        for start in range(0, T - window_size + 1, stride):
            Xw.append(trial[start:start + window_size])
            yw.append(label)
            gw.append(group)
    if len(Xw) == 0:
        raise RuntimeError("No windows created. Check trial lengths and WINDOW_SIZE.")
    return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.int64), np.array(gw, dtype=np.int64)

if __name__ == "__main__":
    print("Loading raw trials from dataset...")
    X, y, g = load_all_trials()
    print(f"Loaded {X.shape[0]} trials. Windowing (L={WINDOW_SIZE}, stride={STRIDE})...")
    Xw, yw, gw = window_trials(X, y, g)
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, "X_all.npy"), Xw)
    np.save(os.path.join(SAVE_DIR, "y_all.npy"), yw)
    np.save(os.path.join(SAVE_DIR, "groups_all.npy"), gw)
    print(f"Saved {Xw.shape[0]} windows to {SAVE_DIR}. Shapes: X={Xw.shape}, y={yw.shape}, groups={gw.shape}")
