import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.model import NeuroCNN, NeuroResNet, NeuroTCN
from src.preprocessing import SignalScaler

BATCH_SIZE = 64
WINDOW_SIZE = 256
STRIDE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ARTIFACTS_DIR = "artifacts"

class EMGTestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]

def load_raw_test_data(test_dir):
    csv_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under test directory: {test_dir}")

    all_windows = []
    file_map = []
    idx = 0

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Warning: cannot read {fpath}: {e}")
            continue
        data = df.values.astype(np.float32)
        if data.shape[1] != 8:
            print(f"Skipping {fpath}: expected 8 channels, got {data.shape[1]}")
            continue

        file_windows = []
        for start in range(0, data.shape[0] - WINDOW_SIZE + 1, STRIDE):
            file_windows.append(data[start:start + WINDOW_SIZE])

        if len(file_windows) == 0:
            # file too short to form a single window
            continue

        all_windows.extend(file_windows)
        file_map.append({"filename": os.path.basename(fpath), "start_idx": idx, "end_idx": idx + len(file_windows)})
        idx += len(file_windows)

    if len(all_windows) == 0:
        raise RuntimeError("No valid windows could be generated from test CSVs.")

    return np.array(all_windows, dtype=np.float32), file_map

def load_models_and_scalers(device):
    models = []
    scalers = []
    for arch in ['cnn', 'resnet', 'tcn']:
        for fold in range(5):
            scaler_path = os.path.join(ARTIFACTS_DIR, f'scaler_{arch}_fold_{fold}.json')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler missing: {scaler_path}")
            scaler = SignalScaler()
            scaler.load(scaler_path)
            scalers.append(scaler)

            model_path = os.path.join(ARTIFACTS_DIR, f'model_{arch}_fold_{fold}.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint missing: {model_path}")

            if arch == 'cnn':
                model = NeuroCNN()
            elif arch == 'resnet':
                model = NeuroResNet()
            else:
                model = NeuroTCN()

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
    return models, scalers

def predict_ensemble(models, scalers, X_windows, device):
    # infer number of classes
    with torch.no_grad():
        sample = torch.FloatTensor(X_windows[:1]).to(device)
        sample_out = models[0](sample)
        n_classes = sample_out.shape[1]

    total_probs = np.zeros((X_windows.shape[0], n_classes), dtype=np.float32)
    num_models = len(models)

    for i, (model, scaler) in enumerate(zip(models, scalers)):
        X_scaled = scaler.transform(X_windows)  # (N_windows, L, C)
        dataset = EMGTestDataset(X_scaled)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds.append(probs.cpu().numpy())
        model_probs = np.concatenate(preds, axis=0)
        total_probs += model_probs

    avg_probs = total_probs / num_models
    return avg_probs

def main():
    # prefer data/raw/test
    TEST_DIR = os.path.join("data", "raw", "test")
    # fallback: if no test dir, use first session for quick local checks
    FALLBACK_DIR = os.path.join("data", "raw", "Synapse_Dataset", "Session1", "session1_subject_1")

    target = FALLBACK_DIR
    if os.path.exists(TEST_DIR):
        # check for CSVs in TEST_DIR
        has_csv = glob.glob(os.path.join(TEST_DIR, "*.csv")) or glob.glob(os.path.join(TEST_DIR, "**", "*.csv"), recursive=True)
        if has_csv:
            target = TEST_DIR

    print(f"Running inference on: {target}")
    X_windows, file_map = load_raw_test_data(target)
    print(f"Generated {X_windows.shape[0]} windows from {len(file_map)} files.")

    models, scalers = load_models_and_scalers(DEVICE)
    window_probs = predict_ensemble(models, scalers, X_windows, DEVICE)

    # aggregate per file
    filenames = []
    labels = []
    for item in file_map:
        probs = window_probs[item['start_idx']:item['end_idx']]
        avg = np.mean(probs, axis=0)
        pred = int(np.argmax(avg))
        filenames.append(item['filename'])
        labels.append(pred)

    os.makedirs("submission", exist_ok=True)
    pd.DataFrame({"filename": filenames, "label": labels}).to_csv("submission/submission.csv", index=False)
    print("submission/submission.csv generated.")

if __name__ == "__main__":
    main()
