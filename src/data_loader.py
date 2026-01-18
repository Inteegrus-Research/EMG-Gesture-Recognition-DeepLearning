import os
import re
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List

# Default dataset root
DEFAULT_DATA_ROOT = os.path.join("data", "raw", "Synapse_Dataset")

# Patterns
SESSION_PATTERNS = ["Session*", "session*"]
SUBJECT_PATTERNS = ["*subject*", "*Subject*"]

# Accept leading zeros in filenames like gesture00_trial01.csv
GESTURE_REGEX = re.compile(r"gesture0*([0-9]+)_trial0*([0-9]+)", re.IGNORECASE)
SUBJECT_REGEX = re.compile(r"subject[_\-]?0*([0-9]+)", re.IGNORECASE)

def find_dataset_root(candidate: str = None) -> str:
    """
    Return the dataset root path. Prefer candidate if provided, otherwise try DEFAULT_DATA_ROOT.
    Raises FileNotFoundError if not found.
    """
    if candidate:
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(DEFAULT_DATA_ROOT):
        return DEFAULT_DATA_ROOT
    # fallback: try to find a directory that looks like the dataset
    search_paths = [
        os.path.join("data", "raw"),
        "data",
        "Synapse_Dataset",
        "datasets",
        ".",
    ]
    for p in search_paths:
        if not os.path.isdir(p):
            continue
        # look for Session* underneath
        for pat in SESSION_PATTERNS:
            if glob.glob(os.path.join(p, pat)):
                return p
    tried = "\n".join([os.path.abspath(p) for p in [candidate or DEFAULT_DATA_ROOT]])
    raise FileNotFoundError(f"Could not locate dataset root. Tried:\n{tried}\nPlease ensure the dataset exists.")

def extract_label(filename: str) -> int:
    """
    Extract gesture id from filename like gesture00_trial01.csv
    Returns int gesture id.
    """
    m = GESTURE_REGEX.search(filename)
    if not m:
        raise ValueError(f"Filename does not match gesture pattern: {filename}")
    return int(m.group(1))

def extract_subject_id(folder_name: str) -> int:
    m = SUBJECT_REGEX.search(folder_name)
    if not m:
        raise ValueError(f"Subject folder does not match expected pattern: {folder_name}")
    return int(m.group(1))

def list_session_dirs(dataset_root: str) -> List[str]:
    sessions = []
    for pat in SESSION_PATTERNS:
        sessions.extend(sorted(glob.glob(os.path.join(dataset_root, pat))))
    return [p for p in sessions if os.path.isdir(p)]

def load_all_trials(dataset_root: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads all labeled trials (CSV) from dataset root.
    Returns (X, y, groups)
      - X: ndarray (N_trials, T_samples, 8)
      - y: ndarray (N_trials,)
      - groups: ndarray (N_trials,) -- subject group ids for GroupKFold
    """
    if dataset_root is None:
        dataset_root = find_dataset_root()
    else:
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"Provided dataset_root does not exist: {dataset_root}")

    session_dirs = list_session_dirs(dataset_root)

    # If no Session folders, allow subject folders directly under dataset_root
    if not session_dirs:
        # treat dataset_root as single session container
        session_dirs = [dataset_root]

    X, y, groups = [], [], []
    subject_to_group = {}
    next_group_id = 0

    for session_path in session_dirs:
        # find subject folders (session1_subject_1 etc.) under session_path
        subject_dirs = []
        for pat in SUBJECT_PATTERNS:
            subject_dirs.extend(sorted(glob.glob(os.path.join(session_path, pat))))
        subject_dirs = [p for p in subject_dirs if os.path.isdir(p)]

        # if no subject dirs, try to use CSVs directly under session_path (edge case)
        if not subject_dirs:
            subject_dirs = [session_path]

        for subject_path in subject_dirs:
            folder = os.path.basename(subject_path)
            try:
                subject_id = extract_subject_id(folder)
            except ValueError:
                # fallback: create deterministic synthetic id
                subject_id = abs(hash(os.path.abspath(subject_path))) % (10**6)

            if subject_id not in subject_to_group:
                subject_to_group[subject_id] = next_group_id
                next_group_id += 1
            group_id = subject_to_group[subject_id]

            csv_files = sorted(glob.glob(os.path.join(subject_path, "*.csv")))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"Warning: failed to read {csv_file}: {e}")
                    continue

                # Expect 8 channels per CSV
                if df.shape[1] != 8:
                    print(f"Skipping {csv_file}: expected 8 channels, found {df.shape[1]}")
                    continue

                # Extract label from filename (skip files without label pattern)
                fname = os.path.basename(csv_file)
                try:
                    label = extract_label(fname)
                except ValueError:
                    print(f"Skipping {csv_file}: filename doesn't contain gesture label.")
                    continue

                X.append(df.values.astype(np.float32))
                y.append(int(label))
                groups.append(int(group_id))

    if len(X) == 0:
        raise RuntimeError(f"No labeled trials found under '{dataset_root}'. Check CSV naming and locations.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(groups, dtype=np.int64)

if __name__ == "__main__":
    X, y, g = load_all_trials()
    print("Trials:", X.shape[0])
    print("Signal shape:", X.shape[1:])
    print("Classes:", np.unique(y))
    print("Subjects (groups):", len(np.unique(g)))
