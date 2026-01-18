# Synapse — The NeuroTech Challenge  
**Robust, Subject-Independent sEMG Gesture Recognition**

---

## Project Overview
This repository presents a **competition-grade**, subject-independent deep learning solution for the **Synapse NeuroTech Challenge**.  
The system classifies **8-channel surface EMG (sEMG)** signals into discrete gesture classes with a strong emphasis on **generalization, robustness, and reproducibility**.

The approach is built around a **Soft-Voting Ensemble** of three complementary neural architectures (**CNN, ResNet, TCN**), trained using **5-Fold Group Cross-Validation** to strictly prevent subject leakage and optimistic bias.

---

## Results Snapshot
- **Consistent top-tier cross-subject validation performance** with minimal fold variance  
- **Low-variance predictions** via 15-model ensemble inference  
- **Leakage-free preprocessing and evaluation pipeline**  
- **Edge-compatible architectures** enabling efficient deployment  

> Design priority: *stability and generalization over fragile peak accuracy.*

---

## Methodology

### Data Preprocessing
To address non-stationary sEMG dynamics and inter-subject variability:

- **Windowing**  
  Fixed windows of `256` time steps with stride `128` (50% overlap)

- **Robust Scaling**  
  Custom `SignalScaler` using **Median & IQR**, chosen over Z-score normalization to suppress:
  - Motion artifacts  
  - Sudden muscle spikes  
  - Subject-specific amplitude bias  

- **Leakage Prevention**  
  Scaling statistics are fit **exclusively on training folds** and reused for validation/test data

---

### Model Architecture
An ensemble of three lightweight yet expressive architectures:

1. **NeuroCNN**  
   - 3-layer 1D CNN  
   - Extracts local spatial patterns across EMG channels  

2. **NeuroResNet**  
   - Deep residual blocks  
   - Learns hierarchical representations while avoiding gradient degradation  

3. **NeuroTCN**  
   - Temporal Convolutional Network with exponential dilation  
   - Dilation: `d = 1, 2, 4, ..., 64`  
   - Effective receptive field ≈ **255 time steps**, spanning entire gesture windows  

---

### Inference Strategy
Final predictions are generated via a **Soft-Voting Ensemble**:

- 3 architectures × 5 folds = **15 trained models**
- Per-window softmax probabilities are averaged
- Final label computed as `argmax(mean_probabilities)`

This ensemble mitigates individual model failure modes caused by **electrode shift, muscle fatigue, and subject-specific activation patterns**, while consistently reducing prediction variance compared to any single architecture or fold.

---

## Quick Start (TL;DR)
```bash
pip install -r requirements.txt
python -m src.inference
```

Output:
```
submission/submission.csv
```

---

## Repository Structure
```
.
├── artifacts/                  # Trained models (.pth) and scalers (.json)
│   ├── model_cnn_fold_0.pth
│   ├── ... (15 models total)
│   ├── scaler_cnn_fold_0.json
│   ├── ... (15 scalers total)
│   ├── training_log_cnn.csv
│   └── ... (3 logs total)
├── data/
│   ├── processed/              # Windowed .npy files
│   └── raw/                    # Raw input CSVs
│       ├── Synapse_Dataset/    # Exact competition dataset folder             
│       └── test/               # Have to place the test .csv files
├── src/
│   ├── data_loader.py          # Subject-aware data ingestion
│   ├── windowing.py            # Sliding window generation
│   ├── preprocessing.py        # Robust SignalScaler
│   ├── model.py                # CNN, ResNet, TCN definitions
│   ├── train.py                # GroupKFold training pipeline
│   └── inference.py            # End-to-end inference script
├── report/                     # The LaTeX Report
│   ├── figures/               
│   ├── main.tex
│   └── main.pdf
├── requirements.txt
└── README.md
```

---

## Environment & Dependencies
- **Python:** 3.10+
- **Hardware:** CUDA-enabled GPU recommended (optional for inference)

### Installation
```bash
pip install -r requirements.txt
```

### Core Dependencies
```
numpy>=2.1.0
pandas>=2.2.3
scipy>=1.14.1
torch>=2.5.1
scikit-learn>=1.5.2
joblib>=1.4.2
tqdm>=4.66.5
matplotlib>=3.9.2
pyyaml>=6.0.2
```

---

## Inference Instructions
1. **Place test CSV files into:**
   ```
   data/raw/test/
   ```

2. **Run inference:**
   ```bash
   python -m src.inference
   ```

3. **Retrieve predictions:**
   ```
   submission/submission.csv
   ```

---

## Training (Reproducibility)
### Step 1 — Place the Synapse_Dataset
Place the Session1, Session2 and Session3 folders inside the Synapse_Dataset Folder.

### Step 2 — Window the Data
```bash
python -m src.windowing
```
Generates windowed `.npy` files in `data/processed/`.

### Step 3 — Train Models
Each command performs 5-Fold Group Cross-Validation and saves artifacts automatically:
```bash
python -m src.train --model cnn
python -m src.train --model resnet
python -m src.train --model tcn
```

---

## Design Rationale
- **Median–IQR scaling** provides resilience against industrial EMG artifacts
- **GroupKFold** enforces strict subject independence and realistic evaluation
- **TCN dilation** captures full temporal muscle activation dynamics
- **Soft-voting ensemble** prioritizes robustness, stability, and reliability under real-world conditions