# ECG Signal Processing & Heartbeat Classification

## Objective
Apply signal processing techniques to clean a noisy ECG recording, detect heartbeat peaks, compute heart rate, and classify heartbeat segments into 5 diagnostic categories using machine learning.

## Part 1: Signal Processing Pipeline

**Input:** 15,000-sample ECG signal recorded at 1 kHz with noise and missing values

**Pipeline:**
1. **Missing value imputation** — Linear interpolation for 5 missing amplitude values
2. **FFT-based noise filtering** — Transform to frequency domain, zero out components above 100 Hz, inverse FFT to recover cleaned signal
3. **Peak detection** — Cross-correlation with derivative kernels `[-1, 1, 0]` and `[0, 1, -1]` to identify local maxima
4. **R-peak isolation** — Threshold (mean + 2σ) to separate heartbeat R-peaks from minor peaks
5. **Heart rate computation** — Beat-to-beat interval converted to instantaneous BPM

## Part 2: Heartbeat Classification

**Dataset:** 3,841 ECG segments × 187 features (5 diagnostic classes, sampled at 125 Hz)

| Model | Best Hyperparameters | Test Accuracy |
|-------|---------------------|---------------|
| KNN | n_neighbors=1 | 88.2% |
| Decision Tree | max_depth=11 | 84.3% |
| Random Forest | max_depth=26, n_estimators=10 | **88.7%** |

All models tuned with 5-fold cross-validation via GridSearchCV.

## Key Findings
- FFT low-pass filtering effectively removed high-frequency noise while preserving ECG waveform structure
- Random Forest achieved the highest classification accuracy (88.7%)
- Cross-correlation peak detection provided robust R-peak identification without relying on library peak-finding functions

## Transferable Techniques
The methods used here, FFT filtering, time-series interpolation, cross-correlation, and multi-class classification on high-dimensional time series, are directly applicable to environmental sensor data (water quality monitors, acoustic ecology, satellite spectral time series).

## Tools
Python, NumPy, pandas, SciPy (FFT, cross-correlation), scikit-learn, matplotlib

## Files
- `ecg_signal_processing_classification.ipynb` — Full analysis notebook
- `project_ecg_part1_signal_missing_value.csv` — Raw ECG signal (Part 1)
- `ECG_dataX.csv` — Heartbeat segment features (Part 2)
- `ECG_dataY.csv` — Heartbeat class labels (Part 2)

## Usage
```bash
git clone https://github.com/tommypotts/ecg-signal-classification.git
cd ecg-signal-classification
jupyter notebook ecg_signal_processing_classification.ipynb
```
