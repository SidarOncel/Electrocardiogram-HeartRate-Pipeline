# Electrocardiogram-HeartRate-Pipeline

A Digital Signal Processing + interpretable machine learning pipeline for estimating heart rate (HR) from ECG (electrocardiogram) signals, designed with wearable-style constraints in mind.

## What this project does
- Splits ECG into 10-second windows and computes ground-truth HR per window.
- DSP front-end (filtering + spectral analysis) to make signal processing essential.
- DSP-only baselines:
  - R-peak/RR-based HR estimation
  - Frequency-domain HR estimation (Welch/FFT peak in the HR band)
- Interpretable ML:
  - Linear regression to combine DSP-derived features into an HR estimate
  - Optional per-user calibration using the first 60 seconds (output bias correction)
- Robustness:
  - Simulates realistic noise (baseline drift + EMG-like noise) at multiple severity levels
 
## Dataset

This project uses the **BIDMC PPG and Respiration Dataset (v1.0.0)** from PhysioNet.
It contains **53 recordings** (each **8 minutes**) of physiological waveforms including **ECG (Lead II)**, **PPG**, and **impedance respiration** sampled at **125 Hz**, plus derived numerics including **heart rate (HR)** sampled at **1 Hz**.

**Data is not included in this repository**. Please download `bidmc_data.mat` from PhysioNet and place it in `data_raw/`.

- Dataset page (PhysioNet): BIDMC PPG and Respiration Dataset v1.0.0
- DOI: 10.13026/C2208R

## How to run
1) Download dataset to `data_raw/`
2) Run the main script (to be added) from MATLAB

## Citing

If you use this repository, please cite:
1) The BIDMC dataset (PhysioNet, DOI: 10.13026/C2208R).
2) Pimentel, M.A.F. et al., *Towards a Robust Estimation of Respiratory Rate from Pulse Oximeters*, IEEE TBME, 2016.
3) Goldberger et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation, 2000.
