# Electrocardiogram-HeartRate-Pipeline

A DSP + interpretable machine learning pipeline for estimating heart rate (HR) from ECG (electrocardiogram) signals, designed with wearable-style constraints in mind.

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
 
