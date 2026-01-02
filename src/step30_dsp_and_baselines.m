%% step30_dsp_and_baselines.m — bandpass preprocess + HR baselines (R-peak, Welch)
clear; close all; clc;

repoRoot = fileparts(fileparts(mfilename("fullpath")));
inPath = fullfile(repoRoot, "data_processed", "windows_noisy_10s.mat");
load(inPath, "out2", "fs", "winSec");

Nw = winSec * fs;
t = (0:Nw-1)'/fs;

%% 1) DSP preprocessing filter (IIR bandpass 0.5–40 Hz)
bp = designfilt("bandpassiir", ...
    "FilterOrder", 6, ...
    "HalfPowerFrequency1", 0.5, ...
    "HalfPowerFrequency2", 40, ...
    "SampleRate", fs);

%% 2) Process each window: filter + baselines
% We'll store results into new fields
for k = 1:numel(out2)
    x = out2(k).ecg_noisy(:);

    % Zero-phase filtering (important)
    xf = filtfilt(bp, x);
    out2(k).ecg_filt = xf;

    % ---------- Baseline A: R-peak / RR-based HR ----------
    % (Built-in peak picking with constraints)
    minDistSec = 0.30; % ~200 bpm max
    minDist = round(minDistSec * fs);

    % Use absolute value to make it robust to polarity
    sig = abs(xf);

    % Adaptive threshold based on signal distribution
    thr = prctile(sig, 90);

    [pks, locs] = findpeaks(sig, "MinPeakDistance", minDist, "MinPeakHeight", thr);

    if numel(locs) >= 2
        rr = diff(locs) / fs;              % seconds
        hr_rpeak = 60 / mean(rr);          % bpm
    else
        hr_rpeak = NaN;
    end
    out2(k).hr_rpeak = hr_rpeak;
    out2(k).n_peaks  = numel(locs);

    % ---------- Baseline B: Welch PSD peak in HR band ----------
    % HR band (Hz): 0.7–3.0 Hz => 42–180 bpm
    [Pxx, F] = pwelch(xf, hann(256), 128, 512, fs); % built-in
    bandMask = (F >= 0.7) & (F <= 3.0);

    if any(bandMask)
        [~, imax] = max(Pxx(bandMask));
        Fband = F(bandMask);
        f_peak = Fband(imax);
        hr_welch = 60 * f_peak;
    else
        hr_welch = NaN;
    end
    out2(k).hr_welch = hr_welch;
end

fprintf("Done. Added fields: ecg_filt, hr_rpeak, hr_welch.\n");

%% 3) Quick sanity plots on one example
k = randi(numel(out2));
x  = out2(k).ecg_noisy(:);
xf = out2(k).ecg_filt(:);

figure;
plot(t, x); hold on; plot(t, xf);
xlabel("Time (s)"); ylabel("ECG");
title(sprintf("Subject %d win %d | %s", out2(k).subject, out2(k).win, out2(k).quality));
legend("noisy","filtered"); grid on;

% PSD plot
[Pxx, F] = pwelch(xf, hann(256), 128, 512, fs);
figure;
plot(F, 10*log10(Pxx));
xlim([0 10]);
xlabel("Frequency (Hz)"); ylabel("PSD (dB/Hz)");
title("Welch PSD of filtered ECG (0–10 Hz view)");
grid on;

%% 4) Save updated struct
outPath = fullfile(repoRoot, "data_processed", "windows_noisy_baselines_10s.mat");
save(outPath, "out2", "fs", "winSec", "-v7.3");
fprintf("Saved: data_processed/windows_noisy_baselines_10s.mat\n");
