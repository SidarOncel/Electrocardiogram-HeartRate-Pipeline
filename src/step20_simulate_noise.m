%% step20_simulate_noise.m — add drift + EMG noise (3 levels) + save labeled windows
clear; close all; clc;

repoRoot = fileparts(fileparts(mfilename("fullpath")));
inPath  = fullfile(repoRoot, "data_processed", "windows_10s.mat");
load(inPath, "all", "fs", "winSec");

rng(0); % reproducible noise

Nw = winSec * fs;
t = (0:Nw-1)'/fs;

% --- EMG noise: band-limited white noise (approx 20–80 Hz) ---
emgBp = designfilt("bandpassiir", ...
    "FilterOrder", 6, ...
    "HalfPowerFrequency1", 20, ...
    "HalfPowerFrequency2", 45, ...
    "SampleRate", fs);

% Severity levels (amplitude ratios relative to std(ecg window))
levels = ["clean","mild","severe"];
driftRat = [0.00, 0.15, 0.35];   % baseline wander amplitude ratios
emgRat   = [0.00, 0.10, 0.25];   % EMG noise amplitude ratios

% ---- Template struct with the extra fields (so assignments are consistent)
template = all(1);
template.ecg_clean = [];
template.ecg_noisy = [];
template.quality  = "";
template.drift_f0  = NaN;
template.drift_rat = NaN;
template.emg_rat   = NaN;

% Expand each window into 3 labeled versions
out2 = repmat(template, numel(all)*3, 1);  % preallocate full size
idx = 0;

for k = 1:numel(all)
    x = all(k).ecg(:);
    s = std(x);

    for L = 1:3
        % --- Drift: low frequency sinusoid (0.15–0.35 Hz)
        f0  = 0.15 + (0.35-0.15)*rand();
        phi = 2*pi*rand();
        drift = (driftRat(L)*s) * sin(2*pi*f0*t + phi);

        % --- EMG: band-limited Gaussian
        w = randn(size(x));
        emg = filtfilt(emgBp, w);
        emg = emg / std(emg);
        emg = (emgRat(L)*s) * emg;

        xNoisy = x + drift + emg;

        idx = idx + 1;

        % copy original metadata
        out2(idx) = template;
        out2(idx).subject = all(k).subject;
        out2(idx).win     = all(k).win;
        out2(idx).ecg     = all(k).ecg;
        out2(idx).hr_gt   = all(k).hr_gt;

        % add new fields
        out2(idx).ecg_clean = x;
        out2(idx).ecg_noisy = xNoisy;
        out2(idx).quality   = levels(L);
        out2(idx).drift_f0  = f0;
        out2(idx).drift_rat = driftRat(L);
        out2(idx).emg_rat   = emgRat(L);
    end
end

fprintf("Expanded %d clean windows into %d labeled windows (x3 levels).\n", numel(all), numel(out2));

outPath = fullfile(repoRoot, "data_processed", "windows_noisy_10s.mat");
save(outPath, "out2", "fs", "winSec", "levels", "driftRat", "emgRat", "-v7.3");
fprintf("Saved: data_processed/windows_noisy_10s.mat\n");

%% Quick visual sanity check (one random example)
k = randi(numel(out2));
x  = out2(k).ecg_clean(:);
xn = out2(k).ecg_noisy(:);

figure;
plot(t, x); hold on; plot(t, xn);
xlabel("Time (s)"); ylabel("ECG");
title(sprintf("Subject %d win %d | quality=%s", out2(k).subject, out2(k).win, out2(k).quality));
legend("clean","noisy"); grid on;

% Save a few representative examples for presentation
exampleIdx = [1, 50, 120];  % pick any window indices
for i = exampleIdx
    figure;
    t = (0:length(out2(i).ecg_noisy)-1)/fs;
    plot(t, out2(i).ecg_clean, 'b'); hold on;
    plot(t, out2(i).ecg_noisy, 'r');
    xlabel("Time (s)"); ylabel("ECG");
    legend("Clean", "Noisy");
    title(sprintf("Example %d — Noise Level: %s", i, out2(i).quality));
    grid on;
    exportgraphics(gcf, fullfile("results", sprintf("example_noise_%03d.png", i)), "Resolution", 300);
end
