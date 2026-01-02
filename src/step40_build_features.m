%% step40_build_features.m — extract interpretable features into a table
clear; close all; clc;

repoRoot = fileparts(fileparts(mfilename("fullpath")));
inPath = fullfile(repoRoot, "data_processed", "windows_noisy_baselines_10s.mat");
load(inPath, "out2", "fs", "winSec");

N = numel(out2);

% Preallocate feature arrays
subject  = zeros(N,1);
win      = zeros(N,1);
quality  = strings(N,1);
hr_gt    = zeros(N,1);

hr_rpeak = nan(N,1);
hr_welch = nan(N,1);
n_peaks  = nan(N,1);

rms_filt = nan(N,1);
std_filt = nan(N,1);
kurt_filt= nan(N,1);

bp_low   = nan(N,1);  % 0.5–5 Hz
bp_high  = nan(N,1);  % 20–40 Hz
bp_ratio = nan(N,1);
spec_ent = nan(N,1);

for k = 1:N
    subject(k) = out2(k).subject;
    win(k)     = out2(k).win;
    quality(k) = string(out2(k).quality);
    hr_gt(k)   = out2(k).hr_gt;

    hr_rpeak(k) = out2(k).hr_rpeak;
    hr_welch(k) = out2(k).hr_welch;
    n_peaks(k)  = out2(k).n_peaks;

    xf = out2(k).ecg_filt(:);

    rms_filt(k)  = rms(xf);
    std_filt(k)  = std(xf);
    kurt_filt(k) = kurtosis(xf);

    % Bandpowers (built-in)
    bp_low(k)  = bandpower(xf, fs, [0.5 5]);
    bp_high(k) = bandpower(xf, fs, [20 40]);
    bp_ratio(k)= bp_low(k) / (bp_high(k) + eps);

    % Spectral entropy from Welch PSD
    [Pxx,F] = pwelch(xf, hann(256), 128, 512, fs);
    mask = (F >= 0.5) & (F <= 40);
    p = Pxx(mask);
    p = p / (sum(p) + eps);
    spec_ent(k) = -sum(p .* log2(p + eps));
end

T = table(subject, win, categorical(quality), hr_gt, ...
          hr_rpeak, hr_welch, n_peaks, ...
          rms_filt, std_filt, kurt_filt, ...
          bp_low, bp_high, bp_ratio, spec_ent, ...
          'VariableNames', {'subject','win','quality','hr_gt', ...
                            'hr_rpeak','hr_welch','n_peaks', ...
                            'rms_filt','std_filt','kurt_filt', ...
                            'bp_low','bp_high','bp_ratio','spec_ent'});

outDir = fullfile(repoRoot, "data_processed");
if ~isfolder(outDir), mkdir(outDir); end

save(fullfile(outDir, "features_10s.mat"), "T", "fs", "winSec", "-v7.3");
writetable(T, fullfile(outDir, "features_10s.csv"));

fprintf("Saved features to data_processed/features_10s.mat and .csv\n");
