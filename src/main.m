%% main.m — sanity check + first visualizations
clear; close all; clc;

% Make paths robust no matter where you run from
repoRoot = fileparts(fileparts(mfilename("fullpath"))); % one level up from /src
dataPath = fullfile(repoRoot, "data_raw", "bidmc_data.mat");

assert(isfile(dataPath), "Could not find bidmc_data.mat at: %s", dataPath);

S = load(dataPath);  % loads variable 'data' inside struct S
assert(isfield(S, "data"), "Loaded file does not contain variable 'data'.");
data = S.data;

fprintf("Loaded BIDMC: %d recordings\n", numel(data));
fs = data(1).ekg.fs;
fprintf("ECG sampling rate: %.0f Hz\n", fs);

% Basic lengths
ecg = data(1).ekg.v(:);
hr = data(1).ref.params.hr.v(:);    % 1 Hz HR samples
fprintf("HR samples : %d (%.1f min)\n", numel(hr),  numel(hr)/60);


% Plot corresponding 10 HR points (1 Hz)
figure;
plot(0:9, hr(1:10), "o-");
xlabel("Time (s)"); ylabel("HR (bpm)");
title("Subject 1 — Ground-truth HR samples (first 10 seconds)");
grid on;


% Plot a 10-second window of ECG
winSec = 10;
Nw = winSec * fs;
t = (0:Nw-1)/fs;

figure;
plot(t, ecg(1:Nw));
xlabel("Time (s)"); ylabel("ECG amplitude");
title("Subject 1 — Raw ECG (first 10 seconds)");
grid on;
