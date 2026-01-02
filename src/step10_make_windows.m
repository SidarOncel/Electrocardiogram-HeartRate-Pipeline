%% 10_make_windows.m — window ECG into 10s segments + compute GT HR per window
clear; close all; clc;

repoRoot = fileparts(fileparts(mfilename("fullpath")));
dataPath = fullfile(repoRoot, "data_raw", "bidmc_data.mat");
S = load(dataPath);
data = S.data;

winSec = 10;
fs = data(1).ekg.fs;
Nw = winSec * fs;

all = [];  % struct array windows
idx = 0;

for subj = 1:numel(data)
    ecg = data(subj).ekg.v(:);
    hr1 = data(subj).ref.params.hr.v(:);  % 1 Hz HR vector (length ~480)

    nWin = floor(numel(ecg) / Nw);

    for w = 1:nWin
        i0 = (w-1)*Nw + 1;
        i1 = i0 + Nw - 1;

        h0 = (w-1)*winSec + 1;
        h1 = h0 + winSec - 1;

        if h1 > numel(hr1)
            continue;
        end

        hrSeg = hr1(h0:h1);

        if any(isnan(hrSeg))
            continue;
        end

        idx = idx + 1;
        all(idx).subject = subj;
        all(idx).win     = w;
        all(idx).ecg     = ecg(i0:i1);
        all(idx).hr_gt   = mean(hrSeg);   % your choice: avg of 10 HR samples
    end
end

fprintf("Created %d windows total.\n", numel(all));

outDir = fullfile(repoRoot, "data_processed");
if ~isfolder(outDir), mkdir(outDir); end

save(fullfile(outDir, "windows_10s.mat"), "all", "fs", "winSec", "-v7.3");
fprintf("Saved: data_processed/windows_10s.mat\n");
