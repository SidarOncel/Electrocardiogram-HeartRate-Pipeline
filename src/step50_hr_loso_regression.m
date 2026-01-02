%% step50_hr_loso_regression.m — LOSO HR regression + 60s bias calibration (Ridge, manual CV)
clear; close all; clc;

repoRoot = fileparts(fileparts(mfilename("fullpath")));
load(fullfile(repoRoot, "data_processed", "features_10s.mat"), "T");

% -----------------------------
% Feature set (interpretable)
% -----------------------------
featNames = ["hr_rpeak","hr_welch","n_peaks","rms_filt","std_filt","kurt_filt", ...
             "bp_low","bp_high","bp_ratio","spec_ent"];

% Missingness indicators so model can learn when baselines fail
missNames = ["miss_rpeak","miss_welch"];
allFeat   = [featNames, missNames];

subjects = unique(T.subject);
levels   = ["clean","mild","severe"];

% -----------------------------
% Results table (preallocated)
% -----------------------------
nRows = numel(subjects) * numel(levels);

R = table('Size',[nRows 9], ...
    'VariableTypes', {'double','categorical','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'subject','quality','n', ...
                      'mape_rpeak','mape_welch','mape_ml','mape_ml_cal', ...
                      'medape_ml','medape_ml_cal'});
row = 0;

% Coef output dir
outCoefDir = fullfile(repoRoot, "results", "beta");
if ~isfolder(outCoefDir), mkdir(outCoefDir); end

% Ridge lambda grid (inner CV)
lambdas = logspace(-4, 2, 15);  % keep modest for speed
Kfold = 5;

% -----------------------------
% LOSO loop
% -----------------------------
for si = 1:numel(subjects)
    testSubj = subjects(si);

    isTest = (T.subject == testSubj);
    Tr = T(~isTest, :);
    Te = T(isTest, :);

    % Quality labels
    q = string(Te.quality);

    % Build X/y
    Xtr = Tr{:, featNames};
    ytr = Tr.hr_gt;

    Xte = Te{:, featNames};
    yte = Te.hr_gt;

    % Missing indicators
    miss_rpeak_tr = isnan(Tr.hr_rpeak);
    miss_welch_tr = isnan(Tr.hr_welch);
    miss_rpeak_te = isnan(Te.hr_rpeak);
    miss_welch_te = isnan(Te.hr_welch);

    Xtr = [Xtr, double(miss_rpeak_tr), double(miss_welch_tr)];
    Xte = [Xte, double(miss_rpeak_te), double(miss_welch_te)];

    % Impute NaNs using training medians
    med = nanmedian(Xtr, 1);
    for j = 1:size(Xtr,2)
        Xtr(isnan(Xtr(:,j)), j) = med(j);
        Xte(isnan(Xte(:,j)), j) = med(j);
    end

    % Standardize using training stats
    [XtrZ, mu, sig] = zscore(Xtr);
    sig(sig==0) = 1;
    XteZ = (Xte - mu) ./ sig;

    % -----------------------------
    % Ridge: choose Lambda via manual K-fold CV
    % -----------------------------
    cvp = cvpartition(numel(ytr), 'KFold', Kfold);
    mse = zeros(numel(lambdas),1);

    for li = 1:numel(lambdas)
        se = zeros(Kfold,1);

        for f = 1:Kfold
            trIdx = training(cvp, f);
            vaIdx = test(cvp, f);

            mdlTmp = fitrlinear(XtrZ(trIdx,:), ytr(trIdx), ...
                "Learner","leastsquares", ...
                "Regularization","ridge", ...
                "Lambda", lambdas(li));

            yva = predict(mdlTmp, XtrZ(vaIdx,:));
            err = yva - ytr(vaIdx);
            se(f) = mean(err.^2);
        end

        mse(li) = mean(se);
    end

    [~, best] = min(mse);
    lambdaBest = lambdas(best);

    mdl = fitrlinear(XtrZ, ytr, ...
        "Learner","leastsquares", ...
        "Regularization","ridge", ...
        "Lambda", lambdaBest);

    yhat = predict(mdl, XteZ);

    % -----------------------------
    % 60s personalization (bias correction)
    % 60s = first 6 windows of 10s
    % Prefer not severe for calibration if possible
    % -----------------------------
    calibMask = false(height(Te),1);
    first6 = (1:min(6,height(Te)))';
    calibMask(first6) = true;

    notSevere = (q ~= "severe");
    calibMask = calibMask & notSevere;

    if sum(calibMask) >= 2
        bias = mean(yte(calibMask) - yhat(calibMask));
    else
        bias = 0;
    end

    yhat_cal = yhat + bias;

    % -----------------------------
    % Metrics (MAPE)
    % -----------------------------
    ape_ml     = abs(yhat     - yte) ./ yte * 100;
    ape_ml_cal = abs(yhat_cal - yte) ./ yte * 100;

    ape_rpeak = abs(Te.hr_rpeak - yte) ./ yte * 100;
    ape_welch = abs(Te.hr_welch - yte) ./ yte * 100;

    % -----------------------------
    % Store summary per quality level
    % -----------------------------
    for lvl = levels
        mask = (q == lvl);

        row = row + 1;
        R.subject(row) = testSubj;
        R.quality(row) = categorical(lvl);
        R.n(row) = sum(mask);

        R.mape_rpeak(row)  = mean(ape_rpeak(mask), "omitnan");
        R.mape_welch(row)  = mean(ape_welch(mask), "omitnan");
        R.mape_ml(row)     = mean(ape_ml(mask), "omitnan");
        R.mape_ml_cal(row) = mean(ape_ml_cal(mask), "omitnan");

        R.medape_ml(row)     = median(ape_ml(mask), "omitnan");
        R.medape_ml_cal(row) = median(ape_ml_cal(mask), "omitnan");
    end

    % -----------------------------
    % Save coefficients (linear + interpretable)
    % NOTE: corresponds to STANDARDIZED features
    % -----------------------------
    beta = [mdl.Bias; mdl.Beta];
    names = ["Intercept", allFeat];
    coefTable = table(names(:), beta(:), 'VariableNames', {'term','beta'});
    writetable(coefTable, fullfile(outCoefDir, sprintf("coef_subject_%02d.csv", testSubj)));
end

% Save results
outPath = fullfile(repoRoot, "results", "hr_loso_summary.csv");
writetable(R, outPath);
fprintf("Saved HR LOSO summary: %s\n", outPath);

% Weighted mean MAPE across all rows (weighted by n)
w = R.n;
fprintf("\nWeighted mean MAPE (ML, no calib): %.2f%%\n", sum(w .* R.mape_ml) / sum(w));
fprintf("Weighted mean MAPE (ML + calib):  %.2f%%\n", sum(w .* R.mape_ml_cal) / sum(w));
