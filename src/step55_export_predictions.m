%% step55_export_predictions.m — generate per-window ML & calibrated predictions (LOSO)
clear; close all; clc;

repoRoot = fileparts(fileparts(mfilename("fullpath")));
load(fullfile(repoRoot, "data_processed", "features_10s.mat"), "T");

featNames = ["hr_rpeak","hr_welch","n_peaks","rms_filt","std_filt","kurt_filt", ...
             "bp_low","bp_high","bp_ratio","spec_ent"];
missNames = ["miss_rpeak","miss_welch"];
allFeat   = [featNames, missNames];

subjects = unique(T.subject);

% Where we'll store predictions (aligned to rows of T)
yhat_all     = nan(height(T),1);
yhat_cal_all = nan(height(T),1);
bias_subj    = nan(numel(subjects),1);

% Ridge tuning
lambdas = logspace(-4, 2, 12);  % moderate for speed
Kfold   = 5;

for si = 1:numel(subjects)
    testSubj = subjects(si);
    isTest = (T.subject == testSubj);

    Tr = T(~isTest,:);
    Te = T(isTest,:);

    Xtr = Tr{:, featNames}; ytr = Tr.hr_gt;
    Xte = Te{:, featNames}; yte = Te.hr_gt;

    % missingness indicators
    Xtr = [Xtr, double(isnan(Tr.hr_rpeak)), double(isnan(Tr.hr_welch))];
    Xte = [Xte, double(isnan(Te.hr_rpeak)), double(isnan(Te.hr_welch))];

    % median impute using training
    med = nanmedian(Xtr, 1);
    for j = 1:size(Xtr,2)
        Xtr(isnan(Xtr(:,j)), j) = med(j);
        Xte(isnan(Xte(:,j)), j) = med(j);
    end

    % standardize using training stats
    [XtrZ, mu, sig] = zscore(Xtr);
    sig(sig==0) = 1;
    XteZ = (Xte - mu) ./ sig;

    % manual K-fold CV to pick lambda
    cvp = cvpartition(numel(ytr), 'KFold', Kfold);
    mse = zeros(numel(lambdas),1);

    for li = 1:numel(lambdas)
        se = zeros(Kfold,1);
        for f = 1:Kfold
            trIdx = training(cvp,f);
            vaIdx = test(cvp,f);

            mdlTmp = fitrlinear(XtrZ(trIdx,:), ytr(trIdx), ...
                "Learner","leastsquares", ...
                "Regularization","ridge", ...
                "Lambda", lambdas(li));

            yva = predict(mdlTmp, XtrZ(vaIdx,:));
            se(f) = mean((yva - ytr(vaIdx)).^2);
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

    % 60s bias calibration: first 6 windows, prefer not severe
    q = string(Te.quality);
    calibMask = false(height(Te),1);
    calibMask(1:min(6,height(Te))) = true;
    calibMask = calibMask & (q ~= "severe");

    if sum(calibMask) >= 2
        bias = mean(yte(calibMask) - yhat(calibMask));
    else
        bias = 0;
    end

    yhat_cal = yhat + bias;

    % Store back into aligned arrays
    yhat_all(isTest)     = yhat;
    yhat_cal_all(isTest) = yhat_cal;
    bias_subj(si) = bias;

    % Save coefficients per subject (optional)
    beta = [mdl.Bias; mdl.Beta];
    names = ["Intercept", allFeat];
    coefTable = table(names(:), beta(:), 'VariableNames', {'term','beta'});
    outCoefDir = fullfile(repoRoot, "results", "beta");
    if ~isfolder(outCoefDir), mkdir(outCoefDir); end
    writetable(coefTable, fullfile(outCoefDir, sprintf("coef_subject_%02d.csv", testSubj)));
end

Tpred = T;
Tpred.yhat_ml     = yhat_all;
Tpred.yhat_ml_cal = yhat_cal_all;

outCsv = fullfile(repoRoot, "results", "predictions_per_window.csv");
writetable(Tpred, outCsv);
fprintf("Saved per-window predictions: %s\n", outCsv);
