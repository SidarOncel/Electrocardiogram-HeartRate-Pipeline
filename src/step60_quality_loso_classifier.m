%% step60_quality_loso_classifier.m — LOSO 3-class quality classifier + confusion matrix
clear; close all; clc;

repoRoot = fileparts(fileparts(mfilename("fullpath")));
load(fullfile(repoRoot, "data_processed", "features_10s.mat"), "T");

% Ensure quality is a categorical with the right order
classes = ["clean","mild","severe"];
T.quality = categorical(string(T.quality), classes, "Ordinal", true);

% Use signal-only features for quality (do NOT use hr_gt)
featNames = ["rms_filt","std_filt","kurt_filt","bp_low","bp_high","bp_ratio","spec_ent"];

subjects = unique(T.subject);

C_total = zeros(numel(classes), numel(classes));  % rows=true, cols=pred

for s = subjects'
    isTest = (T.subject == s);
    Tr = T(~isTest,:);
    Te = T(isTest,:);

    Xtr = Tr{:, featNames};
    ytr = Tr.quality;

    Xte = Te{:, featNames};
    yte = Te.quality;

    % Impute NaNs using training medians
    med = nanmedian(Xtr,1);
    for j=1:size(Xtr,2)
        Xtr(isnan(Xtr(:,j)),j) = med(j);
        Xte(isnan(Xte(:,j)),j) = med(j);
    end

    % Standardize using training stats
    [XtrZ, mu, sig] = zscore(Xtr);
    sig(sig==0) = 1;
    XteZ = (Xte - mu) ./ sig;

    % Train linear ECOC classifier (fast + built-in)
    mdl = fitcecoc(XtrZ, ytr, "Learners", "linear");

    yhat = predict(mdl, XteZ);

    % Confusion matrix aligned to fixed class order
    C = confusionmat(yte, yhat, "Order", categorical(classes, classes, "Ordinal", true));
    C_total = C_total + C;
end

% Display
disp("Overall confusion matrix (rows=true, cols=pred):");
Ctab = array2table(C_total, "VariableNames", classes, "RowNames", classes);
disp(Ctab);

% Recall (sensitivity) per class
recall = diag(C_total) ./ max(1, sum(C_total,2));
recall = recall(:);  % make it a column
recallTab = table(classes(:), recall, 'VariableNames', {'class','recall'});
disp("Per-class recall (sensitivity):");
disp(recallTab);

% Save outputs
outDir = fullfile(repoRoot, "results");
if ~isfolder(outDir), mkdir(outDir); end

writetable(Ctab, fullfile(outDir, "quality_confusion_matrix.csv"), "WriteRowNames", true);
writetable(recallTab, fullfile(outDir, "quality_recall.csv"));

fprintf("Saved: results/quality_confusion_matrix.csv and results/quality_recall.csv\n");

% --- Plot confusion matrix (works on older MATLAB versions) ---
figure;
imagesc(C_total);
colorbar;
axis square;

set(gca, ...
    'XTick', 1:numel(classes), 'XTickLabel', classes, ...
    'YTick', 1:numel(classes), 'YTickLabel', classes);

xlabel("Predicted");
ylabel("True");
title("Quality Classification (LOSO) — Confusion Matrix");

% Optional: write counts on cells
for i = 1:numel(classes)
    for j = 1:numel(classes)
        text(j, i, num2str(C_total(i,j)), ...
            'HorizontalAlignment','center', ...
            'Color','w', 'FontWeight','bold');
    end
end
