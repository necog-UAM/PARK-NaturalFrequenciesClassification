%% Classification PD and HC based on individual natural frequency maps for Parkinson's study (Arana et al., 2026)

% Training once with best model (no normalization no PCA )

addpath(genpath('Z:\LYDIA\Parkinson_github'));
load('G:\Classification\group\split.mat');% load the split of PARK_5!!


rng(42); 

% In sample data
X_in = fnat_all_insample';    % 100 x 1921
Y_in = double(labels_insample');

X_in_use = X_in;
model_final = fitcsvm(X_in_use, Y_in, ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', 1, ...
    'Standardize', false);  

% Haufe weights
cov_in = cov(X_in_use);
beta_haufe_original = cov_in * model_final.Beta;
beta_original_space = model_final.Beta;


%% Save final model
cd('G:\Classification\results')
save('model_in_sample_noNorm_noPCA.mat', ...
     'model_final', 'beta_haufe_original', 'beta_original_space');


%% Evaluation to test training ok

[pred_in, score_in] = predict(model_final, X_in_use);

[sen_in, spec_in] = calcSensSpec(Y_in, pred_in);
acc_in = mean(pred_in == Y_in);

fprintf('Accuracy: %.2f%%\nSens: %.2f%%\nSpec: %.2f%%\n', ...
        acc_in*100, sen_in*100, spec_in*100);

% Conf matrix in-sample
figure;
confusionchart(Y_in, pred_in);
title('Confusion Matrix - In-sample (no norm no PCA)');

% ROC in-sample
posClass = 1;
rm_in = rocmetrics(Y_in, score_in(:,2), posClass);

figure;
plot(rm_in, 'ShowConfidenceIntervals', true, ...
    'Color', [0 0.4470 0.7410], 'LineWidth', 2);
hold on; plot([0 1],[0 1],'k--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC - In-sample (AUC = %.3f)', rm_in.AUC));
hScatter = findobj(gca, 'Type', 'Scatter'); delete(hScatter);
legend('off');
hold off;



%% Evaluation in hold-out set

load('G:\Classification\results\model_in_sample_noNorm_noPCA.mat', ...
     'model_final');

% Hold-out set
X_out = fnat_all_outsample';   % N x 1921
Y_out = double(labels_outsample');

X_out_use = X_out;

% Pred
[pred_out, score_out] = predict(model_final, X_out_use);

% Metrics
[sen_out, spec_out] = calcSensSpec(Y_out, pred_out);
acc_out = mean(pred_out == Y_out);

fprintf('\nResults hold-out test (no normalization no PCA):\n');
fprintf('Accuracy: %.2f%%\nSens: %.2f%%\nSpec: %.2f%%\n', ...
        acc_out*100, sen_out*100, spec_out*100);

% Confusion matrix out-sample
figure;
confusionchart(Y_out, pred_out);
title('Confusion Matrix - Out-sample (no norm no PCA)');

% Curve ROC out-sample
posClass = 1;
rm_out = rocmetrics(Y_out, score_out(:,2), posClass);
auc_out = rm_out.AUC;

fprintf('\nResults out-sample (no norm no PCA):\n');
fprintf('Accuracy: %.2f%%\nSensibility: %.2f%%\nSpecificity: %.2f%%\nAUC: %.3f\n', ...
        acc_out*100, sen_out*100, spec_out*100, auc_out);

figure;
plot(rm_out, 'ShowConfidenceIntervals', true, ...
    'Color', [0.2 0.2 0.2], 'LineWidth', 2);
hold on; plot([0 1],[0 1],'k--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC - Out-sample (AUC = %.3f)', rm_out.AUC));
hScatter = findobj(gca, 'Type', 'Scatter'); delete(hScatter);
legend('off');
hold off;


% Conf matrix hold-out test
C = confusionmat(Y_out, pred_out);      
rowSums = sum(C,2);                     
P = 100 * (C ./ max(rowSums,1));                 

fprintf('\n--- CONF MATRIX HOLD-OUT TEST ---\n');
fprintf('                Pred: 0        Pred: 1\n');
fprintf('True: 0      %3d (%.1f%%)   %3d (%.1f%%)\n', ...
        C(1,1), P(1,1), C(1,2), P(1,2));
fprintf('True: 1      %3d (%.1f%%)   %3d (%.1f%%)\n', ...
        C(2,1), P(2,1), C(2,2), P(2,2));
fprintf('----------------------------------------\n');

TablaConfusion = table(...
    [C(1,1); C(2,1)], [C(1,2); C(2,2)], ...
    [P(1,1); P(2,1)], [P(1,2); P(2,2)], ...
    'VariableNames', {'Pred_0', 'Pred_1', 'Perc_0', 'Perc_1'}, ...
    'RowNames', {'True_0', 'True_1'});

disp(TablaConfusion);


