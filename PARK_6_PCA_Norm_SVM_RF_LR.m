%% Classification PD and HC based on individual natural frequency maps for Parkinson's study (Arana et al., 2026)-PCA and Normalization
% This script is employed for internal cross validation of the model under
% the PCA and normalization scheme.

addpath(genpath('Z:\LYDIA\Parkinson_github'));

load('G:\Classification\group\split.mat');% load the split of PARK_5!!


maxNumCompThreads(3);


X = fnat_all_insample';   % 100 x 1921 matrix of natural frequencies for all subjects
Y = double(labels_insample');  % 100 x 1 vector of labels PD (1) and HC (0)


kfold = 10;
n_reps = 10;
pca_components_list = [10, 40, 89];

results_svm = zeros(length(pca_components_list), n_reps);
results_rf = zeros(length(pca_components_list), n_reps);
results_fl = zeros(length(pca_components_list), n_reps);


for c = 1:length(pca_components_list)
    n_components = pca_components_list(c);
    fprintf('Evaluating PCA with %d components with normalization...\n', n_components);

    acc_svm_folds = zeros(kfold, n_reps);
    sens_svm_folds = zeros(kfold, n_reps);
    spec_svm_folds = zeros(kfold, n_reps);

    acc_rf_folds = zeros(kfold, n_reps);
    sens_rf_folds = zeros(kfold, n_reps);
    spec_rf_folds = zeros(kfold, n_reps);

    acc_fl_folds = zeros(kfold, n_reps);
    sens_fl_folds = zeros(kfold, n_reps);
    spec_fl_folds = zeros(kfold, n_reps);

    labels_all = cell(n_reps * kfold, 1);
    scores_svm_all = cell(n_reps * kfold, 1);
    scores_rf_all = cell(n_reps * kfold, 1);
    scores_fl_all = cell(n_reps * kfold, 1);

    for rep = 1:n_reps
        rng(rep);
        cv = cvpartition(Y, 'KFold', kfold);

        for fold = 1:kfold
            train_idx = training(cv, fold);
            test_idx = test(cv, fold);

            X_train = X(train_idx, :);
            Y_train = Y(train_idx);
            X_test = X(test_idx, :);
            Y_test = Y(test_idx);

            % --- Normalization ---
            mu = mean(X_train, 1);
            sigma = std(X_train, 0, 1);
            sigma(sigma == 0) = 1;
            X_train_norm = (X_train - mu) ./ sigma;
            X_test_norm  = (X_test - mu) ./ sigma;

            % PCA 
            [coeff, ~] = pca(X_train_norm);
            n_comp_use = min(n_components, size(coeff,2));
            fprintf('  Using %d comps from %d available\n', n_comp_use, size(coeff,2));

            coeff_reduced = coeff(:, 1:n_comp_use);

            X_train_pca = X_train_norm * coeff_reduced;
            X_test_pca  = X_test_norm  * coeff_reduced;

            % SVM
            model_linear = fitcsvm(X_train_pca, Y_train, 'KernelFunction', 'linear', 'BoxConstraint', 1);
            [pred_linear, score_linear] = predict(model_linear, X_test_pca);
            acc_svm_folds(fold, rep) = mean(pred_linear == Y_test);
            [sens, spec] = calcSensSpec(Y_test, pred_linear);
            sens_svm_folds(fold, rep) = sens;
            spec_svm_folds(fold, rep) = spec;
           
            scores_svm_all{(rep-1)*kfold + fold} = score_linear(:,2);

            % RF
            model_rf = TreeBagger(100, X_train_pca, categorical(Y_train), 'Method', 'classification', 'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');
            [pred_rf_cell, score_rf] = predict(model_rf, X_test_pca);
            pred_rf = double(categorical(pred_rf_cell)) - 1;
            acc_rf_folds(fold, rep) = mean(pred_rf == Y_test);
            [sens, spec] = calcSensSpec(Y_test, pred_rf);
            sens_rf_folds(fold, rep) = sens;
            spec_rf_folds(fold, rep) = spec;
            scores_rf_all{(rep-1)*kfold + fold} = score_rf(:,2);

            % LR
            model_fl = fitclinear(X_train_pca, Y_train, 'Learner', 'logistic', 'Regularization', 'ridge');
            [pred_fl, score_fl] = predict(model_fl, X_test_pca);
            acc_fl_folds(fold, rep) = mean(pred_fl == Y_test);
            [sens, spec] = calcSensSpec(Y_test, pred_fl);
            sens_fl_folds(fold, rep) = sens;
            spec_fl_folds(fold, rep) = spec;
            
            scores_fl_all{(rep-1)*kfold + fold} = score_fl(:,2);

            labels_all{(rep-1)*kfold + fold} = Y_test;
        end

        results_svm(c, rep) = mean(acc_svm_folds(:, rep));
        results_rf(c, rep) = mean(acc_rf_folds(:, rep));
        results_fl(c, rep) = mean(acc_fl_folds(:, rep));
    end

    % Metrics
    fprintf('PCA %d componentes:\n', n_components);
    fprintf('  SVM Accuracy:  %.2f%% (± %.2f)\n', mean(results_svm(c,:))*100, std(results_svm(c,:))*100);
    fprintf('  RF Accuracy:  %.2f%% (± %.2f)\n', mean(results_rf(c,:))*100, std(results_rf(c,:))*100);
    fprintf('  LR Accuracy:  %.2f%% (± %.2f)\n', mean(results_fl(c,:))*100, std(results_fl(c,:))*100);
    fprintf('  SVM Sens: %.2f%%, Spec: %.2f%%\n', mean(sens_svm_folds(:))*100, mean(spec_svm_folds(:))*100);
    fprintf('  RF Sens: %.2f%%, Spec: %.2f%%\n', mean(sens_rf_folds(:))*100, mean(spec_rf_folds(:))*100);
    fprintf('  LR Sens: %.2f%%, Spec: %.2f%%\n', mean(sens_fl_folds(:))*100, mean(spec_fl_folds(:))*100);

    % AUC
    all_labels = vertcat(labels_all{:});
    all_scores_svm = vertcat(scores_svm_all{:});
    all_scores_rf = vertcat(scores_rf_all{:});
    all_scores_fl = vertcat(scores_fl_all{:});

 

    rng(12345);
    nBoots = 1000;
    posClass = 1;
    [auc_svm, ci_svm] = auc_bootstrap(all_labels, all_scores_svm, posClass, nBoots);
    [auc_rf, ci_rf] = auc_bootstrap(all_labels, all_scores_rf, posClass, nBoots);
    [auc_fl, ci_fl] = auc_bootstrap(all_labels, all_scores_fl, posClass, nBoots);

    fprintf('  SVM AUC = %.3f (95%% CI: %.3f - %.3f)\n', auc_svm, ci_svm(1), ci_svm(2));
    fprintf('  RF AUC = %.3f (95%% CI: %.3f - %.3f)\n', auc_rf, ci_rf(1), ci_rf(2));
    fprintf('  LR AUC = %.3f (95%% CI: %.3f - %.3f)\n\n', auc_fl, ci_fl(1), ci_fl(2));

    % ROC plot
    rm_svm = rocmetrics(all_labels, all_scores_svm, 1, 'NumBootstraps', 1000);
    rm_rf = rocmetrics(all_labels, all_scores_rf, 1, 'NumBootstraps', 1000);
    rm_fl = rocmetrics(all_labels, all_scores_fl, 1, 'NumBootstraps', 1000);

    colors = [0 0.4470 0.7410; 0.4660 0.6740 0.1880; 0.8500 0.3250 0.0980];
    figure('Name', sprintf('ROC PCA %d', n_components),'Position',[100 100 500 500]);
    hold on;
    plot(rm_svm, 'ShowConfidenceIntervals', true, 'Color', colors(1,:), 'LineWidth', 2);
    plot(rm_rf, 'ShowConfidenceIntervals', true, 'Color', colors(2,:), 'LineWidth', 2);
    plot(rm_fl, 'ShowConfidenceIntervals', true, 'Color', colors(3,:), 'LineWidth', 2);
    plot([0 1], [0 1], 'k--');

    % --- Remove default MATLAB dots from the operating point plot---
    hScatter = findobj(gca, 'Type', 'Scatter'); 
    delete(hScatter); 

    xlabel('False positive rate'); ylabel('True positive rate');
    % legend({'SVM', 'RF', 'LR'}, 'Location', 'SouthEast');
    xticks(0:0.2:1);% xticklabels(0:0.2:1);
    yticks(0:0.2:1); %yticklabels(0:0.2:1);
    set(gca, 'FontSize', 12);
    box on
    hold off;

    % Accuracy violins
    figure('Name', sprintf('Accuracy (%) PCA %d', n_components),'Position',[100 100 500 500]);
    acc_data = {reshape(acc_svm_folds*100,[],1), reshape(acc_rf_folds*100,[],1), reshape(acc_fl_folds*100,[],1)};

    for i = 1:3
        fprintf('Model %d: min=%.2f, max=%.2f\n', i, min(acc_data{i}), max(acc_data{i}));
        if any(isnan(acc_data{i}))
            warning('Model %d has NaN in accuracy', i);
        end
    end


    posiciones = 1:3; width = 0.3; hold on
    for i = 1:3

        d = acc_data{i};          
        pos = posiciones(i);      
        color = colors(i,:);      

        [f, yi] = ksdensity(d, 'Support', 'positive'); f = f / max(f) * width;
        fill([pos - f, pos * ones(size(f))], [yi, fliplr(yi)], color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        med = median(d); f_med = interp1(yi, f, med, 'linear', 0);
        plot([pos - f_med, pos], [med, med], 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
        mu = mean(d); f_mu = interp1(yi, f, mu, 'linear', 0);
        plot([pos - f_mu, pos], [mu, mu], 'k-', 'LineWidth', 1.5);
        scatter((rand(size(d))*0.4 - 0.1)*width + pos + width*0.3, d, 15, 'filled', 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.5);
    end
    xlim([0.5 3.5]); ylim([20 110]); yline(50, 'k--');
    ylim([20 110])
    yticks([40,60,80,100])
    set(gca, 'XTick', posiciones, 'XTickLabel', {'SVM', 'RF', 'LR'}, 'FontSize', 12);
    ylabel('Accuracy (%)');
    box on

    hold off


    nBoots = 1000;
    rng(12345);


    % Accuracy SVM
    boot_acc_svm = bootstrp(nBoots, @mean, results_svm(c,:) );
    acc_svm_mean = mean(results_svm(c,:));
    acc_svm_ci = prctile(boot_acc_svm, [2.5 97.5]);

    % Accuracy RF
    boot_acc_rf = bootstrp(nBoots, @mean, results_rf(c,:) );
    acc_rf_mean = mean(results_rf(c,:));
    acc_rf_ci = prctile(boot_acc_rf, [2.5 97.5]);

    % Accuracy LR
    boot_acc_fl = bootstrp(nBoots, @mean, results_fl(c,:) );
    acc_fl_mean = mean(results_fl(c,:));
    acc_fl_ci = prctile(boot_acc_fl, [2.5 97.5]);

    fprintf('SVM Accuracy: %.2f%% (95%% CI: %.2f - %.2f)\n', acc_svm_mean*100, acc_svm_ci(1)*100, acc_svm_ci(2)*100);
    fprintf('RF Accuracy: %.2f%% (95%% CI: %.2f - %.2f)\n', acc_rf_mean*100, acc_rf_ci(1)*100, acc_rf_ci(2)*100);
    fprintf('LR Accuracy: %.2f%% (95%% CI: %.2f - %.2f)\n', acc_fl_mean*100, acc_fl_ci(1)*100, acc_fl_ci(2)*100);


    % SVM Sens
    boot_sens_svm = bootstrp(nBoots, @mean, sens_svm_folds(:));
    sens_svm_mean = mean(sens_svm_folds(:));
    sens_svm_ci = prctile(boot_sens_svm, [2.5 97.5]);

    % SVM Spec
    boot_spec_svm = bootstrp(nBoots, @mean, spec_svm_folds(:));
    spec_svm_mean = mean(spec_svm_folds(:));
    spec_svm_ci = prctile(boot_spec_svm, [2.5 97.5]);

    % RF Sens
    boot_sens_rf = bootstrp(nBoots, @mean, sens_rf_folds(:));
    sens_rf_mean = mean(sens_rf_folds(:));
    sens_rf_ci = prctile(boot_sens_rf, [2.5 97.5]);

    % RF Spec
    boot_spec_rf = bootstrp(nBoots, @mean, spec_rf_folds(:));
    spec_rf_mean = mean(spec_rf_folds(:));
    spec_rf_ci = prctile(boot_spec_rf, [2.5 97.5]);

    % LR Sens
    boot_sens_fl = bootstrp(nBoots, @mean, sens_fl_folds(:));
    sens_fl_mean = mean(sens_fl_folds(:));
    sens_fl_ci = prctile(boot_sens_fl, [2.5 97.5]);

    % LR Spec
    boot_spec_fl = bootstrp(nBoots, @mean, spec_fl_folds(:));
    spec_fl_mean = mean(spec_fl_folds(:));
    spec_fl_ci = prctile(boot_spec_fl, [2.5 97.5]);

    fprintf('SVM Sens: %.2f%% (95%% CI: %.2f - %.2f)\n', sens_svm_mean*100, sens_svm_ci(1)*100, sens_svm_ci(2)*100);
    fprintf('SVM Spec: %.2f%% (95%% CI: %.2f - %.2f)\n', spec_svm_mean*100, spec_svm_ci(1)*100, spec_svm_ci(2)*100);
    fprintf('RF  Sens: %.2f%% (95%% CI: %.2f - %.2f)\n', sens_rf_mean*100, sens_rf_ci(1)*100, sens_rf_ci(2)*100);
    fprintf('RF  Spec: %.2f%% (95%% CI: %.2f - %.2f)\n', spec_rf_mean*100, spec_rf_ci(1)*100, spec_rf_ci(2)*100);
    fprintf('LR  Sens: %.2f%% (95%% CI: %.2f - %.2f)\n', sens_fl_mean*100, sens_fl_ci(1)*100, sens_fl_ci(2)*100);
    fprintf('LR  Spec: %.2f%% (95%% CI: %.2f - %.2f)\n', spec_fl_mean*100, spec_fl_ci(1)*100, spec_fl_ci(2)*100);


end




