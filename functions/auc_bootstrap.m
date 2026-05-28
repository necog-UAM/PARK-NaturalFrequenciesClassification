function [auc, ci] = auc_bootstrap(labels, scores, posClass, nBoots)
    rng(1); 
    n = length(labels);
    aucs = zeros(nBoots,1);

    for b = 1:nBoots
        idx = randsample(n, n, true); 
        boot_labels = labels(idx);
        boot_scores = scores(idx);
        [X, Y, T, AUC] = perfcurve(boot_labels, boot_scores, posClass);
        aucs(b) = AUC;
    end

    auc = mean(aucs);
    ci = prctile(aucs, [2.5 97.5]);
end
