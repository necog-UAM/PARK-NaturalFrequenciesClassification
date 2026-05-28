% Function to calculate sensibility and specificity
function [sens, spec] = calcSensSpec(y_true, y_pred)
    TP = sum((y_true == 1) & (y_pred == 1));
    TN = sum((y_true == 0) & (y_pred == 0));
    FP = sum((y_true == 0) & (y_pred == 1));
    FN = sum((y_true == 1) & (y_pred == 0));
    
    sens = TP / (TP + FN); % sens (recall)
    spec = TN / (TN + FP); % spec
    
    if isnan(sens)
        sens = 0;
    end
    if isnan(spec)
        spec = 0;
    end
end



