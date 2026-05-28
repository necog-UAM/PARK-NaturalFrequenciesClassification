% Boostrap function for CI
function CI = calcBootstrapCI(data, nBoots, confLevel)
    bootStat = bootstrp(nBoots, @(x) mean(x), data);
    alpha = 100 - confLevel;
    CI = prctile(bootStat, [alpha/2, 100-alpha/2]);
end
