%% This script is to obtain samples and matrices of natural frequencies for Parkinson's study (Arana et al., 2026)
% To obtain the subamples and matrices of natural frequencies for subsequent classification pruposes 
% Outputs: fnat_all matrices of natural frequencies of all subjects, with corresponding diagnostic labels and subjects
% insample: the one used for training and cross validation; outsample: the one used for hold out test)


% === INPUTS === 
% The order of subjects in each string (subs_pd and subs_hc) corresponds to the HC-PD gender and age pairs
% db_ is the vector indicating the database of each subj

subs_pd = {'sub-801'	'sub-804'	'sub-810'	'sub-809'	'sub-88019797'	'sub-88019077'	'sub-811'	'sub-40019'	'sub-808'	'sub-817'	'sub-40020'	'sub-40018'	'sub-40005'	'sub-814'	'sub-822'	'sub-40025'	'sub-40021'	'sub-88018225'	'sub-40012'	'sub-40017'	'sub-40026'	'sub-806'	'sub-828'	'sub-40023'	'sub-40024'	'sub-821'	'sub-40022'	'sub-88019705'	'sub-88020649'	'sub-88018585'	'sub-815'	'sub-816'	'sub-818'	'sub-819'	'sub-823'	'sub-807'	'sub-88017953'	'sub-826'	'sub-40010'	'sub-40007'	'sub-88019661'	'sub-88019657'	'sub-813'	'sub-825'	'sub-824'	'sub-805'	'sub-827'	'sub-40016'	'sub-88019885'	'sub-829'	'sub-802'	'sub-88017997'	'sub-820'	'sub-88018401'	'sub-88019749'	'sub-40008'	'sub-88020741'};																																																					

gen_pd = {'M'	'M'	'M'	'M'	'F'	'F'	'M'	'F'	'M'	'M'	'F'	'F'	'M'	'M'	'M'	'F'	'F'	'M'	'M'	'F'	'F'	'F'	'F'	'M'	'M'	'F'	'M'	'F'	'M'	'M'	'M'	'F'	'F'	'M'	'F'	'F'	'M'	'M'	'M'	'M'	'M'	'M'	'M'	'F'	'M'	'M'	'M'	'F'	'M'	'F'	'F'	'M'	'M'	'M'	'M'	'M'	'F'
};

db_pd = [2	2	2	2	1	1	2	3	2	2	3	3	3	2	2	3	3	1	3	3	3	2	2	3	3	2	3	1	1	1	2	2	2	2	2	2	1	2	3	3	1	1	2	2	2	2	2	3	1	2	2	1	2	1	1	3	1		
];

subs_healthy = {'sub-10001'	'sub-100010'	'sub-100011'	'sub-100012'	'sub-100014'	'sub-100015'	'sub-100016'	'sub-100018'	'sub-100020'	'sub-100021'	'sub-100026'	'sub-100028'	'sub-100030'	'sub-100031'	'sub-100034'	'sub-100037'	'sub-100038'	'sub-10005'	'sub-10006'	'sub-10007'	'sub-10008'	'sub-8060'	'sub-8070'	'sub-88041893'	'sub-88041941'	'sub-88049585'	'sub-88053453'	'sub-88053545'	'sub-88055121'	'sub-88055301'	'sub-88068841'	'sub-88075053'	'sub-890'	'sub-891'	'sub-892'	'sub-893'	'sub-894'	'sub-895'	'sub-896'	'sub-897'	'sub-898'	'sub-899'	'sub-900'	'sub-901'	'sub-902'	'sub-903'	'sub-904'	'sub-905'	'sub-906'	'sub-907'	'sub-908'	'sub-909'	'sub-910'	'sub-911'	'sub-912'	'sub-913'	'sub-914'
};

gen_healthy = {'M'	'M'	'M'	'M'	'F'	'F'	'M'	'F'	'M'	'M'	'F'	'F'	'M'	'M'	'M'	'F'	'F'	'M'	'M'	'F'	'F'	'F'	'F'	'M'	'M'	'F'	'M'	'F'	'M'	'M'	'M'	'F'	'F'	'M'	'F'	'F'	'M'	'M'	'M'	'M'	'M'	'M'	'M'	'F'	'M'	'M'	'M'	'F'	'M'	'F'	'F'	'M'	'M'	'M'	'M'	'M'	'F'
};

db_healthy = [3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	2	2	1	1	1	1	1	1	1	1	1	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2		
];



Nvox = 1921;
sess = {'ses-1'};
dpath = 'G:\Classification';

% Total number of PD/HC
n_total_pd = numel(subs_pd);

% --- PD by gender and database ---
pd_F_idx = find(strcmp(gen_pd, 'F'));
pd_M_idx = find(strcmp(gen_pd, 'M'));

% Index by gender and database
bases = 1:3;
pd_base_gender_idx = struct();
for b = bases
    pd_base_gender_idx(b).F = intersect(find(db_pd == b), pd_F_idx);
    pd_base_gender_idx(b).M = intersect(find(db_pd == b), pd_M_idx);
end

% --- Selection of 7 PD ---
out_pd_idx = [];

% At least 1 PD of each gender and database if possible
for b = bases
    if ~isempty(pd_base_gender_idx(b).F)
        out_pd_idx(end+1) = randsample(pd_base_gender_idx(b).F, 1);
    end
    if ~isempty(pd_base_gender_idx(b).M)
        out_pd_idx(end+1) = randsample(pd_base_gender_idx(b).M, 1);
    end
end

% Complete until 7 PD 
remaining_needed = 7 - numel(out_pd_idx);
all_remaining_idx = setdiff(1:n_total_pd, out_pd_idx);
if remaining_needed > 0
    out_pd_idx = [out_pd_idx, randsample(all_remaining_idx, remaining_needed)];
end

% --- Obtention of PD and corresponding HC pairs ---
out_pd = subs_pd(out_pd_idx);
out_healthy = subs_healthy(out_pd_idx);

outsample_subs = [out_healthy, out_pd];

% --- Insample = all minus outsample ---
insample_subs = setdiff([subs_healthy, subs_pd], outsample_subs);

% --- Labels PD/HC ---
labels_insample = [false(1,numel(insample_subs))]; 
labels_outsample = [false(1,numel(outsample_subs))];

% Asign labels
for s = 1:numel(insample_subs)
    if ismember(insample_subs{s}, subs_pd)
        labels_insample(s) = true;
    end
end
for s = 1:numel(outsample_subs)
    if ismember(outsample_subs{s}, subs_pd)
        labels_outsample(s) = true;
    end
end

% --- Load fnat for insample ---
fnat_all_insample = single(NaN(Nvox, numel(insample_subs)));
for s = 1:numel(insample_subs)
    sub = insample_subs{s};
    file_path = fullfile(dpath, sub, sess{1}, 'singlesub_fnat_steps200ms.mat');
    tmp = load(file_path, 'fnat');
    fnat_all_insample(:, s) = tmp.fnat.fnat(:);
end

% --- Load fnat for outsample ---
fnat_all_outsample = single(NaN(Nvox, numel(outsample_subs)));
for s = 1:numel(outsample_subs)
    sub = outsample_subs{s};
    file_path = fullfile(dpath, sub, sess{1}, 'singlesub_fnat_steps200ms.mat');
    tmp = load(file_path, 'fnat');
    fnat_all_outsample(:, s) = tmp.fnat.fnat(:);
end

cd('G:\Classification\group');

save('split.mat', 'fnat_all_insample', 'fnat_all_outsample', ...
    'labels_insample', 'labels_outsample', 'insample_subs', 'outsample_subs');

% --- Summary ---
n_base_out = zeros(1,3);
gen_out = zeros(1,2); % [F, M]
for s = 1:numel(outsample_subs)
    sub = outsample_subs{s};
    idx_pd = find(strcmp(subs_pd, sub));
    if ~isempty(idx_pd)
        n_base_out(db_pd(idx_pd)) = n_base_out(db_pd(idx_pd)) + 1;
        if strcmp(gen_pd{idx_pd}, 'F')
            gen_out(1) = gen_out(1) + 1;
        else
            gen_out(2) = gen_out(2) + 1;
        end
    else
        idx_h = find(strcmp(subs_healthy, sub));
        if ~isempty(idx_h)
            n_base_out(db_healthy(idx_h)) = n_base_out(db_healthy(idx_h)) + 1;
            if strcmp(gen_healthy{idx_h}, 'F')
                gen_out(1) = gen_out(1) + 1;
            else
                gen_out(2) = gen_out(2) + 1;
            end
        end
    end
end

fprintf(' - Insample: %d subs\n', numel(insample_subs));
fprintf(' - Outsample: %d subs (7 Healthy, 7 PD)\n', numel(outsample_subs));
fprintf('   Databases in outsample: base1=%d, base2=%d, base3=%d\n', n_base_out(1), n_base_out(2), n_base_out(3));
fprintf('   Gender in outsample: %d F, %d M\n', gen_out(1), gen_out(2));
