%% K-means of EEG group for Parkinson's study (Arana et al., 2026)

% This script is employed for obtention of the power spectra centroids or clusters.
% 300 power spectra randomly selected per voxel and subject was introduced.
% The main output is the proportion of power spectra per cluster and the peak value of the centroid saved in kmeans_10mm_Nk25_150tr_step200ms.
% The input are the power spectra per subject (freq_allvox_10mm_steps200ms).
% Computation of K-means clustering requires near 32 Gb of RAM for the complete sample.

clear all
close all
clc

restoredefaultpath
addpath ('Z:\Toolbox\fieldtrip-20230118');
ft_defaults
addpath(genpath('Z:\LYDIA\Parkinson_github'));


maxNumCompThreads(3);


dpath = 'G:\Classification\';
outpath = 'G:\Classification\kmeans\';



subs = {'sub-88020741'	'sub-88019749'	'sub-828'	'sub-810'	'sub-40021'	'sub-40005'	'sub-40016' 'sub-40007'	'sub-40008'	'sub-40010'	'sub-40012'	'sub-40017'	'sub-40018'	'sub-40019'	'sub-40020'	'sub-40022'	'sub-40023'	'sub-40024'	'sub-40025'	'sub-40026'	'sub-801'	'sub-802'	'sub-804'	'sub-805'	'sub-806'	'sub-807'	'sub-808'	'sub-809'	'sub-811'	'sub-813'	'sub-814'	'sub-815'	'sub-816'	'sub-817'	'sub-818'	'sub-819'	'sub-820'	'sub-821'	'sub-822'	'sub-823'	'sub-824'	'sub-825'	'sub-826'	'sub-827'	'sub-829'	'sub-88017953'	'sub-88017997'	'sub-88018225'	'sub-88018401'	'sub-88018585'	'sub-88019077'	'sub-88019657'	'sub-88019661'	'sub-88019705'	'sub-88019797'	'sub-88019885'	'sub-88020649' ...
'sub-914'	'sub-912'	'sub-8070'	'sub-100011'	'sub-100038'	'sub-100030'	'sub-905' 'sub-10001'	'sub-100010'	'sub-100012'	'sub-100014'	'sub-100015'	'sub-100016'	'sub-100018'	'sub-100020'	'sub-100021'	'sub-100026'	'sub-100028'	'sub-100031'	'sub-100034'	'sub-100037'	'sub-10005'	'sub-10006'	'sub-10007'	'sub-10008'	'sub-8060'	'sub-88041893'	'sub-88041941'	'sub-88049585'	'sub-88053453'	'sub-88053545'	'sub-88055121'	'sub-88055301'	'sub-88068841'	'sub-88075053'	'sub-890'	'sub-891'	'sub-892'	'sub-893'	'sub-894'	'sub-895'	'sub-896'	'sub-897'	'sub-898'	'sub-899'	'sub-900'	'sub-901'	'sub-902'	'sub-903'	'sub-904'	'sub-906'	'sub-907'	'sub-908'	'sub-909'	'sub-910'	'sub-911'	'sub-913'};

sess = {'ses-1'};


% 2.1. Preparation of power spectra matrix for subsequent K-means clustering (output: powsptot ksub kvox ktrial)
% 2.2. K-means clustering (output: idx C sumd D)
% 2.3. Proportion of power spectra categorized in each cluster (output: propk)


%% 2.1. Preparation of power spectra matrix for subsequent K-means clustering

powsptot = [];
ksub     = [];
kvox     = [];
ktrial   = [];
rng('default')
rng('shuffle')


for sub=1:length(subs)
    route = fullfile([dpath, subs{sub} '\' sess{1}]); % reconstruct route
    cd(route)
    load freq_allvox_10mm_steps200ms

    ct = 1;
    valid_tr = [];

    for tr = 1:size(powsp,3)
        if sum(sum(isnan(powsp(:,:,tr)))) == 0
            valid_tr(ct) = tr;
            ct = ct+1;
        end
    end

    Ntr      = 300; % number of spectra per subject into kmeans

    rndtr   = randperm(length(valid_tr),Ntr); % caution if Ntr > length(valid<-tr)
    select_tr = valid_tr(rndtr);
    powsp2  = powsp(:,:,select_tr);
    ct = 1;
    powsp3  = single(zeros(size(powsp2,1)*size(powsp2,3),size(powsp2,2)));
    ksub2   = [];
    kvox2   = [];
    ktrial2 = [];
    for i = 1:size(powsp2,1)
        for tr = 1:size(powsp2,3)
            powsp3(ct,:) = powsp2(i,:,tr);
            ksub2(ct)    = sub;
            kvox2(ct)    = i;
            ktrial2(ct)  = select_tr(tr);
            ct = ct+1;
        end
    end

    powsptot = [powsptot; powsp3];         % concatenation of power spectra
    ksub     = [ksub ksub2];               % keep track of subject, voxel and trial of each power spectrum
    kvox     = [kvox kvox2];
    ktrial   = [ktrial ktrial2];
end

clear freq powsp

bl       = sum(powsptot,2);                                % save kmeans_10mm_baseline bl
powsptot = powsptot./repmat(bl,[1 size(powsptot,2)]);      % compute relative power to correct the center of the head bias


cd(outpath);
save('kmeans_10mm_powsp_200ms_EEG.mat', 'powsptot', 'ksub' , 'kvox' , 'ktrial', '-v7.3');


%% 2.2. K-means clustering
% Computation of K-means clustering
% Proportion of power spectra (out of 300) categorized in each cluster

% Read matrix with the power spectra of all subjects, time segments and voxels
Nk = 25;


% K-means clustering
[idx,C,sumd,D] = kmeans(powsptot,Nk,'Distance','cosine','Display','iter','Replicates',5,'MaxIter',200);


%% 2.3. Proportion of power spectra categorized in each cluster
% This computation is done for each subject and voxel (for each
% subject and voxel, the sum across clusters is equal to 1)

Nvox = 1921; 
Ntr = 300;
Nsub=numel(subs);

propk = NaN(Nk,Nvox,Nsub);

for k = 1:Nk
    disp(['Cluster ' num2str(k) '/' num2str(Nk)])
    ctk = find(idx==k);
    ctksub = ksub(ctk)';
    ctkvox = kvox(ctk)';
    ctksubfilt = ctksub==[1:Nsub];
    ctkvoxfilt = ctkvox==[1:Nvox];
    for s=1:Nsub
        propk(k,:,s)=sum(ctksubfilt(:,s) & ctkvoxfilt)./Ntr;
    end
end


cd(outpath);
save('kmeans_10mm_Nk25_300tr_step200ms_EEG.mat', 'idx', 'C', 'sumd', 'D', 'Nvox', 'propk', '-v7.3');

