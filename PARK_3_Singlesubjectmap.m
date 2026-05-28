%% Single subject maps for Parkinson's study (Arana et al., 2026)
% This script is employed for the single subject mapping of natural frequencies.
% The output is a natural frequencies vector of 1921 for each session (singlesub_fnat_steps200m) and a figure of the map of each session.
% The inputs are freq_allvox_10mm_steps200ms, kmeans_10mm_Nk25_300tr_step200ms_EEG and sources for plotting.

clear all
close all
clc

restoredefaultpath
addpath ('Z:\Toolbox\fieldtrip-20230118');
ft_defaults
addpath(genpath('Z:\LYDIA\Parkinson_github'));


maxNumCompThreads(3);

dpath = 'G:\Classification\';
figpath = 'G:\Classification\figs';


subs = {'sub-88020741'	'sub-88019749'	'sub-828'	'sub-810'	'sub-40021'	'sub-40005'	'sub-40016' 'sub-40007'	'sub-40008'	'sub-40010'	'sub-40012'	'sub-40017'	'sub-40018'	'sub-40019'	'sub-40020'	'sub-40022'	'sub-40023'	'sub-40024'	'sub-40025'	'sub-40026'	'sub-801'	'sub-802'	'sub-804'	'sub-805'	'sub-806'	'sub-807'	'sub-808'	'sub-809'	'sub-811'	'sub-813'	'sub-814'	'sub-815'	'sub-816'	'sub-817'	'sub-818'	'sub-819'	'sub-820'	'sub-821'	'sub-822'	'sub-823'	'sub-824'	'sub-825'	'sub-826'	'sub-827'	'sub-829'	'sub-88017953'	'sub-88017997'	'sub-88018225'	'sub-88018401'	'sub-88018585'	'sub-88019077'	'sub-88019657'	'sub-88019661'	'sub-88019705'	'sub-88019797'	'sub-88019885'	'sub-88020649' ...
'sub-914'	'sub-912'	'sub-8070'	'sub-100011'	'sub-100038'	'sub-100030'	'sub-905' 'sub-10001'	'sub-100010'	'sub-100012'	'sub-100014'	'sub-100015'	'sub-100016'	'sub-100018'	'sub-100020'	'sub-100021'	'sub-100026'	'sub-100028'	'sub-100031'	'sub-100034'	'sub-100037'	'sub-10005'	'sub-10006'	'sub-10007'	'sub-10008'	'sub-8060'	'sub-88041893'	'sub-88041941'	'sub-88049585'	'sub-88053453'	'sub-88053545'	'sub-88055121'	'sub-88055301'	'sub-88068841'	'sub-88075053'	'sub-890'	'sub-891'	'sub-892'	'sub-893'	'sub-894'	'sub-895'	'sub-896'	'sub-897'	'sub-898'	'sub-899'	'sub-900'	'sub-901'	'sub-902'	'sub-903'	'sub-904'	'sub-906'	'sub-907'	'sub-908'	'sub-909'	'sub-910'	'sub-911'	'sub-913'};


sess = {'ses-1'};


cd('G:\Classification\kmeans\'); % load group Nk according to condition!!!
load 'kmeans_10mm_Nk25_300tr_step200ms_EEG.mat'

cd([dpath, '\sub-88075053\ses-1\']); % load sources of a subject as template to conduct the plotting
load source_forward
load source_inverse


load correction_vox_inside_eeg
voxel_inside = find(source.inside==1);  % voxel inside cortical mask
Nvox= 1921;
Nk = 25;


% 3.1. Preparation of power spectra matrix for subsequent K-means clustering
% 3.2. Classification of power spectra into previous clusters of the whole group
% 3.3. Proportion of power spectra categorized in each cluster
% 3.4. Compute natural frequency of each voxel (output: fnat and propkz)
% 3.5. Plots of single-subject natural frequency maps (output: figures)

for sub=1:numel(subs)

    route = fullfile(dpath, subs{sub}, sess{1}); % reconstruct route
    cd (route);


    load freq_allvox_10mm_steps200ms


    %% 3.1. Preparation of power spectra matrix for subsequent K-means clustering


    rng('shuffle')

    ct = 1;
    powspsub  = single(zeros(size(powsp,1)*size(powsp,3),size(powsp,2)));
    kvox   = [];
    ktrial = [];
    for i = 1:size(powsp,1)
        for tr = 1:size(powsp,3)
            powspsub(ct,:) = powsp(i,:,tr);
            kvox(ct)    = i;
            ktrial(ct)  = tr;
            ct = ct+1;
        end
    end

    bl       = sum(powspsub,2);                                % save kmeans_10mm_baseline bl
    powspsub = powspsub./repmat(bl,[1 size(powspsub,2)]);      % compute relative power to correct the center of the head bias


    %% 3.2. Classification of power spectra into clusters of the whole group

    % K-means clustering
    [D,I] = pdist2(C,powspsub, 'cosine','Smallest',1);

    %% 3.3. Proportion of power spectra categorized in each cluster

    propk = NaN(Nk,Nvox);
    for k = 1:Nk
        disp(['Cluster ' num2str(k) '/' num2str(Nk)])
        ctk = find(I==k);
        ctkvox = kvox(ctk)';
        ctkvoxfilt = ctkvox==[1:Nvox];
        propk(k,:)=sum(ctkvoxfilt)./Ntr_ok;
    end

    %% 3.4. Compute natural frequency of each voxel

    f   = 0.55:0.05:3.55;        % until 34.8 Hz
    foi = exp(f);
    ff  = [];
    for k = 1:Nk
        sp = C(k,:);
        [pks,locs] = findpeaks(sp);
        fx  = round(foi(locs(find(pks == max(pks)))),1);
        ff(k) = fx;
    end

    [ffsort,idf]=sort(ff);
    ffsort = unique(ffsort);

    ff2=ff;
    ff = NaN(Nk,2);
    badk = [];
    %     figure
    for k = 1:Nk
        sp = C(idf(k),:);
        %         plot(foi,sp)
        %         title(num2str(ff2(idf(k))))
        %         pause
        [pks,locs] = findpeaks(sp,'MinPeakHeight',0.1,'MinPeakProminence',0.02);
        % if it does not find any peak, go to the next centroid
        if length(pks) == 1                                % unimodal spectrum
            ff(idf(k),1) = round(foi(locs(1)),1);
            ff(idf(k),2) = round(foi(locs(1)),1);      % if unimodal spectrum, repeat 1st peak
        elseif length(pks) == 2                            % bimodal spectrum
            if (foi(locs(1)-1)*2 > foi(locs(2)-1) &  foi(locs(1)-1)*2 < foi(locs(2)+1)) | ...
                    (foi(locs(1)+1)*2 > foi(locs(2)-1) &  foi(locs(1)+1)*2 < foi(locs(2)+1))
                ff(idf(k),1) = round(foi(locs(1)),1);     % harmonics: only 1st peak (fundamental frequency)
                ff(idf(k),2) = round(foi(locs(1)),1);
            else
                ff(idf(k),1) = round(foi(locs(1)),1);     % no harmonics: take both peaks
                ff(idf(k),2) = round(foi(locs(2)),1);
            end
        elseif length(pks) > 2                         % in case of a third residual peak
            [~,id] = sort(pks,'descend');
            locs = locs(id(1:2));
            if (foi(locs(1)-1)*2 > foi(locs(2)-1) &  foi(locs(1)-1)*2 < foi(locs(2)+1)) | ...
                    (foi(locs(1)+1)*2 > foi(locs(2)-1) &  foi(locs(1)+1)*2 < foi(locs(2)+1))
                ff(idf(k),1) = round(foi(locs(1)),1);     % harmonics: only 1st peak (fundamental frequency)
                ff(idf(k),2) = round(foi(locs(1)),1);
            else
                ff(idf(k),1) = round(foi(locs(1)),1);     % no harmonics: take both peaks
                ff(idf(k),2) = round(foi(locs(2)),1);
            end
        elseif length(pks) == 0
            badk = [badk idf(k)];
        end
    end

    propk2 = zeros(length(ffsort),Nvox);

    for i=1:length(ffsort)
        [r,c] = find(ff==ffsort(i));
        r = unique(r);
        for j=1:length(r)
            if ff(r(j),1) ~= ff(r(j),2)
                propk2(i,:) = propk2(i,:) + 1/2.*sum(propk(r(j),:),1);    % bimodal
            elseif ff(r(j),1) == ff(r(j),2)
                propk2(i,:) = propk2(i,:) + sum(propk(r(j),:),1);    % unimodal
            end
        end
    end


    emptycol = find(sum(propk2,2)==0);
    propk2(emptycol,:) = [];
    ffsort(emptycol) = [];

    propkz = (propk2-mean(propk2,2))./std(propk2,0,2);        % z-normalize

    [dim,xx,yy,zz,connmat, dtempl] = Omega_neighbors_ly(source); % pers function

    propkz2sm = NaN(size(propkz));
    fnatsm=[];
    fnatsm_T=[];
    fnatsm_p=[];
    f2 = interp(ffsort,10);

    for vx=1:Nvox  % smooth with neighbours
        vxneigh = connmat(vx,:)==1;
        propkzneig= propkz(:,vxneigh);
        propinterp = [];
        for i=1:size(propkzneig,2)
            propinterp(:,i) = interp(propkzneig(:,i),10);
            % propinterp(:,i) = interp1(propkzneig(:,i),f2,'pchip');
        end
        [h,p,ci,stats] = ttest(propinterp');

        [tmax,idmax] = max(stats.tstat);
        fnatsm(vx) = f2(idmax);
        fnatsm_T(vx) = tmax;
        fnatsm_p(vx) = p(idmax);

        tval = stats.tstat;
        tval(p>.05) = NaN;
        tval(tval < 0) = NaN;
        [pks,locs] = findpeaks(tval);

        fnat.pks{vx}=f2(locs);
        fnat.w{vx}=pks*100./sum(pks);

    end

    fnatsm2 = fnatsm;
    fnatsm2(fnatsm_p>.05) = NaN;

    fnat.fnat = fnatsm;
    fnat.fnatsig = fnatsm2;
    fnat.tval = fnatsm_T;
    fnat.pval = fnatsm_p;
    cd (route);
    save singlesub_fnat_steps200ms fnat propkz

    %% 3.5.  Plots of single-subject natural frequency maps

    source2 = source;
    source2.avg.pow(voxel_inside) = log(fnatsm2);
    source2.avg.mom = cell(length(source2.inside),1);

    cfg=[];
    cfg.parameter  = 'pow';
    cfg.downsample = 2;
    cfg.interpmethod = 'nearest';
    source_interp = ft_sourceinterpolate (cfg, source2, source_forward.mri);

    figure('WindowState','maximized','Color',[1 1 1]);
    % figure

    cfg               = [];
    cfg.figure        = 'gca';
    cfg.method        = 'surface';
    cfg.funparameter  = 'pow';
    cfg.maskparameter = cfg.funparameter;
    cfg.funcolorlim   = [0.7 3.4];
    cfg.funcolormap   = 'jet_omega_mod';
    cfg.projmethod    = 'nearest';
    cfg.opacity       = 0.8;
    cfg.camlight      = 'no';
    cfg.colorbar      = 'no';
    cfg.surffile     = 'surface_pial_left.mat';
    cfg.surfinflated  = 'surface_inflated_left_caret_white.mat';
    subplot(2,2,1), ft_sourceplot(cfg,source_interp), view([-90 0]), camlight('left')
    subplot(2,2,3), ft_sourceplot(cfg,source_interp), view([90 0]),  camlight('left')

    cfg.surffile     = 'surface_pial_right.mat';
    cfg.surfinflated  = 'surface_inflated_right_caret_white.mat';
    subplot(2,2,2), ft_sourceplot(cfg,source_interp), view([90 0]),  camlight('right')
    subplot(2,2,4), ft_sourceplot(cfg,source_interp), view([-90 0]), camlight('right')

    cd(figpath); % folder to save figs
    print('-dtiff','-r300',[subs{sub} '_singlesub.tiff']);
    close
end

