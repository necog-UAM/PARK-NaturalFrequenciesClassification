%% This script is to obtain the brain's natural frequencies at rest from EEG recordings for Parkinson's study (Arana et al., 2026)
% (without corregistration)
% From preprocessed data with Gil pipeline (.set) to power spectra of all subs and voxels.
% The main outputs are dataclean, badsegments, source_forward, source_inverse, and the power spectra in freq_allvox_10mm_steps100ms.

% Warning: ensure proper paths troughout the whole script
clear all
close all
clc

restoredefaultpath
addpath ('Z:\Toolbox\fieldtrip-20230118');
ft_defaults
addpath(genpath('Z:\LYDIA\Parkinson_github'));

maxNumCompThreads(3);


dpath = 'G:\Classification\';
outpath = 'G:\Classification\';

subs = {'sub-88020741'	'sub-88019749'	'sub-828'	'sub-810'	'sub-40021'	'sub-40005'	'sub-40016' 'sub-40007'	'sub-40008'	'sub-40010'	'sub-40012'	'sub-40017'	'sub-40018'	'sub-40019'	'sub-40020'	'sub-40022'	'sub-40023'	'sub-40024'	'sub-40025'	'sub-40026'	'sub-801'	'sub-802'	'sub-804'	'sub-805'	'sub-806'	'sub-807'	'sub-808'	'sub-809'	'sub-811'	'sub-813'	'sub-814'	'sub-815'	'sub-816'	'sub-817'	'sub-818'	'sub-819'	'sub-820'	'sub-821'	'sub-822'	'sub-823'	'sub-824'	'sub-825'	'sub-826'	'sub-827'	'sub-829'	'sub-88017953'	'sub-88017997'	'sub-88018225'	'sub-88018401'	'sub-88018585'	'sub-88019077'	'sub-88019657'	'sub-88019661'	'sub-88019705'	'sub-88019797'	'sub-88019885'	'sub-88020649' ...
'sub-914'	'sub-912'	'sub-8070'	'sub-100011'	'sub-100038'	'sub-100030'	'sub-905' 'sub-10001'	'sub-100010'	'sub-100012'	'sub-100014'	'sub-100015'	'sub-100016'	'sub-100018'	'sub-100020'	'sub-100021'	'sub-100026'	'sub-100028'	'sub-100031'	'sub-100034'	'sub-100037'	'sub-10005'	'sub-10006'	'sub-10007'	'sub-10008'	'sub-8060'	'sub-88041893'	'sub-88041941'	'sub-88049585'	'sub-88053453'	'sub-88053545'	'sub-88055121'	'sub-88055301'	'sub-88068841'	'sub-88075053'	'sub-890'	'sub-891'	'sub-892'	'sub-893'	'sub-894'	'sub-895'	'sub-896'	'sub-897'	'sub-898'	'sub-899'	'sub-900'	'sub-901'	'sub-902'	'sub-903'	'sub-904'	'sub-906'	'sub-907'	'sub-908'	'sub-909'	'sub-910'	'sub-911'	'sub-913'};


sess = {'ses-1'};


Nsub=numel(subs);

% 1.1. Reading data from .set (output: dataclean, badsegments)
% 1.2. Beamforming: head model, forward model (output: source_forward), inverse model (output: source_inverse)
% 1.3. Reconstruction of source-level time-series
% 1.4. Frequency analysis parameters
% 1.5. Frequency analysis computation (output: freq_allvox_10mm_steps100ms)

%% 1.1. Reading data
% This part reads data already preprocessed with Gil % et al. pipeline (adapted) 
% It is required for each subj the clean data in Fieldtrip format (dataclean) and badsegments (.mat of 2 columns, 1st col with time in seconds of the start of each badsegment and 2nd the end

for sub=1:numel(subs)
    route = fullfile(dpath, subs{sub}, sess{1}); % reconstruct route
    cd(route)

    load dataclean

    %% 1.2. Beamforming
    % Head model
    
    
    % Load standard volume conduction, standard MRI, standard electrode positions
    load standard_mri.mat % standard MRI from fieldtrip
    load standard_vol.mat
    elec = ft_read_sens('standard_1005.elc');

    % Select electrodes common to all datasets
    elec_standard = elec;
    cfg = [];
    cfg.channel = {'Fp1'	'Fz'	'F3'	'F7'	'C3'	'T7'	'Pz'	'P3'	'P7'	'O1'	'Oz'	'O2'	'P4'	'P8'	'Cz'	'C4'	'T8'	'F4'	'F8'	'Fp2'	'FC3'	'FCz'	'CP3'	'CP4'	'FC4'};
    dataclean = ft_selectdata(cfg, dataclean);


    elec.label    = dataclean.label;
    elec.unit     = 'mm';
    elec.elecpos  = [];
    elec.chanpos  = [];
    elec.chantype = [];
    elec.chanunit = [];

    for i = 1:length(elec.label)
        elec.elecpos(i,1:3) = elec_standard.elecpos(find(strcmp(elec_standard.label,elec.label{i})),:);
        elec.chanpos(i,1:3) = elec_standard.elecpos(find(strcmp(elec_standard.label,elec.label{i})),:);
        elec.chantype{i}    = 'eeg';
        elec.chanunit{i}    = 'V';
    end

    dataclean.elec = elec;
    [vol, elec] = ft_prepare_vol_sens (vol,elec);

    % Forward model 
    % Create the grid and compute leadfields

    % Load normalized template grid (10mm)
    load standard_sourcemodel3d10mm
    grid = sourcemodel;

    % Convert elec, vol and grid to common units (mm)
    elec = ft_convert_units(elec, vol.unit);
    grid = ft_convert_units(grid, vol.unit);


    % Select only voxels within cortical mask (e.g. cerebellum is excluded)
    % and corrected (inside the cortical surface projected with ft_sourceplot)
    % created with select_corticalvox_aal
    load correction_vox_inside_eeg.mat % Caution, 1921 voxels instead of 1925 in MEG, because dipole collided with electrode and the vox were NaN, eliminated vox 139,162,581,866 (corresponding with 2122,2157,3050,3581 from correction_vox_inside_10mm in MEG)
    grid.inside = inside;


    % Compute leadfields for each grid's voxel
    cfg             = [];
    cfg.grid        = grid;
    cfg.elec        = elec;
    cfg.vol         = vol;
    cfg.normalize   = 'yes';
    cfg.reducerank  = 'no'; %for MEG yes
    grid2           = ft_prepare_leadfield(cfg);

    % Check that volume, electrodes and grid are correct

    % figure
    % 
    % % Outer layer (1) of volume (skin surface) in green
    % plot3 (vol.bnd(1).pos(:,1), vol.bnd(1).pos(:,2), vol.bnd(1).pos(:,3), '.','MarkerEdgeColor',[0 0.8 0]), hold on
    % % Electrodos en rojo
    % plot3 (elec.elecpos(:,1), elec.elecpos(:,2), elec.elecpos(:,3), '.','MarkerEdgeColor',[0.8 0 0],'MarkerSize',25), hold on
    % 
    % % Inner layer (3) of colume (cortical surface) in blue
    % plot3 (vol.bnd(3).pos(:,1), vol.bnd(3).pos(:,2), vol.bnd(3).pos(:,3), '.','MarkerEdgeColor',[0 0 1]), hold on
    % % Grid: only positions inside cortical volume (grid.inside)
    % plot3 (grid2.pos(grid2.inside,1), grid2.pos(grid2.inside,2), grid2.pos(grid2.inside,3), '+k')

    % Save info about volume, electrodes and grid in a structure (source_forward)

    source_forward      = [];
    source_forward.vol  = vol;
    source_forward.mri  = mri;
    source_forward.elec = elec;
    source_forward.grid = grid2;

    cd(fullfile(outpath, subs{sub}, sess{1}));
    save source_forward source_forward

    % Inverse solution: spatial filter

    % Re-referencing data to average (just in case, already rereference in preprocessing, but mandatory for EEG)
    cfg            = [];
    cfg.reref      = 'yes' ;
    cfg.refchannel = 'all';
    dataclean    = ft_preprocessing(cfg, dataclean);

    % Compute covariance matrix, common to all conditions (in datos_cov.cov)

    cfg            = [];
    cfg.covariance = 'yes';
    datacov      = ft_timelockanalysis(cfg, dataclean);

    % Computation of beamforming weights, filter (in source.avg.filter)

    cfg                   = [];
    cfg.method            = 'lcmv';
    cfg.grad              = source_forward.elec;
    cfg.headmodel         = source_forward.vol;
    cfg.grid              = source_forward.grid;
    cfg.lcmv.fixedori     = 'yes';
    cfg.lcmv.normalize    = 'yes';
    cfg.lcmv.projectnoise = 'yes';
    cfg.lcmv.keepfilter   = 'yes';          % important: save filters to use them later
    cfg.lcmv.lambda       = '10%';          % the higher the smoother
    cfg.lcmv.reducerank   = 3;
    source                = ft_sourceanalysis(cfg, datacov);

    load standard_sourcemodel3d10mm
    source.avg.ori = {};
    source.avg.mom = {};
    source.avg.noisecov = {};
    source.pos     = sourcemodel.pos;            % standard grid positions
    source.inside  = grid.inside;

    cd(fullfile(outpath, subs{sub}, sess{1}));
    save source_inverse source

    %% 1.3. Reconstruction of source-level time-series
    % Frequency analysis parameters
    % Frequency analysis computation

    % Need dataclean, source_forward, source_inverse
    % Reconstruct source-space data % CAUTION, DATASOURCE >2GB
    time         = dataclean.time{1};
    voxel_inside = find(source.inside==1);
    Nvox         = length(voxel_inside);
    datasource   = zeros(Nvox,length(time));
    for i = 1:Nvox
        disp([num2str(i) ' / ' num2str(length(voxel_inside))])
        datasource(i,:) = source.avg.filter{voxel_inside(i)} * dataclean.trial{1};
    end

    datasource = datasource./repmat(std(datasource,0,2),[1,length(time)]);   % baseline correction to account for the centre of the head bias
    badsegments = [];
    if exist('badsegments.mat', 'file')
        load('badsegments.mat')
        if ~isempty(badsegments)
            for b = 1:length(badsegments)
                t1 = findbin(time,badsegments{b}(1)); % findbin is personalized
                t2 = findbin(time,badsegments{b}(2));
                time(t1:t2) = [];
                datasource(:,t1:t2) = [];
            end
        end
    end

    % Use a max of 10-minute recordings for all participants
    if time(end) > 600      % 
        t1 = findbin(time,600);
        t2 = findbin(time,time(end));
        time(t1:t2) = [];
        datasource(:,t1:t2) = [];
    end

    % Organize source-reconstructed data in a new fieldtrip structure
    dataclean.trial{1}   = single(datasource);
    dataclean.time{1}    = time;
    dataclean.label      = {};
    dataclean.sampleinfo = [1 size(dataclean.trial{1},2)];
    for i = 1:Nvox
        dataclean.label{i} = ['V' num2str(i)];
    end

    clear datasource

    % If there were not badsegments, data is organized in 1 single trial
    % If there were badsegments (discontinuities), data is organized in several trials


    if ~isempty(badsegments)
        dtime      = diff(time);
        sampling_interval = 1 / dataclean.fsample;
        threshold = sampling_interval * 1.5;  % UPDATE

        [pks, locs] = findpeaks(diff(time), 'MinPeakHeight', threshold); % UPDATE

        % [pks,locs] = findpeaks(dtime);
        pkstime    = time(locs);
        if length(pkstime) > 0
            trl = [];
            trl(1,1) = 1;
            for t = 1:length(pks)
                tt = findbin(time, pkstime(t));
                trl(t,2) = tt - 1;
                trl(t+1,1) = tt + 1;
            end

            trl(t+1,2) = length(time);
            trl(:,3)   = 0;

            cfg     = [];
            cfg.trl = trl;
            dataclean = ft_redefinetrial(cfg,dataclean);
        end
    end

    %% 1.4. Frequency analysis parameters
    % Parameters: foi, toi, t_ftimwin
    
    f         = 0.55:0.05:3.55;
    foi       = exp(f);                % logarithmically spaced frequencies
    t_ftimwin = 5./foi;                % time-window adapted to each frequency (5 cycles); change to 3 or 7 cycles for control analyses
    t_fstep   = 0.2;                   % sliding time-window (200 ms steps)

    bd   = t_ftimwin(1).*2;            % remove borders = 2*time-window
    bdpt = bd.*dataclean.fsample;

    Ntr  = 0;
    toi2 = {};
    ct   = 1;
    valid_tr = [];
    for tr = 1:length(dataclean.trial)
        toi = bd:t_fstep:dataclean.time{tr}(end)-bd;
        if length(toi) > 0
            toi2{ct} = toi;
            Ntr = Ntr+length(toi);
            valid_tr(ct) = tr;
            ct = ct+1;
        end
    end

    %% 1.5. Frequency analysis computation

    Nvox  = length(dataclean.label);
    Nf    = length(foi);
    powsp = single(NaN(Nvox,Nf,Ntr));

    tr2 = 0;
    for tr = 1:length(valid_tr)
        tr1 = tr2+1;
        tr2 = tr1+length(toi2{tr})-1;

        cfg            = [];
        cfg.trials     = valid_tr(tr);
        cfg.method     = 'mtmconvol';
        cfg.taper      = 'hanning';
        cfg.output     = 'pow';
        cfg.foi        = foi;
        cfg.toi        = toi2{tr};
        cfg.t_ftimwin  = t_ftimwin;
        cfg.pad        = 'nextpow2';
        cfg.keeptrials = 'yes';
        freq           = ft_freqanalysis(cfg, dataclean);

        powsp(:,:,tr1:tr2)= single(squeeze(freq.powspctrm));
    end

    trok=[];
    ct=1;
    for tr = 1:size(powsp,3)
        if sum(sum(isnan(powsp(:,:,tr))))==0
            trok(ct) = tr;
            ct=ct+1;            end
    end

    powsp = powsp(:,:,trok);
    Ntr_ok=length(trok); % all Ntr of the subj


    cd(fullfile(outpath, subs{sub}, sess{1}));
    save freq_allvox_10mm_steps200ms powsp foi Ntr_ok
    pause(3);
    
end
