restoredefaultpath
addpath ('Z:\Toolbox\fieldtrip-20230118');
ft_defaults
addpath(genpath('Z:\LYDIA\Parkinson_github'));
addpath('Z:\Toolbox\slanCM\slanCM') %colormap

cd('G:\Classification\sub-88075053\ses-1\'); % load template sources of a subject for plotting
load 'source_forward.mat'
load 'source_inverse.mat'

cd('G:\Classification\results');
load('model_in_sample_noNorm_noPCA.mat', 'beta_haufe_original'); % load haufe weights

voxel_inside = find(source.inside==1);

source2 = source;
source2.avg.pow(voxel_inside) = beta_haufe_original; 
source2.avg.mom=cell(length(source2.avg.noise),1);
source2.time=1;

cfg = [];
cfg.parameter  = 'avg.pow';
cfg.downsample = 2;
cfg.interpmethod  =  'nearest';
source_interp  = ft_sourceinterpolate (cfg, source2, source_forward.mri);
   

    figure('WindowState','maximized','Color',[1 1 1]);
    % figure

    cfg               = [];
    cfg.figure        = 'gca';
    cfg.method        = 'surface';
    cfg.funparameter  = 'pow';
    cfg.maskparameter = cfg.funparameter;
    cfg.funcolormap   = flipud(slanCM('fusion'));
    cfg.funcolorlim   = [-3 3];  % Adjust

    cfg.zlim          = [-3 3];  % Adjust

    cfg.projmethod    = 'nearest';
    cfg.opacity       = 0.8;
    cfg.camlight      = 'no';
    cfg.colorbar      = 'yes';
    cfg.surffile     = 'surface_pial_left.mat';
    cfg.surfinflated  = 'surface_inflated_left_caret_white.mat';
    subplot(2,2,1), ft_sourceplot(cfg,source_interp), view([-90 0]), camlight('left')
    

    subplot(2,2,3), ft_sourceplot(cfg,source_interp), view([90 0]),  camlight('left')
    

    cfg.surffile     = 'surface_pial_right.mat';
    cfg.surfinflated  = 'surface_inflated_right_caret_white.mat';
    subplot(2,2,2), ft_sourceplot(cfg,source_interp), view([90 0]),  camlight('right')
   
    subplot(2,2,4), ft_sourceplot(cfg,source_interp), view([-90 0]), camlight('right')
    


