% Plot results from backprop learning for biophysical model
% i.e., Fig. 5b

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

% Modify for different input correlations
sigma = 20; % correlation radius, um

load(strcat('data_r',num2str(sigma),'/grc_bp_biophys_r',num2str(sigma),'.mat'))

err_grc = err_rms_grc; err_mf = err_rms_mf;
thresh = 0.2; % threshold for determining learning speed
T = size(err_grc,3);

% Get number of epochs until learning is complete
% both for GC- and for MF-based learning
grc = nan(length(N_syn),length(f_mf));
mf = nan(length(N_syn),length(f_mf));
for j = 1:length(N_syn)
    for k = 1:length(f_mf)
        e_grc = reshape(err_grc(j,k,:),1,T);
        e_mf = reshape(err_mf(j,k,:),1,T);
        
        temp = find(e_grc<=thresh);
        if numel(temp) > 0
            grc(j,k) = temp(1);
        end
        
        temp = find(e_mf<=thresh);
        if numel(temp) > 0
            mf(j,k) = temp(1);
        end
    end
end

% Learning speed is 1 / number epochs to reach threshold
speed_grc = (1./grc);
speed_mf = (1./mf);
speed_norm = speed_grc./speed_mf;

% Find points where MF and GC learning speeds are equal
f_mf_zeros = []; x = [];
for j = 1:length(N_syn)
    temp = grc(j,:) - mf(j,:);
    for i = 1:length(temp)-1
        if temp(i) >0 && temp(i+1) <0
            f_mf_zeros = [f_mf_zeros, interp1(temp([i,i+1]),f_mf([i,i+1]),0)];
            x = [x,j];
        elseif temp(i) <0 && temp(i+1) >0
            f_mf_zeros = [f_mf_zeros, interp1(temp([i,i+1]),f_mf([i,i+1]),0)];
            x = [x,j];
        elseif temp(i) == 0
            f_mf_zeros = [f_mf_zeros, f_mf(i)];
            x = [x,j];
        end
    end    
end

f = fit(x',f_mf_zeros','exp2','StartPoint',[.5,.01,-1,-.5]);
y2 = f.a*exp(f.b*x) + f.c*exp(f.d*x);

% Do not plot region in which MF speed is faster than GC speed
for d = N_syn
    temp = find(f_mf < f.a*exp(f.b*d) + f.c*exp(f.d*d));
    speed_norm(d,temp) = NaN;
end

figure, imagesc(f_mf,N_syn,speed_mf);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Raw MF speed');
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,N_syn,speed_grc);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Raw GC speed');
xlabel('Fraction active MFs'); ylabel('Number inputs')

% Plot Fig. 1f
figure, y=imagesc(f_mf,N_syn,speed_norm);
set(y,'AlphaData',~isnan(speed_norm))
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Normalized learning speed');
xlabel('Fraction active MFs'); ylabel('Number inputs')
hold on, plot(y2,x,'Color',[.3,.8,1],'LineWidth',5)

%% Plot Fig. 5c

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

% Modify for different input correlations
sigma = 20; % correlation radius, um

load(strcat('data_r',num2str(sigma),'/grc_cov_biophys_r',num2str(sigma),'.mat'))
load(strcat('data_r',num2str(sigma),'/grc_spar_biophys_r',num2str(sigma),'.mat'))

figure, imagesc(f_mf,N_syn,spar_grc./spar_mf -1)
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Norm. pop. sparseness'); colormap(bluewhitered)
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,N_syn,var_grc./var_mf -1)
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Norm. total variance'); colormap(bluewhitered)
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,N_syn,log10(cov_grc./cov_mf))
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Log norm. Pearson correlation'); colormap(bluewhitered)
xlabel('Fraction active MFs'); ylabel('Number inputs')

%% Plot Fig. 5d

% Total variance
speed_norm_sparse = []; speed_norm_dense = [];

for sigma = 0:10:20
    
    % First get speed
    load(strcat('data_r',num2str(sigma),'/grc_bp_biophys_r',num2str(sigma),'.mat'))

    err_grc = err_rms_grc; err_mf = err_rms_mf;
    thresh = 0.2; % threshold for determining learning speed
    T = size(err_grc,3);

    % Get number of epochs until learning is complete
    % both for GC- and for MF-based learning
    grc = nan(length(N_syn),length(f_mf));
    mf = nan(length(N_syn),length(f_mf));
    for j = 1:length(N_syn)
        for k = 1:length(f_mf)
            e_grc = reshape(err_grc(j,k,:),1,T);
            e_mf = reshape(err_mf(j,k,:),1,T);

            temp = find(e_grc<=thresh);
            if numel(temp) > 0
                grc(j,k) = temp(1);
            end

            temp = find(e_mf<=thresh);
            if numel(temp) > 0
                mf(j,k) = temp(1);
            end
        end
    end

    % Learning speed is 1 / number epochs to reach threshold
    speed_grc = (1./grc);
    speed_mf = (1./mf);
    speed_norm_sparse = [speed_norm_sparse, nanmedian(speed_grc(4,:)./speed_mf(4,:))];
    speed_norm_dense = [speed_norm_dense, nanmedian(speed_grc(16,:)./speed_mf(16,:))];

end

figure, plot(0:10:20,speed_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, plot(0:10:20,speed_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,25],[1,1],'k'); axis([-5,25,0,3])
xlabel('Correlation radius (\mum)'), ylabel('Norm. speed')
set(gca,'FontSize',20)


%% Plot Fig. 5e,f,g

% Total variance
total_var_norm_sparse = []; total_var_norm_dense = [];

% Population correlation
pop_corr_norm_sparse = []; pop_corr_norm_dense = [];

% Population sparseness
sp_norm_sparse = []; sp_norm_dense = [];


for sigma = 0:10:20
    
    load(strcat('data_r',num2str(sigma),'/grc_cov_biophys_r',num2str(sigma),'.mat'))
    load(strcat('data_r',num2str(sigma),'/grc_spar_biophys_r',num2str(sigma),'.mat'))
    
    total_var_norm_sparse = [total_var_norm_sparse, nanmedian(var_grc(4,:)./var_mf(4,:))];
    sp_norm_sparse = [sp_norm_sparse, nanmedian(spar_grc(4,:)./spar_mf(4,:))];
    pop_corr_norm_sparse = [pop_corr_norm_sparse, nanmedian(cov_grc(4,:)./cov_mf(4,:))];
    
    total_var_norm_dense = [total_var_norm_dense, nanmedian(var_grc(16,:)./var_mf(16,:))];
    sp_norm_dense = [sp_norm_dense, nanmedian(spar_grc(16,:)./spar_mf(16,:))];
    pop_corr_norm_dense = [pop_corr_norm_dense, nanmedian(cov_grc(16,:)./cov_mf(16,:))];
    
end

figure, plot(0:10:20,sp_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, plot(0:10:20,sp_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,25],[1,1],'k'); axis([-5,25,1,2])
xlabel('Correlation radius (\mum)'), ylabel('Norm. pop. sparseness')
set(gca,'FontSize',20)

figure, plot(0:10:20,total_var_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, plot(0:10:20,total_var_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,25],[1,1],'k'); axis([-5,25,0,2])
xlabel('Correlation radius (\mum)'), ylabel('Norm. total variance')
set(gca,'FontSize',20)

figure, semilogy(0:10:20,pop_corr_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, semilogy(0:10:20,pop_corr_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,25],[1,1],'k'); axis([-5,25,.5,15])
xlabel('Correlation radius (\mum)'), ylabel('Norm. pop. correlation')
set(gca,'FontSize',20)

%% Plot Fig. 5h

% Modify for different input correlations
sigma = 20; % correlation radius, um

N_syn_full = 1:20; % Synaptic connectivity / Number inputs
N_syn = [1,2,4,8,16]; % Subsampled Nsyn, for shuffled data
f_mf = linspace(.05,.95,19); % fraction active MFs

% Load unshuffled data
load(strcat('data_r',num2str(sigma),'/grc_bp_biophys_r',num2str(sigma),'.mat'))
load(strcat('data_r',num2str(sigma),'/grc_cov_biophys_r',num2str(sigma),'.mat'))

% Subsample unsampled data to Nsyn of [1,2,4,8,16]
% accts for fact that shuffled analysis performed only on 1,2,4,8,16
% because of computational speed
subsample = zeros(size(N_syn_full));
for k = 1:length(N_syn)
    subsample(find(N_syn_full==N_syn(k))) = 1;
end
cov_grc = cov_grc(subsample==1,:);    
cov_mf = cov_mf(subsample==1,:); 
err_rms_grc = err_rms_grc(subsample==1,:,:);    

% Load shuffled data
load(strcat('data_r',num2str(sigma),'/grc_bp_biophys_r',num2str(sigma),'_shuff.mat'))

err_grc = err_rms_grc; err_sh = err_rms_sh;
thresh = 0.2; % threshold for determining learning speed
T = size(err_grc,3);

% Get number of epochs until learning is complete
% both for GC- and for MF-based learning
grc = nan(length(N_syn),length(f_mf));
sh = nan(length(N_syn),length(f_mf));
for j = 1:length(N_syn)
    for k = 1:length(f_mf)
        e_grc = reshape(err_grc(j,k,:),1,T);
        e_sh = reshape(err_sh(j,k,:),1,T);
        
        temp = find(e_grc<=thresh);
        if numel(temp) > 0
            grc(j,k) = temp(1);
        end
        
        temp = find(e_sh<=thresh);
        if numel(temp) > 0
            sh(j,k) = temp(1);
        end
    end
end

% Contribution of correlation to learning speed
speed_grc = 1./grc;
speed_sh = 1./sh;
speed_norm_corr = speed_grc./speed_sh;

% Normalized population correlation
pop_corr_norm = cov_grc./cov_mf;

figure, plot(pop_corr_norm,speed_norm_corr,'ok')
hold on, plot([.5,4],[1,1],'k'), plot([1,1],[0,4],'k')
axis([.5,4,0,4])
set(gca,'FontSize',20)
xlabel('Norm. population correlation')
ylabel({'Contribution of correlation','to learning speed'})



