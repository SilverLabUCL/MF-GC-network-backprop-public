% Partially shuffle code and compare activity patterns
% Plots Fig. 5a,b,c

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

N_mf = 187; N_grc = 487;
N_patt = 640;

theta_initial = 3; NADT = 0;

N_repeats = 1;%25;

% Modify for different input correlations
sigma = 20; % correlation radius, um

% Total variance
total_var_grc = nan(length(N_syn),length(f_mf),N_repeats);
total_var_sh = nan(length(N_syn),length(f_mf),N_repeats);
total_var_mf =  zeros(length(N_syn),length(f_mf),N_repeats);

% Population correlation
pop_corr_grc =  nan(length(N_syn),length(f_mf),N_repeats);
pop_corr_sh =  nan(length(N_syn),length(f_mf),N_repeats);
pop_corr_mf =  zeros(length(N_syn),length(f_mf),N_repeats);

% Population sparseness
sp_grc = nan(length(N_syn),length(f_mf),N_repeats);
sp_sh = nan(length(N_syn),length(f_mf),N_repeats);

% Number inactive GCs
f_inactive_grc = nan(length(N_syn),length(f_mf),N_repeats);
f_inactive_sh = nan(length(N_syn),length(f_mf),N_repeats);

% Average activity
avg_grc = nan(length(N_syn),length(f_mf),N_repeats);
avg_sh = nan(length(N_syn),length(f_mf),N_repeats);

tic
for k1 = N_syn
    k1, toc
    load(strcat('../network_structures/GCLconnectivity_',int2str(k1),'.mat'))
    conn_mat = double(conn_mat);
    
    for k2 = 1:length(f_mf)
        
        for k3 = 1:N_repeats
        
            % Input MF patterns
            if sigma == 0 % Independent case
                x_mf = zeros(N_mf,N_patt);
                for i = 1:N_patt
                    mf_on = randsample(N_mf,round(f_mf(k2)*N_mf));
                    x_mf(mf_on,i) = 1.;
                end
            elseif sigma >0 % Correlated case -- generated following Macke et al. 2009
                load(strcat('../input_statistics/mf_patterns_r',num2str(sigma),'.mat'))
                R = Rs(:,:,k2); g = gs(k2);
                t = R' * randn(N_mf,N_patt);
                S = (t>-g(1)*ones(N_mf,N_patt)); 
                x_mf = S;
            end

            theta = theta_initial + NADT*f_mf(k2); % threshold
            in = 4/k1*conn_mat'*x_mf; % input 
            x_grc = max(in-theta,0); % Output GC activity

            x_mf = double(x_mf); x_grc = double(x_grc);
            
            % Eigenvalues of covariance matrix of MF patterns
            C_mf = cov(x_mf');
            [~,L_mf] = eig(C_mf); L_mf = diag(L_mf);
            L_mf = real(sqrt(L_mf)); 
            
            % Total variance of MFs
            total_var_mf(k1,k2,k3) = sum(L_mf.^2);

            % Population correlation of MFs
            pop_corr_mf(k1,k2,k3) = (max(L_mf)/sum(L_mf) - 1./N_mf)/(1-1/N_mf);
           
            if max(x_grc(:)) > 0
                
                % Eigenvalues of covariance matrix of GC patterns
                C_grc = cov(x_grc'); 
                [~,L_grc] = eig(C_grc);  L_grc = diag(L_grc);
                L_grc =real(sqrt(L_grc));
                
                % Total variance of GCs
                total_var_grc(k1,k2,k3) = sum(L_grc.^2); % total variation

                % Population correlation of GCs
                pop_corr_grc(k1,k2,k3) = (max(L_grc)/sum(L_grc) - 1./N_grc)/(1-1/N_grc);
                
                % Avg. population sparseness
                sptemp = zeros(1,N_patt); 
                for t = 1:N_patt
                    sptemp(t) =(N_grc-sum(x_grc(:,t))^2/sum(x_grc(:,t).^2))/(N_grc-1);
                end
                sp_grc(k1,k2,k3) = nanmean(sptemp);
                
                % Number inactive GCs
                f_inactive_grc(k1,k2,k3) = mean(sum((x_grc>0),1)/N_grc);
                
                % Average GC activity
                avg_grc(k1,k2,k3) = mean(x_grc(:));
                
                if pop_corr_grc(k1,k2,k3) < pop_corr_mf(k1,k2,k3)
                    [x_grc_sh, pop_corr_sh(k1,k2,k3), total_var_sh(k1,k2,k3)] = part_shuffle_higher(x_grc,pop_corr_grc(k1,k2,k3),pop_corr_mf(k1,k2,k3));
                elseif pop_corr_grc(k1,k2,k3) > pop_corr_mf(k1,k2,k3)
                    [x_grc_sh, pop_corr_sh(k1,k2,k3), total_var_sh(k1,k2,k3)] = part_shuffle(x_grc,pop_corr_grc(k1,k2,k3),pop_corr_mf(k1,k2,k3));
                end
                
                % Avg. population sparseness
                sptemp = zeros(1,N_patt); 
                for t = 1:N_patt
                    sptemp(t) =(N_grc-sum(x_grc_sh(:,t))^2/sum(x_grc_sh(:,t).^2))/(N_grc-1);
                end
                sp_sh(k1,k2,k3) = nanmean(sptemp);
                
                % Number inactive GCs
                f_inactive_sh(k1,k2,k3) = mean(sum((x_grc_sh>0),1)/N_grc);
                
                % Average GC activity
                avg_sh(k1,k2,k3) = mean(x_grc_sh(:));
                
            end
        end
    end
end

total_var_norm = mean(total_var_grc./total_var_mf,3);
total_var_norm_sh = mean(total_var_sh./total_var_mf,3);

pop_corr_norm = mean(pop_corr_grc./pop_corr_mf,3);
pop_corr_norm_sh = mean(pop_corr_sh./pop_corr_mf,3);

% Plot Fig. 5a
bins = linspace(.7,3,19);
figure, h = histc(pop_corr_norm(:),bins);
b = bar(bins,h);
set(b,'EdgeColor','w','FaceColor',[1,0,0])
set(gca,'FontSize',20); xlim([.5,3])
xlabel('Norm. pop. correlation'); ylabel('Number')

figure, h = hist(pop_corr_norm_sh(:),bins)
b = bar(bins,h);
set(b,'EdgeColor','w','FaceColor',[.35,0,.5])
set(gca,'FontSize',20); xlim([.5 3])
xlabel('Norm. pop. correlation'); ylabel('Number')

% Plot Fig. 5b,c
figure, plot(total_var_norm, total_var_norm_sh,'.k','MarkerSize',20)
hold on, plot([0,6],[0,6],'k'); axis([0,6,0,6])
title('Norm. total variance')
xlabel('GC patterns'); ylabel('Shuffled patterns')
set(gca,'FontSize',20)

figure, plot(mean(avg_grc,3),mean(avg_sh,3),'.k','MarkerSize',20)
hold on, plot([0,1],[0,1],'k'); axis([0,1,0,1])
title('Average GC activity')
xlabel('GC patterns'); ylabel('Shuffled patterns')
set(gca,'FontSize',20)

figure, plot(mean(sp_grc,3),mean(sp_sh,3),'.k','MarkerSize',20)
hold on, plot(mean(f_inactive_grc,3),mean(f_inactive_sh,3),'.','Color',[0,.5,0],'MarkerSize',20)
plot([0,1],[0,1],'k'); title('Sparseness'); axis([0,1,0,1])
xlabel('GC patterns'); ylabel('Shuffled patterns')
set(gca,'FontSize',20)

%% Plot effect on GC learning, Fig. 5d,e

load(strcat('results_bp/grc_toy_r',num2str(sigma),'.mat'))
load(strcat('results_bp_shuff/grc_toy_r',num2str(sigma),'_shuff.mat'))

err_grc = err_rms_grc; err_sh = err_rms_sh;
thresh = 0.2; % threshold for determining learning speed
T = size(err_grc,3);

% Get number of epochs until learning is complete
% both for GC- and for MF-based learning
grc = nan(length(N_syn),length(f_mf));
grc_sh = nan(length(N_syn),length(f_mf));
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
            grc_sh(j,k) = temp(1);
        end
    end
end

% Learning speed is 1 / number epochs to reach threshold
speed_grc = (1./grc);
speed_sh = (1./grc_sh);

% Contribution of correlation to learning speed
speed_norm_corr = speed_grc./speed_sh;

% Plot Fig. 5d
figure, imagesc(f_mf,1:20,speed_grc);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title({'True GC patterns','Learning Speed'});
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,1:20,speed_sh);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title({'Shuffled GC patterns','Learning Speed'});
xlabel('Fraction active MFs'); ylabel('Number inputs')

% Plot Fig. 5e
figure, plot(pop_corr_norm,speed_norm_corr,'ok')
hold on, plot([.5,4],[1,1],'k'), plot([1,1],[0,4],'k')
axis([.5,4,0,4])
set(gca,'FontSize',20)
xlabel('Norm. population correlation')
ylabel({'Contribution of correlation','to learning speed'})

