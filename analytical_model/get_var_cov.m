% Computes population activity metrics
% i.e., population sparseness, total variance,
% population correlation, and correlation coefficient
% generates panels for Fig. 2a,c, Fig. 3c,d, Fig. 4a,b
% and Supplementary Fig. 4

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

N_mf = 187; N_grc = 487;
N_patt = 640;

theta_initial = 3; NADT = 0;

N_repeats = 25;

% Modify for different input correlations
sigma = 20; % correlation radius, um

% Tools for averaging correlations without double counting
J_grc = zeros(N_grc,N_grc);
for i = 1:N_grc
    for j = (i+1):N_grc
        J_grc(i,j) = 1;
    end
end
J_mf = zeros(N_mf,N_mf);
for i = 1:N_mf
    for j = (i+1):N_mf
        J_mf(i,j) = 1;
    end
end

% Total variance
total_var_grc = nan(length(N_syn),length(f_mf),N_repeats);
total_var_mf =  zeros(length(N_syn),length(f_mf),N_repeats);

% Population correlation
pop_corr_grc =  nan(length(N_syn),length(f_mf),N_repeats);
pop_corr_mf =  zeros(length(N_syn),length(f_mf),N_repeats);

% Pearson correlation coefficient
corr_coef_grc =  nan(length(N_syn),length(f_mf),N_repeats);
corr_coef_mf = zeros(length(N_syn),length(f_mf),N_repeats);

% Population sparseness
sp_grc = nan(length(N_syn),length(f_mf),N_repeats);
sp_mf = zeros(length(N_syn),length(f_mf),N_repeats);

for k1 = N_syn
    k1
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
            
            % Pearson correlation coefficients
            C = corrcoef(x_grc'); corr_coef_grc(k1,k2,k3) = nanmean(abs(C(J_grc==1)));
            C = corrcoef(x_mf'); corr_coef_mf(k1,k2,k3) = nanmean(abs(C(J_mf==1)));
            
            % Eigenvalues of covariance matrix of MF patterns
            C_mf = cov(x_mf');
            [~,L_mf] = eig(C_mf); L_mf = diag(L_mf);
            L_mf = real(sqrt(L_mf)); 
            
            % Total variance of MFs
            total_var_mf(k1,k2,k3) = sum(L_mf.^2);

            % Population correlation of MFs
            pop_corr_mf(k1,k2,k3) = (max(L_mf)/sum(L_mf) - 1./N_mf)/(1-1/N_mf);
            
            % Following can be uncommented to account for hetereogeneity
            % of cell variances
            %x_temp = x_mf;
            %for i = 1:N_mf
            %    if sum(x(i,:)) > 0
            %    x_temp(i,:) = x(i,:)/std(x(i,:));
            %    end
            %end
            %C_mf = cov(x_temp');
            %[~,L_mf] = eig(C_mf); L_mf = diag(L_mf);
            %L_mf = real(sqrt(L_mf)); 
            %pop_corr_mf(k1,k2,k3) = (max(L_mf)/sum(L_mf) - 1./N_mf)/(1-1/N_mf);
            
            
            % Avg. population sparseness
            sptemp = zeros(1,N_patt); 
            for t = 1:N_patt
                sptemp(t) =(N_mf-sum(x_mf(:,t))^2/sum(x_mf(:,t).^2))/(N_mf-1);
            end  
            sp_mf(k1,k2,k3) = nanmean(sptemp);

            if max(x_grc(:)) > 0
                
                % Eigenvalues of covariance matrix of GC patterns
                C_grc = cov(x_grc'); 
                [~,L_grc] = eig(C_grc);  L_grc = diag(L_grc);
                L_grc =real(sqrt(L_grc));
                
                % Total variance of GCs
                total_var_grc(k1,k2,k3) = sum(L_grc.^2); % total variation

                % Population correlation of GCs
                pop_corr_grc(k1,k2,k3) = (max(L_grc)/sum(L_grc) - 1./N_grc)/(1-1/N_grc);
                
                % Following can be uncommented to account for hetereogeneity
                % of cell variances
                %x_temp = x_grc;
                %for i = 1:N_grc
                %    if sum(x(i,:)) > 0
                %    x_temp(i,:) = x(i,:)/std(x(i,:));
                %    end
                %end
                %C_grc = cov(x_temp');
                %[~,L_grc] = eig(C_grc); L_grc = diag(L_grc);
                %L_grc = real(sqrt(L_grc)); 
                %pop_corr_grc(k1,k2,k3) = (max(L_grc)/sum(L_grc) - 1./N_grc)/(1-1/N_grc);

                % Avg. population sparseness
                sptemp = zeros(1,N_patt); 
                for t = 1:N_patt
                    sptemp(t) =(N_grc-sum(x_grc(:,t))^2/sum(x_grc(:,t).^2))/(N_grc-1);
                end
                sp_grc(k1,k2,k3) = nanmean(sptemp);
            end
        end
    end
end

total_var_norm = mean(total_var_grc./total_var_mf,3);
pop_corr_norm =   mean(pop_corr_grc./pop_corr_mf,3);
corr_coef_norm =  mean(corr_coef_grc./corr_coef_mf,3);
sp_norm =  mean(sp_grc./sp_mf,3);

figure, imagesc(f_mf,1:20,sp_norm-1);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Norm. pop. sparseness'); colormap(bluewhitered)
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,1:20,total_var_norm-1);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Norm. total variance'); colormap(bluewhitered)
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,1:20,log10(pop_corr_norm));
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Log norm. pop. correlation'); colormap(bluewhitered)
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,1:20,log10(corr_coef_norm));
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Log norm. Pearson correlation'); colormap(bluewhitered)
xlabel('Fraction active MFs'); ylabel('Number inputs')

%% Plot population activity metrics as fn of correlation radius
% i.e., for Fig 2b,d and Fig.3b (top panels)

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

N_mf = 187; N_grc = 487;
N_patt = 640;

theta_initial = 3; NADT = 0;

N_repeats = 25;

% Total variance
total_var_norm_sparse = []; total_var_norm_dense = [];

% Population correlation
pop_corr_norm_sparse = []; pop_corr_norm_dense = [];

% Population sparseness
sp_norm_sparse = []; sp_norm_dense = [];

for sigma = 0:5:30
    sigma
    % Sparse & dense connectivity
    for k1 = [4,16]
        load(strcat('../network_structures/GCLconnectivity_',int2str(k1),'.mat'))  
        conn_mat = double(conn_mat);
    
        total_var_norm_temp = zeros(length(f_mf),N_repeats);
        pop_corr_norm_temp = zeros(length(f_mf),N_repeats);
        sp_norm_temp = zeros(length(f_mf),N_repeats);

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
                total_var_mf = sum(L_mf.^2);

                % Population correlation of MFs
                pop_corr_mf = (max(L_mf)/sum(L_mf) - 1./N_mf)/(1-1/N_mf);

                % Avg. population sparseness
                sptemp = zeros(1,N_patt); 
                for t = 1:N_patt
                    sptemp(t) =(N_mf-sum(x_mf(:,t))^2/sum(x_mf(:,t).^2))/(N_mf-1);
                end  
                sp_mf = nanmean(sptemp);

                if max(x_grc(:)) > 0

                    % Eigenvalues of covariance matrix of GC patterns
                    C_grc = cov(x_grc'); 
                    [~,L_grc] = eig(C_grc);  L_grc = diag(L_grc);
                    L_grc =real(sqrt(L_grc));

                    % Total variance of GCs
                    total_var_grc = sum(L_grc.^2); % total variation

                    % Population correlation of GCs
                    pop_corr_grc = (max(L_grc)/sum(L_grc) - 1./N_grc)/(1-1/N_grc);

                    % Avg. population sparseness
                    sptemp = zeros(1,N_patt); 
                    for t = 1:N_patt
                        sptemp(t) =(N_grc-sum(x_grc(:,t))^2/sum(x_grc(:,t).^2))/(N_grc-1);
                    end
                    sp_grc = nanmean(sptemp);
                else
                    sp_grc = NaN; pop_corr_grc = NaN; total_var_grc = NaN;
                end
            
                total_var_norm_temp(k2,k3) = total_var_grc/total_var_mf;
                pop_corr_norm_temp(k2,k3) = pop_corr_grc/pop_corr_mf;
                sp_norm_temp(k2,k3) = sp_grc/sp_mf;
            end
        end
        if k1 == 4
            total_var_norm_sparse = [total_var_norm_sparse, nanmedian(nanmean(total_var_norm_temp,2))];
            pop_corr_norm_sparse = [pop_corr_norm_sparse, nanmedian(nanmean(pop_corr_norm_temp,2))];
            sp_norm_sparse = [sp_norm_sparse, nanmedian(nanmean(sp_norm_temp,2))];
        elseif k1 == 16
            total_var_norm_dense = [total_var_norm_dense, nanmedian(nanmean(total_var_norm_temp,2))];
            pop_corr_norm_dense = [pop_corr_norm_dense, nanmedian(nanmean(pop_corr_norm_temp,2))];
            sp_norm_dense = [sp_norm_dense, nanmedian(nanmean(sp_norm_temp,2))];
        end
    end
    
end

figure, plot(0:5:30,sp_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, plot(0:5:30,sp_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,35],[1,1],'k'); axis([-5,35,0,4])
xlabel('Correlation radius (\mum)'), ylabel('Norm. pop. sparseness')
set(gca,'FontSize',20)

figure, plot(0:5:30,total_var_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, plot(0:5:30,total_var_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,35],[1,1],'k'); axis([-5,35,0,3])
xlabel('Correlation radius (\mum)'), ylabel('Norm. total variance')
set(gca,'FontSize',20)

figure, semilogy(0:5:30,pop_corr_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, semilogy(0:5:30,pop_corr_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,35],[1,1],'k'); axis([-5,35,.8,10])
xlabel('Correlation radius (\mum)'), ylabel('Norm. pop. correlation')
set(gca,'FontSize',20)

%% Plot distribution of eigenvalues
% Generates Fig. 3a
clear all; clc

% Parameters to plot
N_syn = 4; % number inputs
f_mf_ix = 10; % corresponds to f_mf = 0.5

N_mf = 187; N_grc = 487;
N_patt = 100000;

theta_initial = 3; NADT = 0;
f_mf = linspace(.05,.95,19); % fraction active MFs

load(strcat('../network_structures/GCLconnectivity_',int2str(N_syn),'.mat'))
conn_mat = double(conn_mat);
    
% Modify for different input correlations
sigma = 0; % correlation radius, um

% Input MF patterns
x_mf = zeros(N_mf,N_patt);
for i = 1:N_patt
    mf_on = randsample(N_mf,round(f_mf(f_mf_ix)*N_mf));
    x_mf(mf_on,i) = 1.;
end

theta = theta_initial + NADT*f_mf(f_mf_ix); % threshold
in = 4/N_syn*conn_mat'*x_mf; % input 
x_grc = max(in-theta,0); % Output GC activity

x_mf = double(x_mf); x_grc = double(x_grc);

% Eigenvalues of covariance matrix oLf MF patterns
C_mf = cov(x_mf');
[~,L_mf] = eig(C_mf); L_mf = diag(L_mf);
L_mf = real((L_mf)); 
L_mf = sort(L_mf,'descend');

% Eigenvalues of covariance matrix of GC patterns
C_grc = cov(x_grc'); 
[~,L_grc] = eig(C_grc);  L_grc = diag(L_grc);
L_grc =real((L_grc));
L_grc = sort(L_grc,'descend');

figure, plot((1:N_grc)/N_grc,L_grc,'r','LineWidth',2)
hold on, plot((1:N_mf)/N_mf,L_mf,'b','LineWidth',2)
set(gca,'FontSize',20)
xlabel('Normalized rank')
ylabel('Eigenvalue')



