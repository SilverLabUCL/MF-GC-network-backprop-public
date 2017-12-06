% Computes standard error of population correlation and Pearson correlation
% coefficient, i.e., Supplementary Fig. 3

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

% Population correlation
pop_corr_grc =  nan(length(N_syn),length(f_mf),N_repeats);
pop_corr_mf =  zeros(length(N_syn),length(f_mf),N_repeats);

% Pearson correlation coefficient
corr_coef_grc =  nan(length(N_syn),length(f_mf),N_repeats);
corr_coef_mf = zeros(length(N_syn),length(f_mf),N_repeats);

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
            
            % Population correlation of MFs
            pop_corr_mf(k1,k2,k3) = (max(L_mf)/sum(L_mf) - 1./N_mf)/(1-1/N_mf);
            
            if max(x_grc(:)) > 0
                % Eigenvalues of covariance matrix of GC patterns
                C_grc = cov(x_grc'); 
                [~,L_grc] = eig(C_grc);  L_grc = diag(L_grc);
                L_grc =real(sqrt(L_grc));
                
                % Population correlation of GCs
                pop_corr_grc(k1,k2,k3) = (max(L_grc)/sum(L_grc) - 1./N_grc)/(1-1/N_grc);

            end
        end
    end
end

% Mean values
pop_corr_norm =   mean(pop_corr_grc./pop_corr_mf,3);
corr_coef_norm =  mean(corr_coef_grc./corr_coef_mf,3);

% Standard error
pop_corr_norm_stderr = std(pop_corr_grc./pop_corr_mf,[],3)/sqrt(N_repeats);
corr_coef_norm_stderr = std(corr_coef_grc./corr_coef_mf,[],3)/sqrt(N_repeats);

figure, imagesc(f_mf,1:20,corr_coef_norm_stderr);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title({'Standard error','Norm. correlation coefficient'}); 
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,1:20,pop_corr_norm_stderr);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title({'Standard error','Norm. pop. correlation'}); 
xlabel('Fraction active MFs'); ylabel('Number inputs')

% Plot region in which the norm. population correlation <1 but norm. avg.
% correlation coefficient >1 (i.e., decorrelation observed in population
% correlation but not in avg. correlation coefficient
figure, imagesc(f_mf,N_syn,pop_corr_norm<1 & corr_coef_norm>1);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title({'Region in which Norm. pop. correlation<1','but norm. correlation coefficient<1'}); 
xlabel('Fraction active MFs'); ylabel('Number inputs')

%% Plot regions in which difference between MF and GC correlation is statistically significant
% For red outline in bottom panels of Supplementary Fig. 3

% Get p-values for both measures of correlation
pop_corr_p_values = zeros(length(N_syn),length(f_mf));
corr_coef_p_values = zeros(length(N_syn),length(f_mf));
for i = 1:length(N_syn)
    for j = 1:length(f_mf)
        x = pop_corr_mf(i,j,:); y = pop_corr_grc(i,j,:);
        [p,~] = signrank(x(:),y(:));
        pop_corr_p_values(i,j) = p;
        x = corr_coef_mf(i,j,:); y = corr_coef_grc(i,j,:);
        [p,~] = signrank(x(:),y(:));
        corr_coef_p_values(i,j) = p;
    end
end

% with Bonferroni correction
alpha = 0.05/(length(f_mf)*length(N_syn));

% Plot regions in which there is a statistically significant difference
% between MF and GC correlation, according to the two measures
figure, imagesc(f_mf,N_syn,1-(corr_coef_p_values<alpha));
set(gca,'YDir','normal'); set(gca,'FontSize',20);
colormap(gray); caxis([-2,1])
title({'Region of significance (grey)','Norm. correlation coefficient'})
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,N_syn,1-(pop_corr_p_values<alpha));
set(gca,'YDir','normal'); set(gca,'FontSize',20);
colormap(gray); caxis([-2,1])
title({'Region of significance (grey)','Norm. population correlation'})
xlabel('Fraction active MFs'); ylabel('Number inputs')



