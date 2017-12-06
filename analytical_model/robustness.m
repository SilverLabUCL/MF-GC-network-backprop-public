% This script calculates robustness of different properties 
% as a function of the correlation radius
% This section generates Fig. 1g (bottom panel)

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

robustness_speed = [];
for sigma = 0:5:30

    load(strcat('results_bp/grc_toy_r',num2str(sigma),'.mat'))

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
    

    temp = (1./grc)./(1./mf)-1;
    temp = temp(isfinite(temp)); % only count cases that have converged
    
    robustness_speed = [robustness_speed, sum(temp(:)>0)/length(temp(:))];

end

figure, plot(0:5:30,robustness_speed,'-ok','LineWidth',3,'MarkerFaceColor','k')
axis([-5,35,.2,.8])
xlabel('Correlation radius (\mum)'), ylabel('Robustness of learning')
set(gca,'FontSize',20)

%% This section generates Fig. 2b,d and Fig. 3b (bottom panel)

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

N_mf = 187; N_grc = 487;
N_patt = 640;

theta_initial = 3; NADT = 0;

N_repeats = 25;

robustness_sp = [];
robustness_total_var = [];
robustness_pop_corr = [];

for sigma = 0:5:30
    sigma
    
    % Total variance
    total_var_norm = nan(length(N_syn),length(f_mf),N_repeats);
    pop_corr_norm =  nan(length(N_syn),length(f_mf),N_repeats);
    sp_norm = zeros(length(N_syn),length(f_mf),N_repeats);

    for k1 = N_syn
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
                
                sp_norm(k1,k2,k3) = sp_grc./sp_mf;
                total_var_norm(k1,k2,k3) = total_var_grc./total_var_mf;
                pop_corr_norm(k1,k2,k3) = pop_corr_grc./pop_corr_mf;
                
            end
        end
    end
    
    temp = mean(sp_norm,3)-1;
    temp = temp(isfinite(temp)); % only count cases that have converged
    
    robustness_sp = [robustness_sp, sum(temp(:)>0)/length(temp(:))];
    
    temp =  mean(total_var_norm,3)-1;
    temp = temp(isfinite(temp)); % only count cases that have converged
   
    robustness_total_var = [robustness_total_var, sum(temp(:)>0)/length(temp(:))];
    
    temp = mean(pop_corr_norm,3)-1;
    temp = temp(isfinite(temp)); % only count cases that have converged
   
    robustness_pop_corr = [robustness_pop_corr, sum(temp(:)<0)/length(temp(:))];
end

figure, plot(0:5:30,robustness_sp,'-ok','LineWidth',3,'MarkerFaceColor','k')
axis([-5,35,0.6,1])
xlabel('Correlation radius (\mum)'), ylabel('Robustness of sparsening')
set(gca,'FontSize',20)

figure, plot(0:5:30,robustness_total_var,'-ok','LineWidth',3,'MarkerFaceColor','k')
axis([-5,35,0,1])
xlabel('Correlation radius (\mum)'), ylabel('Robustness of expansion')
set(gca,'FontSize',20)

figure, plot(0:5:30,robustness_pop_corr,'-ok','LineWidth',3,'MarkerFaceColor','k')
axis([-5,35,0,.3])
xlabel('Correlation radius (\mum)'), ylabel('Robustness of decorrelation')
set(gca,'FontSize',20)

%% This section generates Fig. 4c (bottom panel)

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

% Modify for different input correlations
sigma = 20; % correlation radius, um

robustness_speed = [];
for theta = [0:.5:2,2.25:.25:3.75]
    theta
    
    if theta == 3
        load(strcat('results_bp/grc_toy_r',num2str(sigma),'.mat'))
    else
        load(strcat('results_bp_th/grc_toy_r',num2str(sigma),'_',num2str(theta,'%.2f'),'.mat'))
    end

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
    

    temp = (1./grc)./(1./mf)-1;
    robustness_speed = [robustness_speed, nansum(temp(:)>0)/numel(temp(:))];

end

figure, plot([0:.5:2,2.25:.25:3.75],robustness_speed,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, plot([3,3],[0,1],':k','LineWidth',2), axis([0,4,0,1])
xlabel('Correlation radius (\mum)'), ylabel('Robustness of learning')
set(gca,'FontSize',20)

%% This section generates Fig. 4c (top panel)

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

N_mf = 187; N_grc = 487;
N_patt = 640;

N_repeats = 25;

% Modify for different input correlations
sigma = 20; % correlation radius, um

robustness_total_var = [];
robustness_pop_corr = [];

for theta = [0:.5:2,2.25:.25:3.75]
    theta
    
    % Total variance
    total_var_norm = nan(length(N_syn),length(f_mf),N_repeats);
    pop_corr_norm =  nan(length(N_syn),length(f_mf),N_repeats);
    
    for k1 = N_syn
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
                
                if max(x_grc(:)) > 0

                    % Eigenvalues of covariance matrix of GC patterns
                    C_grc = cov(x_grc'); 
                    [~,L_grc] = eig(C_grc);  L_grc = diag(L_grc);
                    L_grc =real(sqrt(L_grc));

                    % Total variance of GCs
                    total_var_grc = sum(L_grc.^2); % total variation

                    % Population correlation of GCs
                    pop_corr_grc = (max(L_grc)/sum(L_grc) - 1./N_grc)/(1-1/N_grc);

                else
                    pop_corr_grc = NaN; total_var_grc = NaN;
                end
                
                total_var_norm(k1,k2,k3) = total_var_grc./total_var_mf;
                pop_corr_norm(k1,k2,k3) = pop_corr_grc./pop_corr_mf;
                
            end
        end
    end
    
    temp =  mean(total_var_norm,3)-1;
    temp = temp(isfinite(temp)); % only count cases that have converged
   
    robustness_total_var = [robustness_total_var, sum(temp(:)>0)/length(temp(:))];
    
    temp = mean(pop_corr_norm,3)-1;
    temp = temp(isfinite(temp)); % only count cases that have converged
   
    robustness_pop_corr = [robustness_pop_corr, sum(temp(:)<0)/length(temp(:))];
end

green = [0,.3,0]; purple = [.4,0,.6];
figure, plot([0:.5:2,2.25:.25:3.75],robustness_total_var,'-o','Color',green,'LineWidth',3,'MarkerFaceColor',green)
hold on, plot([0:.5:2,2.25:.25:3.75],robustness_pop_corr,'-o','Color',purple,'LineWidth',3,'MarkerFaceColor',purple)
plot([3,3],[0,1],':k','LineWidth',2), axis([0,4,0,1])
xlabel('Correlation radius (\mum)'), ylabel('Robustness')
set(gca,'FontSize',20)



