% Computes population covariance as function of fraction observed GCs
% i.e., Fig. 3e

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

N_mf = 187; N_grc = 487;
N_patt = 640;

theta_initial = 3; NADT = 0;

% Modify for different input correlations or different Nsyn
sigma = 20; % correlation radius, um
Nsyn = 4;
load(strcat('../network_structures/GCLconnectivity_',int2str(Nsyn),'.mat'))
conn_mat = double(conn_mat);

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

% Different sizes of subpopulation
N_grc_sub = round(linspace(2,N_grc,10));
N_repeats = 25; % Different trials 
N_subsets = 10; % Number of random samples assessed in each trial

pop_corr_norm_mean = zeros(length(N_grc_sub),1);
pop_corr_norm_median = zeros(length(N_grc_sub),1);
pop_corr_norm_std = zeros(length(N_grc_sub),1);
tic

for k0 = 1:length(N_grc_sub)
    [k0,toc]

    pop_corr_norm_mean_temp =  zeros(length(f_mf),1);
    pop_corr_norm_std_temp =  zeros(length(f_mf),1); 
    
    
    for k2 = 1:length(f_mf)
        
        pop_corr_mf_temp =  zeros(N_repeats*N_subsets,1);
        pop_corr_grc_temp =  zeros(N_repeats*N_subsets,1); 
        ix_mf = 1; ix_grc = 1;

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
            in = 4/Nsyn*conn_mat'*x_mf; % input 
            x_grc = max(in-theta,0); % Output GC activity

            x_mf = double(x_mf); x_grc = double(x_grc);

            % Get average population
            for k4 = 1:N_subsets
                N = min(N_grc_sub(k0),N_mf);
                subset = randsample(N_mf,N);
                x_mf_sub = x_mf(subset,:);
                C_mf = cov(x_mf_sub');
                [~,L_mf] = eig(C_mf); L_mf = diag(L_mf);
                L_mf = real(sqrt(L_mf)); 
                pop_corr_mf_temp(ix_mf) = (max(L_mf)/sum(L_mf) - 1/N)/(1-1/N);
                ix_mf = ix_mf + 1;
            end

            if max(x_grc(:)) == 0
                for k4 = 1:N_subsets
                    pop_corr_grc_temp(ix_grc) = NaN; 
                    ix_grc = ix_grc+1;
                end
            else

                for k4 = 1:N_subsets
                    N=N_grc_sub(k0);
                    subset = randsample(N_grc,N);
                    x_grc_sub = x_grc(subset,:);
                    C_grc = cov(x_grc_sub');
                    [~,L_grc] = eig(C_grc); L_grc = diag(L_grc);
                    L_grc = real(sqrt(L_grc)); 
                    pop_corr_grc_temp(ix_grc)=(max(L_grc)/sum(L_grc) - 1/N)/(1-1/N);
                    ix_grc = ix_grc+1;
                end
            end

        end
        
        
        pop_corr_norm_mean_temp(k2) = nanmean(pop_corr_grc_temp./pop_corr_mf_temp);
        
        % Get std over different (1) input patterns and (2) choices of
        % subpopulation size, for fixed f_mf
        pop_corr_norm_std_temp(k2) = nanstd(pop_corr_grc_temp./pop_corr_mf_temp);

    end

    pop_corr_norm_mean(k0) = nanmean(pop_corr_norm_mean_temp);

    % Average std over f_mf
    pop_corr_norm_std(k0) = nanmean(pop_corr_norm_std_temp);
end


meanPlusSTD = pop_corr_norm_mean'+pop_corr_norm_std';
meanMinusSTD = pop_corr_norm_mean'-pop_corr_norm_std';
figure, fill( [N_grc_sub fliplr(N_grc_sub)]/N_grc,  [meanPlusSTD fliplr(meanMinusSTD)], [0.8,0.8,0.8]);
hold on, plot(N_grc_sub/N_grc,pop_corr_norm_mean,'-*k','LineWidth',3)
plot([0,1],[1,1],'k','LineWidth',1.5)
axis([0.1150,1,0.6,1.4])
set(gca,'FontSize',20)
xlabel('Fraction observed GCs')
ylabel('Norm. pop. correlation')
