% Generates Supplementary Fig. 5
% This section generates Supplementary Fig. 5a

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

N_mf = 187; N_grc = 487;
N_patt = 640;

theta_initial = 3; NADT = 0;

N_repeats = 25;

% Modify for different input correlations
sigma = 30; % correlation radius, um

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

% fraction active GCs
f_grc = zeros(length(N_syn),length(f_mf),N_repeats); 

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
            
            f_grc(k1,k2,k3) = mean(sum((x_grc>0),1)/N_grc);
            
        end
    end
end

% get RMS error
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


figure, imagesc(f_mf,1:20,mean(f_grc,3));
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Fraction active GCs'); caxis([0,1])
xlabel('Fraction active MFs'); ylabel('Number inputs')
hold on, plot(y2,x,'-w','LineWidth',5)

%% This section generates Supplementary Fig. 5b,c
% Iterates over all input statistics (f_mf and sigma)
% as well as over all Nsyn

speed_norm_all = []; f_grc_all = [];
for sigma = 0:5:30

    % use RMS error to get norm. speed
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
    
    speed_norm = (1./grc)./(1./mf);
    speed_norm_all = [speed_norm_all;speed_norm(:)];
    
    % Next, get f_grc for all Nsyn and all f_mf
    f_grc = zeros(length(N_syn),length(f_mf),N_repeats); 
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

                f_grc(k1,k2,k3) = mean(sum((x_grc>0),1)/N_grc);

            end
        end
    end
    
    f_grc_mean = mean(f_grc,3);
    f_grc_all = [f_grc_all; f_grc_mean(:)];
    
end

figure, plot(f_grc_all(speed_norm_all<=1),speed_norm_all(speed_norm_all<=1),'.','Color',[.5,.5,.5],'MarkerSize',10)
hold on, plot(f_grc_all(speed_norm_all>1),speed_norm_all(speed_norm_all>1),'.','Color',[.2,.2,.5],'MarkerSize',10)
plot([0,1],[1,1],'k','LineWidth',2)
set (gca,'FontSize',20)
xlabel('Fraction active GCs'); ylabel('Norm speed')

figure, subplot(2,1,1), h = histc(f_grc_all,0:.1:1); p1 = bar(0:.1:1,h/sum(h));
set(gca,'FontSize',20); ylabel('Fraction'); axis([-.05,1.05,0,0.4]);
set(p1,'LineStyle','-','LineWidth',3,'EdgeColor','w','FaceColor',[.5,.5,.5])

subplot(2,1,2), h=histc(f_grc_all(speed_norm_all>1),0:.1:1); p2 = bar(0:.1:1,h/sum(h));
set (gca,'FontSize',20); xlabel('Fraction active GCs'); ylabel('Fraction'); axis([-.05,1.05,0,0.4]);
set(p2,'LineStyle','-','LineWidth',3,'EdgeColor','w','FaceColor',[.2,.2,.5])
