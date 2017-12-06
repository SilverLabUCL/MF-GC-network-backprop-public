% Plot results from backprop learning
% i.e., Fig. 1f

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs

% Modify for different input correlations
sigma = 0; % correlation radius, um

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

figure, imagesc(f_mf,1:20,speed_mf);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Raw MF speed');
xlabel('Fraction active MFs'); ylabel('Number inputs')

figure, imagesc(f_mf,1:20,speed_grc);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Raw GC speed');
xlabel('Fraction active MFs'); ylabel('Number inputs')

% Plot Fig. 1f
figure, y=imagesc(f_mf,1:20,speed_norm);
set(y,'AlphaData',~isnan(speed_norm))
set(gca,'YDir','normal'); set(gca,'FontSize',20)
title('Normalized learning speed');
xlabel('Fraction active MFs'); ylabel('Number inputs')
hold on, plot(y2,x,'Color',[.3,.8,1],'LineWidth',5)


%% Here plot Fig. 1d

% Parameters to plot

Nsyn = 4; % number inputs
f_mf_ix = 10; % corresponds to f_mf = 0.5

figure, hold on
plot([0,T],[thresh,thresh],':','Color',[.5,.5,.5],'LineWidth',3)
plot(reshape(err_grc(4,10,:),1,T),'r','LineWidth',3)
plot(reshape(err_mf(4,10,:),1,T),'b','LineWidth',3)
axis([0,2000,0,.4])
xlabel('Training epochs'); ylabel('RMS error')
set(gca,'FontSize',20)


%% Plot speed as fn of correlation radius
% i.e., for Fig 1e,g (top panel)

N_syn = 1:20; % Synaptic connectivity / Number inputs
f_mf = linspace(.05,.95,19); % fraction active MFs



speed_mf = []; 
speed_grc_sparse = []; speed_grc_dense = [];
speed_norm_sparse = []; speed_norm_dense = [];

for sigma = 0:5:30
    sigma
    
    load(strcat('results_bp/grc_toy_r',num2str(sigma),'.mat'))
    
    err_grc = err_rms_grc; err_mf = err_rms_mf;
    thresh = 0.2; % threshold for determining learning speed
    T = size(err_grc,3);

    % Get number of epochs until learning is complete
    % both for GC- and for MF-based learning
    for j = [4,16]
        grc = nan(length(f_mf),1);
        mf = nan(length(f_mf),1);
        for k = 1:length(f_mf)
            e_grc = reshape(err_grc(j,k,:),1,T);
            e_mf = reshape(err_mf(j,k,:),1,T);

            temp = find(e_grc<=thresh);
            if numel(temp) > 0
                grc(k) = temp(1);
            end

            temp = find(e_mf<=thresh);
            if numel(temp) > 0
                mf(k) = temp(1);
            end
        end
        
        if j == 4
            % Learning speed is 1 / number epochs to reach threshold
            speed_grc_sparse = [speed_grc_sparse, nanmedian(1./grc)];
            speed_norm_sparse = [speed_norm_sparse, nanmedian((1./grc)./(1./mf))];
        elseif j == 16
            % Learning speed is 1 / number epochs to reach threshold
            speed_grc_dense = [speed_grc_dense, nanmedian(1./grc)];
            speed_norm_dense = [speed_norm_dense, nanmedian((1./grc)./(1./mf))];
        end
        
    end
    
    % Learning speed is 1 / number epochs to reach threshold
    speed_mf = [speed_mf, nanmedian(1./mf)];    
    
end

figure, plot(0:5:30,speed_norm_sparse,'-ok','LineWidth',3,'MarkerFaceColor','k')
hold on, plot(0:5:30,speed_norm_dense,'--ok','LineWidth',3,'MarkerFaceColor','k')
plot([-5,35],[1,1],'k'); axis([-5,35,0,5])
xlabel('Correlation radius (\mum)'), ylabel('Norm. speed')
set(gca,'FontSize',20)


figure, plot(0:5:30,speed_grc_sparse,'-ob','LineWidth',3,'MarkerFaceColor','b')
hold on, plot(0:5:30,speed_grc_dense,'--ob','LineWidth',3,'MarkerFaceColor','b')
 plot(0:5:30,speed_mf,'-or','LineWidth',3,'MarkerFaceColor','r')
xlabel('Correlation radius (\mum)'), ylabel('Speed')
set(gca,'FontSize',20)

