% Generates Supplementary Fig. 6

%% Babadi & Sompolinksy method

N_mf = 187; N_patt = 640;

% Tools to avoid double counting correlations
J_mf = zeros(N_mf,N_mf);
for i = 1:N_mf
    for j = (i+1):N_mf
        J_mf(i,j) = 1;
    end
end

Delta_S = 0.3; P = 10;
num_in_cluster = 640/P;

x_mf = zeros(N_mf,N_patt); ix = 1;
for patt_c_ix = 1:P
    % Cluster centre
    patt_c = (rand(N_mf,1)<0.5);
    % Other patterns in cluster: randomly flip states
    % with probability Delta_S/2
    for patt_n_ix = 1:64        
        flip = (rand(N_mf,1)<Delta_S/2);
        patt_n = patt_c.*(1-flip)+(1-patt_c).*flip;
        x_mf(:,ix)=patt_n;
        ix = ix+1;
    end
end

% Double check that f_mf is ~0.5
mean(x_mf(:))

C = corrcoef(x_mf'); figure, imagesc(C);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
xlabel('MF number'); ylabel('MF number')
title({'Babadi & Sompolinsky','Correlation coefficient'})
caxis([-.6,1])

figure, hold on, plot([1,1]*mean(C(J_mf==1)),[0,10000],'k');
hist(C(J_mf==1)); h = findobj(gca,'Type','patch');
h.FaceColor = [1 .8 0]; h.EdgeColor = 'w';
set(gca,'FontSize',20); set(gca,'FontSize',20)
xlabel('MF number'); ylabel('Number')
title({'Babadi & Sompolinsky','Correlation coefficient'})
xlim([-1,1])

%% Our method
% Change sigma to compare independent vs. correlated

f_mf = linspace(.05,.95,19); % fraction active MFs
f_mf_ix = 10; % corresponding to f_mf = 0.5

% Modify for different input correlations
sigma = 0; % correlation radius, um

% Input MF patterns
if sigma == 0 % Independent case
    x_mf = zeros(N_mf,N_patt);
    for i = 1:N_patt
        mf_on = randsample(N_mf,round(f_mf(f_mf_ix)*N_mf));
        x_mf(mf_on,i) = 1.;
    end
elseif sigma >0 % Correlated case -- generated following Macke et al. 2009
    load(strcat('../input_statistics/mf_patterns_r',num2str(sigma),'.mat'))
    R = Rs(:,:,f_mf_ix); g = gs(f_mf_ix);
    t = R' * randn(N_mf,N_patt);
    S = (t>-g(1)*ones(N_mf,N_patt)); 
    x_mf = S;
end

% Double check that f_mf is ~0.5
mean(x_mf(:))

C = corrcoef(x_mf'); figure, imagesc(C);
set(gca,'YDir','normal'); set(gca,'FontSize',20)
xlabel('MF number'); ylabel('MF number')
if sigma == 0
    title({'Independent MFs','Correlation coefficient'})
    caxis([-.2,.2])
else
    title({strcat('Correlated MFs \sigma=',num2str(sigma),'\mum'),'Correlation coefficient'})
    caxis([-.2,1])
end

figure, hold on, plot([1,1]*mean(C(J_mf==1)),[0,10000],'k');
hist(C(J_mf==1)); h = findobj(gca,'Type','patch');
h.FaceColor = [1 .8 0]; h.EdgeColor = 'w';
set(gca,'FontSize',20); set(gca,'FontSize',20)
xlabel('MF number'); ylabel('Number')
if sigma == 0
    title({'Independent MFs','Correlation coefficient'})
    xlim([-.2,.2])
else
    title({strcat('Correlated MFs \sigma=',num2str(sigma),'\mum'),'Correlation coefficient'})
    xlim([-1,1])
end