% Plot MF statistics, i.e., Fig 1b

% Parameters to plot
f_mf = linspace(.05,.95,19);
f_mf_ix = 6; % corresponds to f_mf = 0.5
sigma = 20; 

% Get intersomatic distances
load ../network_structures/GCLconnectivity_1.mat
[N_mf,~] = size(conn_mat);

dist = @(x,y) sqrt(sum((x-y).^2));

J_mf = zeros(N_mf,N_mf); % Tools for plotting correlations
dists = zeros(N_mf,N_mf); % matrix of inter-glomerular distances
for i = 1:N_mf
    for j = (i+1):N_mf
        dists(i,j) = dist(glom_pos(i,:),glom_pos(j,:));
        J_mf(i,j) = 1;
    end
end

% Generate MF patterns using Dichotomized Gaussian model (Macke et al. 2009)
load(strcat('../input_statistics/mf_patterns_r',num2str(sigma),'.mat'))

N_patt = 640;
scale = normpdf(0,0,sigma)/.9; % Scale factor

t = Rs(:,:,f_mf_ix)' * randn(N_mf,N_patt);
x_mf = (t>-gs(f_mf_ix)*ones(N_mf,N_patt)); 
rho = corrcoef(x_mf');


figure, hold on
h = histc(mean(x_mf,1),linspace(0,1,10));
b=bar(linspace(0,1,10),h); plot([1,1]*f_mf(f_mf_ix),[0,150],'--k','LineWidth',2)
set(b,'EdgeColor','w','FaceColor',[.5,.5,.5])
set(gca,'FontSize',20); xlim([-.05,1.05])
xlabel('Fraction active'), ylabel('Number')

figure, plot(dists(J_mf==1),rho(J_mf==1),'.','Color',[.5,.5,.5],'MarkerSize',20);
hold on, plot(0:.1:80,normpdf(0:.1:80,0,sigma)/scale,'k','LineWidth',3)
plot([0,80],[0,0],'k'); axis([0,80,-.1,1])
set(gca,'FontSize',20)
xlabel('Distance (\mum)'), ylabel('Correlation')