% Generates and saves parameters required for generating MF input patterns
% with specified firing rates and correlations using Dichotomized Gaussian
% model from Macke et al. 2009
%
% Requires pop_spike repo from Macke Lab
% see https://bitbucket.org/mackelab/pop_spike/src

warning off
addpath('../../MATLAB/mackelab-pop_spike-a161a8ed79e0/util/')
addpath('../../MATLAB/mackelab-pop_spike-a161a8ed79e0/dich_gauss/')

load ../network_structures/GCLconnectivity_1.mat
[N_mf,N_grc] = size(conn_mat);

% Modify for different input correlations
sigma = 35; % correlation radius, um

% Scale factor so that max correlation is 0.9
% Results in better fits
scale = normpdf(0,0,sigma)/.9;

dist = @(x,y) sqrt(sum((x-y).^2));

dists = zeros(N_mf,N_mf); % matrix of inter-glomerular distances
rho_specified = zeros(N_mf,N_mf); % specified correlation matrix
for i = 1:N_mf
    for j = (i+1):N_mf
        dists(i,j) = dist(glom_pos(i,:),glom_pos(j,:));
        rho_specified(i,j) = normpdf(dists(i,j),0,sigma)/scale;
        rho_specified(j,i) = rho_specified(i,j); 
    end
    rho_specified(i,i) = 1;
end

% Change for more
f_mf = linspace(.05,.95,19);

% Parameters to generate patterns
Rs = zeros(N_mf,N_mf,length(f_mf));
gs = zeros(length(f_mf),1);

for k = 1:length(f_mf)
    f_mf(k)
    
    % Get specified covariance from specified
    cov_specified = rho_specified*(f_mf(k)-f_mf(k)^2);
    if sum(eig(cov_specified)<0) > 0
        error('not positive definite!')
    end
    
    [g,R] = sampleDichGauss01(f_mf(k)*ones(N_mf,1),cov_specified);
    
    Rs(:,:,k) = R;
    gs(k) = g(1);
    
    % Check that all elements of g are the same
    % (because all MFs have same mean)
    if abs(sum(g) - g(1)*N_mf) > sqrt(eps)
        error('g should be homogenous')
    end
    
end

save(strcat('mf_patterns_r',num2str(sigma),'.mat'),'Rs','gs')

