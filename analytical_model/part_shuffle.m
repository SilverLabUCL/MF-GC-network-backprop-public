% Shuffle activity patterns x with population correlation cov_x
% to a LOWER desired population correlation, cov_desired

function [x,cov_new,var_new] = part_shuffle(x,cov_x,cov_desired)

[N,T] = size(x);

if cov_desired >= cov_x
    error('Cannot increase correlations. Use part_shuffle_higher.m instead.')
end

% Find cells that are not always on or always off
% Only shuffles these cells
switch_cells = zeros(N,1);
for i = 1:N
    if length(unique(x(i,:))) > 1
        switch_cells(i)=1;
    end
end

cov_new=cov_x;

while cov_new > cov_desired
    
    % Specify the number of cells that will be shuffled simultaneously
    % Equivalent to adaptive learning rate to speed up code
    % Set to be linear function of (cov_new-cov_desired)/cov_new
    % with slope N/0.5, hard max at 1 and hard min at 0
    num_to_switch_simul = max(1,min(N,round((cov_new-cov_desired)/cov_new*N/0.5)));
    
    % Switch activity of cell i in time T1 with activity of cell i at time
    % T2. Can increase to shuffling larger subsets of time points to speed
    % code, if desired.
    for i = 1:num_to_switch_simul
        randcell = randsample(find(switch_cells),1);
    
        T1 = randsample(T,1);
        % Make sure activity at time T2 is different from time T1
        T2 = randsample(find(x(randcell,:)~=x(randcell,T1)),1); 
        
        x_T1 = x(randcell,T1);
        x_T2 = x(randcell,T2);
        
        x(randcell,T1) = x_T2;
        x(randcell,T2) = x_T1;
    end
     
    % Following can be uncommented to account for hetereogeneity
    % of cell variances
    %x_temp = x;
    %for i = 1:N
    %    if sum(x(i,:)) > 0
    %    x_temp(i,:) = x(i,:)/std(x(i,:));
    %    end
    %end
    %[~,L] = eig(cov(x_temp')); L = real(sqrt(diag(L))); 
    
    [~,L] = eig(cov(x')); L = real(sqrt(diag(L))); 
    
    cov_new = (max(L)/sum(L) - 1./N)/(1-1/N);
end

[~,L] = eig(cov(x'));  L =real(sqrt(diag(L)));
var_new = sum(L.^2);
