% Shuffle activity patterns x with population correlation cov_x
% to a HIGHER desired population correlation, cov_desired

function [x,cov_new,var_new] = part_shuffle_higher(x,cov_x,cov_desired)

[N,T] = size(x);

if cov_desired <= cov_x
    error('Cannot decrease correlations. Use part_shuffle.m instead.')
end

% Find cells that are not always on or always off
% Only shuffles these cells
switch_cells = zeros(N,1);
for i = 1:N
    if length(unique(x(i,:))) > 1
        switch_cells(i)=1;
    end
end
whichswitch =  find(switch_cells);

% means of cells
mu = mean(x,2);

cov_new=cov_x;

while cov_new < cov_desired
    
    % Switch time pairs of all cells so that the lower values are in one
    % time bin, and the higher values are in the other time bin
    Ts = randsample(T,2);  % 1st is low, 2nd is high
    for i = 1:sum(switch_cells)
        x_T1 = x(whichswitch(i),Ts(1)); 
        x_T2 = x(whichswitch(i),Ts(2)); 
        % Switch activities if they are in the wrong bin
        % T1 should have lower value, T2 should have higher value
        if x_T1 > mu(whichswitch(i)) && x_T2 < mu(whichswitch(i))
            x(whichswitch(i),Ts(1)) = x_T2; 
            x(whichswitch(i),Ts(2)) = x_T1;
        end
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
