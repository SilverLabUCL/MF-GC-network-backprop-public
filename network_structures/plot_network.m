% Visualize MF-GC network, i.e., Fig. 1a

% Parameters to plot
N_syn = 4; % number inputs

load(strcat('GCLconnectivity_',int2str(N_syn),'.mat'))

[N_mf,N_grc] = size(conn_mat);

dist = @(x,y) sqrt(sum((x-y).^2));
%%
figure, hold on
% Plot connections and get dendrtic lengths
dend = [];
for k1 = 1:N_mf
    for k2 = 1:N_grc
        if conn_mat(k1,k2)==1
            h = plot3([glom_pos(k1,1),grc_pos(k2,1)],[glom_pos(k1,2),grc_pos(k2,2)],[glom_pos(k1,3),grc_pos(k2,3)],'Color',[0.6,0.6,0.6],'LineWidth',1.5);
            dend = [dend, dist(glom_pos(k1,:),grc_pos(k2,:))];
        end
    end
end


[x,y,z] = sphere; r = 2;

% Plot MF glomeruli
for k1 = 1:N_mf
    h=surfl(r*x+glom_pos(k1,1),r*y+glom_pos(k1,2),r*z+glom_pos(k1,3));
    set(h,'FaceColor',[0,0,1]);
    set(h,'EdgeColor','none');
    light('Position',[-100 0 0],'Style','infinite');
    h.SpecularStrength = 0.03*(1+rand/2); h.AmbientStrength = .3*(1+rand/2); h.DiffuseStrength = 0.3*(1+rand/2);
end

% Plot GCs
for k1 = 1:N_grc
    h=surfl(r*x+grc_pos(k1,1),r*y+grc_pos(k1,2),r*z+grc_pos(k1,3));
    set(h,'FaceColor',[1,0,0]);
    set(h,'EdgeColor','none');
    light('Position',[-100 0 0],'Style','infinite');
    h.SpecularStrength = 0.03*(1+rand/2); h.AmbientStrength = .3*(1+rand/2); h.DiffuseStrength = 0.3*(1+rand/2);
end

% Plot 20 um scale bar
hold on, plot3([-30,-30],[35,35],[-30,-10],'k','LineWidth',2)
set(gca,'FontSize',20)

%% Plot distribution of dendritic lengths
bins = 0:2.5:40; h = histc(dend,bins);
figure, b = bar(bins,h); xlim([0,42])
hold on, plot(median(dend),950,'v','Color',[.35,0,.5])
set(gca,'FontSize',20)
xlabel('Dendritic length (\mum)')
ylabel('Number')
set(b,'EdgeColor','w','FaceColor',[.35,0,.5])
