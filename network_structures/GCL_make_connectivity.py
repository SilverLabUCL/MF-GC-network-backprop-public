
import numpy as np
import pickle as pkl
from datetime import datetime


# Following generates uniformly distributed granule positions of fixed density within ball of diameter diam
def generate_grc_positions(density,diam):
	# The average number of granule cells within a diam x diam x diam cube based on given density is:
	avg_num_incube = int(diam**3*density) 
	grc_pos = np.random.uniform(low=-diam/2,high=diam/2,size=(avg_num_incube,3))
	# Delete all granule cells that do not lie within ball:
	grc_pos = grc_pos[np.where(np.sqrt(grc_pos[:,0]**2+grc_pos[:,1]**2+grc_pos[:,2]**2)<=(diam)/2)[0],:]
	return grc_pos

# Following generates glomeruli positions 
def generate_glom_positions(p_num_glom,glom_density,dx,dy,dz,diam):
	# Average number of glomeruli on each mossy fiber:
	avg_glom_per_mf = (p_num_glom * np.array(range(1,6))).sum() 
	# Average number of glomeruli in bug_diam x big_diam x big_diam cube:
	big_diam = diam*3; # embed in big cube to avoid boundary effects
	avg_total_glom_incube = big_diam**3*glom_density 
	# Average number of mossy fibers in big_diam x big_diam x big_diam cube:
	avg_mf_incube = int(avg_total_glom_incube/avg_glom_per_mf)
	# Generate the number of glomeruli that each mossy fiber has:
	num_glom = draw_from_p_num_glom(p_num_glom,avg_mf_incube)
	glom_mf_id = np.zeros((num_glom.sum()),int) # vector that indexes which mossy fiber is associated with each glomeruli 
	glom_pos = np.zeros((num_glom.sum(),3)); ix = 0 
	for k in range(0,avg_mf_incube):
		glom_mf_id[ix:ix+num_glom[k]] = k
		# For each mossy fiber, the first glomerulus is randomly (uniformly) positioned
		glom_pos[ix,:] = np.random.uniform(low=-big_diam/2,high=big_diam/2,size=(3))
		for j in range(1,int(num_glom[k])):
			# Distance of each glomerulus belonging to mossy fiber k is exponentially distributed away from the previous glomerulus
			glom_pos[ix+j] = glom_pos[ix+j-1,:] + [(-1)**np.round(np.random.uniform())*np.random.exponential(scale=dx),(-1)**np.round(np.random.uniform())*np.random.exponential(scale=dy),(-1)**np.round(np.random.uniform())*np.random.exponential(scale=dz)]
		ix = ix + num_glom[k]
	# Delete all glomeruli that do not lie within ball of diameter diam:
	which_glom_in_ball = np.where(np.sqrt(glom_pos[:,0]**2+glom_pos[:,1]**2+glom_pos[:,2]**2)<=(diam)/2)[0]
	glom_pos = glom_pos[which_glom_in_ball,:]; glom_mf_id = glom_mf_id[which_glom_in_ball]; glom_mf_id = renumber(glom_mf_id)
	return glom_pos, glom_mf_id

def renumber(glom_mf_id):
	N_glom = glom_mf_id.shape[0]; mfs = np.unique(glom_mf_id)
	N_mf = mfs.shape[0]
	glom_mf_id_new = np.zeros((N_glom),int)
	count = 0
	for k in range(N_mf):
		glom_mf_id_new[np.where(glom_mf_id==mfs[k])[0]] = k
	return glom_mf_id_new

def draw_from_p_num_glom(p,numsamples):
	values = np.arange(1,len(p)+1.)
	p_cum = np.cumsum(np.append([0],p[0:-1]))
	samples = np.zeros((numsamples))
	randvars = np.random.uniform(0,1,numsamples)
	for k in range(0,numsamples):
		samples[k] = values[np.where(randvars[k]>p_cum)[0][-1]]
	return samples

def get_degreedist(glom_mf_id,N_grc,N_glom,d):
	ddist = np.zeros((N_glom),int)
	N_mf = np.unique(glom_mf_id).shape[0]
	for g in range(N_grc):
		# choose d random mossy fibers, without replacement
		mf_chosen = np.random.choice(range(N_mf),size=d,replace=False)
		mf_chosen.sort()
		# for each mossy fiber, choose a random granule cell
		for j in range(d):
			gl = np.random.choice(np.where(glom_mf_id == mf_chosen[j])[0])
			ddist[gl] = ddist[gl] + 1
	return ddist

# Find granule cell that is closest to being dlen away from a given glomeruli
# that is also not connected to that mossy fiber or that has more than d connections
def closest_allowed_grc(attached_grcs,grc_pos,this_glom_pos,conn_mat,d,dlen):
	N_grc = grc_pos.shape[0]
	grcs_not_yet_attached = [n for n in range(N_grc) if n not in attached_grcs]
	grcs_not_yet_full = [n for n in range(N_grc) if conn_mat[:,n].sum() < d]
	grcs_available = [n for n in grcs_not_yet_full if n in grcs_not_yet_attached]
	dists_from_glom = np.sqrt(((grc_pos-this_glom_pos)**2).sum(axis=1))
	dists = np.abs(dists_from_glom-dlen)
	if len(dists[grcs_available])>0:
		grc_closest = grcs_available[np.argmin(dists[grcs_available])]
	else:
		grc_closest = -1
	return grc_closest

# Checks that connectivity is valid, or throws an error if:
#   1. no granule cell is connected to multiple glomeruli from the same mossy fiber
#   2. degree distribution of glomeruli does not match ddist
#   3. any granule cell has >d connections
def check_valid_connectivity(conn_mat,ddist,glom_mf_id,d):
	N_mf = np.unique(glom_mf_id).shape[0]
	for mf in range(N_mf):
		gloms = np.where(glom_mf_id == mf)[0]
		assert(np.all(conn_mat[gloms,:].sum(axis=0)<=1)),'Mossy fiber '+str(mf)+' has multiple connections to same granule cell.'
	assert(np.all(conn_mat.sum(axis=1) - ddist==0)),'Glomeruli degree distribution does not match desired distribution.'
	assert(np.all(conn_mat.sum(axis=0) == d)),'Granule cells do not all have ' +str(d)+ ' dendrites.'

# Connection the following swapping procedure:
#  1. Choose an incomplete glomerulus (i) and an incomplete granule cell (A) connected to it
#  2. Choose a complete glomerulus (ii) that is not connected to grc A through any mossy fibers
#  3. Choose a complete granule cell (B) that is connected to glom ii
#  4. Delete connection ii -> B 
#  5. Connect i -> B and ii -> A
# Choose the swap that minimizes (dist(i,B)-dlen)^2 + (dist(ii,A)-dlen)^2
def optswap(glom_incomplete,grcs_incomplete,grcs_swappable,grc_pos,glom_pos,conn_mat,dlen):
	N_glom = glom_pos.shape[0]; N_grc = grc_pos.shape[0]
	# First make list of glomeruli that can be swapped from an eligible granule cell to an incomplete granule cell
	gloms_swappable = []
	for j in range(len(grcs_incomplete)):
		gloms = [];
		for k in range(len(grcs_swappable)):
			# glomeruli that are connected to grcs_swappable[j] but not to grcs_incomplete[j] (through any gloms on that mf)
			gloms.append([gl for gl in range(N_glom) if (conn_mat[gl,grcs_swappable[k]]==1 and conn_mat[np.where(glom_mf_id==glom_mf_id[gl])[0],grcs_incomplete[j]].sum()==0)])
		gloms_swappable.append(gloms)
	# Next, make table of 
	deviation = np.zeros((len(grcs_incomplete),len(grcs_swappable)),float)
	index_ii = np.zeros((len(grcs_incomplete),len(grcs_swappable)),int)
	glom_i = glom_pos[glom_incomplete]
	for i in range(len(grcs_incomplete)):
		grc_A = grc_pos[grcs_incomplete[i]] 
		for j in range(len(grcs_swappable)):
			grc_B = grc_pos[grcs_swappable[j]]
			dist_iB = np.sqrt(((glom_i - grc_B)**2).sum())
			gloms = gloms_swappable[i][j]
			deviation_temp = np.zeros((len(gloms)),float)
			for k in range(len(gloms)):
				glom_ii = glom_pos[gloms[k]]
				dist_iiA = np.sqrt(((glom_ii - grc_A)**2).sum())
				deviation_temp[k] = (dlen - dist_iB)**2 + (dlen - dist_iiA)**2
			deviation[i,j] = deviation_temp.min()
			index_ii[i,j] = deviation_temp.argmin()
	# Best swap:
	index = np.where(deviation == deviation.min())
	index_A = index[0][0]; index_B = index[1][0]
	opt_grc_A = grcs_incomplete[index_A]
	opt_grc_B = grcs_swappable[index_B]
	opt_glom_ii = gloms_swappable[index_A][index_B][index_ii[index_A,index_B]]
	# Sanity check on selected gloms and grcs
	assert(conn_mat[opt_glom_ii,opt_grc_B]==1),'Something is wrong with function optswap: glom ii is not connected to grc B.'
	assert(conn_mat[glom_incomplete,opt_grc_A]==1 ),'Something is wrong with function optswap: glom i is not connected to grc A.'
	assert(conn_mat[opt_glom_ii,opt_grc_A]==0),'Something is wrong with function optswap: glom ii is connected to grc A.'
	# Update connectivity matrix to reflect swap
	conn_mat[opt_glom_ii,opt_grc_B] = 0
	conn_mat[opt_glom_ii,opt_grc_A] = 1
	conn_mat[glom_incomplete,opt_grc_B] = 1
	return conn_mat

def shuffle_conns(grc_pos,glom_pos,d,dlen,gloms_incomplete,conn_mat,ddist):
	N_grc = grc_pos.shape[0]
	for gl in gloms_incomplete:
		incomplete_conns = ddist[gl] - conn_mat[gl,:].sum()
		for conn in range(incomplete_conns):
			# list of granule cells that are incomplete (have <d dendrites)
			grcs_incomplete = [n for n in range(N_grc) if conn_mat[:,n].sum() < d]
			# list of granule cells that could be swapped, i.e., have = d dendrites
			# and that are not connected to the glomerulus in question
			grcs_swappable = [n for n in range(N_grc) if (n not in grcs_incomplete and conn_mat[gl,n]==0 )]
			# find optimal "swap", update connectivity matrix
			conn_mat = optswap(gl,grcs_incomplete,grcs_swappable,grc_pos,glom_pos,conn_mat,dlen)
	#
	return conn_mat

# Connect granule cells and glomeruli according desired degree distribution, to fulfill all desired properties
def alg_connections(grc_pos,glom_pos,glom_mf_id,d,dlen):
	N_grc = grc_pos.shape[0]; N_glom = glom_pos.shape[0]
	ddist = get_degreedist(glom_mf_id,N_grc,N_glom,d)
	#
	conn_mat = np.zeros((N_glom,N_grc),int)
	for conn in range(1,ddist.max()+1):
		for gl in range(N_glom):
			if conn <= ddist[gl]:
				mf = glom_mf_id[gl]
				# all glom on same mf
				glom_on_mf = np.where(glom_mf_id==mf)[0]
				# find all grcs that are attached to that mf
				attached_grcs = np.where(conn_mat[glom_on_mf].sum(axis=0))[0]
				this_grc = closest_allowed_grc(attached_grcs,grc_pos,glom_pos[gl,:],conn_mat,d,dlen)
				if this_grc >= 0:
					conn_mat[gl,this_grc] = 1
	#
	# If ddist is not fully satisfied, label which glomeruli are incompletely connected
	if not np.all(conn_mat.sum(axis=1) - ddist==0):
		gloms_incomplete = np.where(conn_mat.sum(axis=1) != ddist)[0]
		conn_mat = shuffle_conns(grc_pos,glom_pos,d,dlen,gloms_incomplete,conn_mat,ddist)
	# Make sure connectivity matrix is valid
	check_valid_connectivity(conn_mat,ddist,glom_mf_id,d)
	return conn_mat

# First, set up positions of the glomeruli and granule cells in a sphere
glom_density = 6.6*10**-4 # per cubic um
glom_dx = 60; glom_dy = 20; glom_dz = 2;
grc_density = 1.9*10**-3 # per cubic um

diam = 80 # diameter of ball in um
dlen = 15 # target length of granule cell dendrites in um
N_syn_range = range(1,21) # range of dendrites/synapses per cell

# This gives the probability of each mossy fiber having 1, 2, 3, etc. glomeruli
# Taken from Sultan et al. 
p_num_glom = np.array([0.0,45.0,17.0,8.0,5.0])
p_num_glom = p_num_glom / p_num_glom.sum()
assert ( p_num_glom.sum() == 1)

# Find randomly placed positions of the granule cells and glomeruli
grc_pos = generate_grc_positions(grc_density,diam)
glom_pos, glom_mf_id = generate_glom_positions(p_num_glom,glom_density,glom_dx,glom_dy,glom_dz,diam)
N_glom = glom_pos.shape[0]; N_grc = grc_pos.shape[0]

for d in N_syn_range:
	# Connect granule cells to each other
	print('Number of dendrites: '+str(d))
	startTime = datetime.now()
	conn_mat = alg_connections(grc_pos,glom_pos,glom_mf_id,d,dlen)
	print datetime.now() - startTime
	# Glomerular degrees
	ddist = conn_mat.sum(axis=1)
	# Dendritic lengths
	dlens = np.zeros((d*N_grc),float); ix=0
	for gl in range(N_glom):
		for grc in range(N_grc):
			if conn_mat[gl,grc] == 1:
				dlens[ix] = np.sqrt(((grc_pos[grc,:]-glom_pos[gl,:])**2).sum())
				ix = ix+1
	# Save to file
	file = open('GCLconnectivity_'+str(d)+'.pkl','w')
	p = {'conn_mat':conn_mat,'ddist':ddist,'dlens':dlens,'glom_pos':glom_pos,'grc_pos':grc_pos,'glom_mf_id':glom_mf_id}
	pkl.dump(p,file); file.close()



