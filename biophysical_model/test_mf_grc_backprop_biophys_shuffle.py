## Backpropagation learning on shuffled data
# to be run on a server or cluster
# Run as: python test_mf_grc_backprop_biophys_shuffle.py basedir
# Where basedir is the base directory containing spike pattern data


import numpy as np
import pickle as pkl
import scipy.io as io
from datetime import datetime
import sys, os


def get_var_cov(x):
	N = x.shape[0]
	L,V = np.linalg.eig(np.cov(x)); L = np.real(np.sqrt(L+0J))
	var_x = np.sum(L**2)
	cov_x = (np.max(L)/np.sum(L) - 1./N)/(1.-1./N)
	return var_x, cov_x

# Partial shuffling algorithm
# See MATLAB code for more detailed comments
def part_shuffle(x,cov_x,cov_desired):
	N,T = x.shape
	#
	switch_cells = np.zeros((N))
	for i in range(N):
		if len(np.unique(x[i,:]))>1:
			switch_cells[i] = 1
	whichswitch = np.where(switch_cells)[0]
	#
	cov_new = cov_x.copy()
	#
	if cov_desired < cov_x: # Decrease correlations
		while cov_new > cov_desired:
			num_to_switch_simul = int(np.max([1,np.min([N,np.round((cov_new-cov_desired)/cov_new*N/0.5)])]))
			for i in range(num_to_switch_simul):
				randcell = np.random.choice(whichswitch)
				T1 = np.random.choice(T)
				T2 = np.random.choice(np.where(x[randcell,:] != x[randcell,T1])[0])
				x_T1 = x[randcell,T1]
				x_T2 = x[randcell,T2]
				x[randcell,T1] = x_T2
				x[randcell,T2] = x_T1
			var_new, cov_new = get_var_cov(x)
	elif cov_desired > cov_x: # Increase correlations
		while cov_new < cov_desired:
			Ts = np.random.choice(T,2) # 1st value is low, 2nd value is high
			for i in range(len(whichswitch)):
				x_T1 = x[whichswitch[i],Ts[0]]
				x_T2 = x[whichswitch[i],Ts[1]]
				x_mean = x[whichswitch[i],:].mean()
				if x_T1 > x_mean and x_T2 < x_mean:
					x[whichswitch[i],Ts[0]] = x_T2
					x[whichswitch[i],Ts[1]] = x_T1
			var_new, cov_new = get_var_cov(x)
	#
	var_new, cov_new = get_var_cov(x)
	return x, cov_new, var_new

def backprop_step_nohid(W_out,gamma,training_pattern,target_pattern):
	# Dynamics of units
	s = lambda x: 1./(1.+np.exp(-x)); ds = lambda s: s*(1.-s) # sigmoidal
	#####################################
	# First step: feedforward propagation
	o_in = training_pattern
	o_out = s(np.dot(np.append(o_in,1),W_out))
	# Second step: output layer backpropagation
	D = np.diag(ds(o_out))
	err = o_out - target_pattern
	err_d = np.prod((target_pattern==target_pattern.max()) == (o_out==o_out.max()))
	delta_out = np.dot(D,err)
	dW_out = - gamma * np.outer(np.append(o_in,1),delta_out)
	# Third step: update weights
	W_out = W_out + dW_out;
	#
	return err, err_d, W_out

def backprop_nohid(training_set,target,n_epochs,gamma):
	#
	n_in = training_set.shape[0]; n_out = target.shape[0]
	W_out = np.random.uniform(-1.,1.,size=(n_in+1,n_out))*1./(n_in+1)
	# Shuffle order of training set
	temp = range(training_set.shape[1])
	np.random.shuffle(temp)
	training_set = training_set[:,temp]
	target = target[:,temp]
	#
	errors_rms = np.zeros((n_epochs),float)
	errors_discrim = np.zeros((n_epochs),float)
	for ep in range(n_epochs):
		errors_temp = np.zeros((target.shape[1]),float)
		errors_d_temp = np.zeros((target.shape[1]),float)
		for k in range(target.shape[1]):
			# Backpropagation step
			err, err_d, W_out = backprop_step_nohid(W_out,gamma,training_set[:,k],target[:,k],neuron_type)
			# Record errors
			errors_temp[k] = np.sqrt(np.mean(err**2)) # RMS error
			errors_d_temp[k] = err_d # Discrimination error
		# Record average error for the epoch
		errors_rms[ep] = errors_temp.mean()
		errors_discrim[ep] = errors_d_temp.mean()
		# Reshuffle order of training data
		temp = range(training_set.shape[1])
		np.random.shuffle(temp)
		training_set = training_set[:,temp]
		target = target[:,temp]
	#
	return errors_rms, errors_discrim, W_out

def get_var_cov(x):
	N = x.shape[0]
	L,V = np.linalg.eig(np.cov(x)); L = np.real(np.sqrt(L+0J))
	var_x = np.sum(L**2)
	cov_x = (np.max(L)/np.sum(L) - 1./N)/(1.-1./N)
	return var_x, cov_x

basedir = sys.argv[1]

# Network parameters
samples_mf = np.loadtxt(basedir+'MF_samples_1_0.05.txt')
samples_grc = np.loadtxt(basedir+'GrC_samples_1_0.05.txt')
N_grc = samples_grc.shape[0]
N_mf = samples_mf.shape[0]
N_syn_range = [1,2,4,8,16] # Subsample Nsyn for speed
f_mf = np.linspace(.05,.95,19)

# Backprop parameters
N_epochs = 5000
gamma = 0.01
C = 10
N_out = C
num_patterns = 64*C

# File name to save
end_filename = basedir.split('data_')[1][:-1]+'_shuff_'+str(k1)+'_'+str(N_syn_range[k2])
filename = 'grc_bp_biophys_'+end_filename

for k1 in range(len(N_syn_range)):
	N_syn = N_syn_range[k1]
	print N_syn
	for k2 in range(len(f_mf)):
		# Load spike count samples
		samples_mf = np.loadtxt(basedir+'MF_samples_'+str(N_syn)+'_'+'{:.2f}'.format(f_mf[k2])+'.txt')
		samples_grc = np.loadtxt(basedir+'GrC_samples_'+str(N_syn)+'_'+'{:.2f}'.format(f_mf[k2])+'.txt')
		# Get total variance and population correlation
		var_mf, cov_mf = get_var_cov(samples_mf)
		var_grc, cov_grc = get_var_cov(samples_grc)
		# Shuffle
		samples_sh, cov_sh, var_sh =  part_shuffle(samples_grc,cov_grc,cov_mf)
		# Get pattern classifications
		target = np.zeros((C,num_patterns))
		for k in range(num_patterns):
			target[np.random.choice(C),k] = 1
		# Single layer backpropagation
		err_rms_sh, err_sh, W_sh = backprop_nohid(samples_sh,target,N_epochs,gamma,'sigmoid')
		#

# Save results
p = {'err_rms_sh':err_rms_sh, 'err_sh':err_sh, 'cov_sh':cov_sh, 'var_sh':var_sh}
io.savemat(basedir+filename,p)



