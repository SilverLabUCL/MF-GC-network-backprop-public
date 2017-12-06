## Backpropagation learning 
# to be run on a server or cluster
# Run as: python test_mf_grc_backprop.py r_ix NADT 
# Where r_ix is index of vector of radii, 1 corresponding to 0um and 7 to 30um
# and NADT is NADT parameter

import numpy as np
import pickle as pkl
import scipy.io as io
from datetime import datetime
import sys

def get_samples(r,f_mf_ix,N_syn,num_patterns,NADT):
	file = open('../network_structures/GCLconnectivity_'+str(N_syn)+'.pkl')
	p = pkl.load(file); conn_mat = p['conn_mat']; glom_pos = p['glom_pos']
	N_mf, N_grc = conn_mat.shape
	theta = 3. + NADT*f_mf[f_mf_ix]
	if r == 0:
		x_mf = np.zeros((N_mf,num_patterns))
		for i in range(num_patterns):
			mf_on = np.random.choice(N_mf,int(round(f_mf[f_mf_ix]*N_mf)),replace=False)
			x_mf[mf_on,i] = 1.
	else:
		p=io.loadmat('../input_statistics/mf_patterns_r'+str(r)+'.mat')
		g = p['gs'][f_mf_ix]; R = p['Rs'][:,:,f_mf_ix]
		t = np.dot(R.transpose(), np.random.randn(N_mf,num_patterns))
		x_mf = 1.*(t>-g*np.ones((N_mf,num_patterns)))
	#
	inp = 4./N_syn*np.dot(conn_mat.transpose(),x_mf)
	x_grc = np.maximum(inp-theta,0)
	#
	return x_mf, x_grc

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
			err, err_d, W_out = backprop_step_nohid(W_out,gamma,training_set[:,k],target[:,k])
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

# Input parameters
r_ix = int(sys.argv[1])-1
r = ((r_ix)*5)
NADT = float(sys.argv[2])

# Network parameters, vary for different network instances
N_grc = 487; N_mf = 187
N_syn_range = range(1,21)
f_mf = np.linspace(.05,.95,19)

# Backprop parameters
gamma = 0.01
N_epochs = 5000 
C = 10
N_out = C
num_patterns = 64*C

# File name to save
if NADT == 0:
	filename = 'results_bp/grc_toy_r'+str(r)
else:
	filename = 'results_bp_NADT/grc_toy_r'+str(r)


err_mf = np.zeros((len(N_syn_range),len(f_mf),N_epochs),float)
err_rms_mf = np.zeros((len(N_syn_range),len(f_mf),N_epochs),float)

err_grc = np.zeros((len(N_syn_range),len(f_mf),N_epochs),float)
err_rms_grc = np.zeros((len(N_syn_range),len(f_mf),N_epochs),float)

for k1 in range(len(N_syn_range)):
	N_syn = N_syn_range[k1]
	print N_syn
	for k2 in range(len(f_mf)):
		# Get activity samples x_mf and x_grc
		samples_mf, samples_grc = get_samples(r,k2,N_syn,num_patterns,NADT)
		# Get random pattern classifications
		target = np.zeros((C,num_patterns))
		for k in range(num_patterns):
			target[np.random.choice(C),k] = 1
		# Single layer backpropagation
		err_rms_mf[k1,k2,:], err_mf[k1,k2,:], W_mf = backprop_nohid(samples_mf,target,N_epochs,gamma)
		err_rms_grc[k1,k2,:], err_grc[k1,k2,:], W_grc = backprop_nohid(samples_grc,target,N_epochs,gamma)
		#

# Save results
p = {'err_rms_mf':err_rms_mf, 'err_rms_grc':err_rms_grc, 'err_mf':err_mf, 'err_grc':err_grc}	
io.savemat(filename,p)




