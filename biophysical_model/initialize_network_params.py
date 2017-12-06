# Generates file params_file.pkl, which enumerates parameter sets to be run in parallel
# Run as: python initialize_network_params.py

import numpy as np
import pickle as pkl

N_syn_range = range(1,21)
f_mf_range = np.linspace(.05,.95,19)
run_num_range = range(0,640) # Number of patterns

# Total runs over all parameters
total_params = len(N_syn_range) * len(f_mf_range) * len(run_num_range)

ix = 0

N_syn = np.zeros((total_params),int)
f_mf = np.zeros((total_params))
run_num = np.zeros((total_params),int)

for k1 in N_syn_range:
	for k2 in f_mf_range:
		for k3 in run_num_range:
			N_syn[ix] = k1
			f_mf[ix] = k2
			run_num[ix] = k3
			ix = ix+1

p = {'N_syn':N_syn,'f_mf':f_mf,'run_num':run_num}
file = open('params_file.pkl','w')
pkl.dump(p,file); file.close()
