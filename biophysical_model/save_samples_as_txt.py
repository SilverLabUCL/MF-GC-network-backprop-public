# After all runs are finished, converts .dat files
# into .txt files for further analysis
# run as python save_samples_as_txt.py basedir
# where basedir is desired directory eg data_r20

import numpy as np
import pickle as pkl
import sys
import os

def get_spike_counts(data,N):
	x = np.zeros((N))
	if len(data) > 0:
		for i in range(N):
			if len(data.shape) == 1:
                		data = data.reshape(1,2)
			ixs = np.where(data[:,0]==i)[0]
			# Number of spikes after 150 ms 'burn in' period
			x[i] = len(np.where(data[ixs,:][:,1]>=0.150)[0])
	return x

basedir = sys.argv[1]

file = open('params_file.pkl','rb')
p = pkl.load(file); file.close()

# Network parameters
N_mf = 187
N_grc = 487
N_patt = 640

N_syn = np.unique(p['N_syn'])
p_mf_ON = np.unique(p['p_mf_ON'])

for ii in range(len(N_syn)):
	for jj in range(len(p_mf_ON)):
		print ii,jj
		samples_mf = np.zeros((N_mf,N_patt))
		samples_grc = np.zeros((N_grc,N_patt))
		for kk in range(N_patt):
			# Get MF spike counts
			data = np.loadtxt(basedir+'MF_spikes_'+str(N_syn[ii])+'_'+'{:.2f}'.format(p_mf_ON[jj])+'_'+str(kk)+'.dat')
			x = get_spike_counts(data,N_mf)
			samples_mf[:,kk] = x
			# Get GC spike counts
			data = np.loadtxt(basedir+'GrC_spikes_'+str(N_syn[ii])+'_'+'{:.2f}'.format(p_mf_ON[jj])+'_'+str(kk)+'.dat')
			x = get_spike_counts(data,N_grc)
			samples_grc[:,kk] = x
		# Save as .txt files
		np.savetxt(basedir+'MF_samples_'+str(N_syn[ii])+'_'+'{:.2f}'.format(p_mf_ON[jj])+'.txt',samples_mf)
		np.savetxt(basedir+'GrC_samples_'+str(N_syn[ii])+'_'+'{:.2f}'.format(p_mf_ON[jj])+'.txt',samples_grc)
		# Remove .dat files
		os.system('rm MF_spikes_'+str(N_syn[ii])+'_'+'{:.2f}'.format(p_mf_ON[jj])+'_*.dat')
		os.system('rm GrC_spikes_'+str(N_syn[ii])+'_'+'{:.2f}'.format(p_mf_ON[jj])+'_*.dat')

