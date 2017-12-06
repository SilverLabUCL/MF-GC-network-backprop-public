# Get population sparseness 
# Run as: python get_spar_biophys.py basedir
# basedir is eg data_r20
# Saves results as .mat in basedir

import numpy as np
import pickle as pkl
import scipy.io as io
from datetime import datetime
import sys

basedir = sys.argv[1]

def get_spar(x):
	N = x.shape[0]; T =x.shape[1]
	sptemp = np.zeros((T))
	sptemp2 = np.zeros((T))
	for t in range(T):
		sptemp[t] = (N-np.sum(x[:,t])**2./np.sum(x[:,t]**2.))/(N-1.)
		sptemp2[t] = len(np.where(x[:,t]>0)[0])*1./N
	spar = np.nanmean(sptemp)
	active = np.nanmean(sptemp2)
	return spar,active

filename = 'grc_spar_biophys_'+basedir.split('_')[1][:-1]

samples_mf = np.loadtxt(basedir+'MF_samples_1_0.05.txt')
samples_grc = np.loadtxt(basedir+'GrC_samples_1_0.05.txt')
N_grc = samples_grc.shape[0]
N_mf = samples_mf.shape[0]

N_syn = range(1,21)
f_mf = np.linspace(.05,.95,19)

spar_mf = np.zeros((len(N_syn),len(f_mf)),float)
spar_grc = np.zeros((len(N_syn),len(f_mf)),float)

active_mf = np.zeros((len(N_syn),len(f_mf)),float)
active_grc = np.zeros((len(N_syn),len(f_mf)),float)

for k1 in range(len(N_syn)):
	print N_syn[k1]
	for k2 in range(len(f_mf)):
		#
		samples_mf = np.loadtxt(basedir+'MF_samples_'+str(N_syn[k1])+'_'+'{:.2f}'.format(f_mf[k2])+'.txt')
		samples_grc = np.loadtxt(basedir+'GrC_samples_'+str(N_syn[k1])+'_'+'{:.2f}'.format(f_mf[k2])+'.txt')
		spar_mf[k1,k2], active_mf[k1,k2] = get_spar(samples_mf)
		spar_grc[k1,k2], active_grc[k1,k2] = get_spar(samples_grc)
		#

p = {'spar_mf':spar_mf, 'spar_grc':spar_grc, 'active_mf':active_mf, 'active_grc':active_grc}
io.savemat(basedir+filename,p)



