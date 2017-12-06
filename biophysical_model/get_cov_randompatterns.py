# Get population correlation and total variance 
# Run as: python get_var_cov_biophys.py basedir
# basedir is eg data_r20
# Saves results as .mat in basedir

import numpy as np
import pickle as pkl
import scipy.io as io
from datetime import datetime
import sys

basedir = sys.argv[1]

def get_var_cov(x):
	N = x.shape[0]
	L,V = np.linalg.eig(np.cov(x)); L = np.real(np.sqrt(L+0J))
	var_x = np.sum(L**2)
	cov_x = (np.max(L)/np.sum(L) - 1./N)/(1.-1./N)
	return var_x, cov_x

filename = 'grc_cov_biophys_'+basedir.split('_')[1][:-1]

samples_mf = np.loadtxt(basedir+'MF_samples_1_0.05.txt')
samples_grc = np.loadtxt(basedir+'GrC_samples_1_0.05.txt')
N_grc = samples_grc.shape[0]
N_mf = samples_mf.shape[0]

N_syn = range(1,21)
f_mf = np.linspace(.05,.95,19)

var_mf = np.zeros((len(N_syn),len(f_mf)),float)
cov_mf = np.zeros((len(N_syn),len(f_mf)),float)

var_grc = np.zeros((len(N_syn),len(f_mf)),float)
cov_grc = np.zeros((len(N_syn),len(f_mf)),float)

for k1 in range(len(N_syn)):
	print N_syn[k1]
	for k2 in range(len(f_mf)):
		#
		samples_mf = np.loadtxt(basedir+'MF_samples_'+str(N_syn[k1])+'_'+'{:.2f}'.format(f_mf[k2])+'.txt')
		samples_grc = np.loadtxt(basedir+'GrC_samples_'+str(N_syn[k1])+'_'+'{:.2f}'.format(f_mf[k2])+'.txt')
		var_mf[k1,k2], cov_mf[k1,k2] = get_var_cov(samples_mf)
		var_grc[k1,k2], cov_grc[k1,k2] = get_var_cov(samples_grc)
		#

p = {'cov_mf':cov_mf, 'var_mf':var_mf, 'cov_grc':cov_grc, 'var_grc':var_grc}
io.savemat(basedir+filename,p)

