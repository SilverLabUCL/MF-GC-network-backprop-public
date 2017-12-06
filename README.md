# MF-GC-network-backprop-public

This is the code necessary to reproduce all major figures for Cayco-Gajic, Clopath, & Silver 2017, "Sparse synaptic connectivity is required for decorrelation and pattern separation in feedforward networks." A simplified version of the biophysical model is available in the standalone folder osb_model. See also: http://www.opensourcebrain.org/projects/grclayer-caycogajic2017

### Getting started
This repo contains both Matlab and Python code. In general, the Matlab scripts are designed to be run locally and Python scripts to be run on a cluster. All simulations require pre-generated network connectivities and input statistics, which are used by both the analytical and biophysical models. These can either be generated from scratch (see below) or example pre-generated models can be downloaded from a sister GitHub repo ('Pre-simulated data').

To generate new network connectivities, use:
```
network_structures/GCL_make_connectivity.py
```

To change input statistics, use 
```
input_statistics/generate_input_patterns.m
``` 
This requires the Dichotomized Gaussian tools from the Macke Lab in order to generate spike trains with arbitrary correlations. See: https://bitbucket.org/mackelab/pop_spike/src

### Pre-simulated data
To quickly generate figures from the paper, pre-generated files / pre-simulated data can be downloaded from: https://github.com/SilverLabUCL/MF-GC-network-backprop-data.

### Simulations
For the analytical model, simulations are done within the .m plotting files, or within the .py file for backpropagation. On the other hand, the biophysical model must be simulated on a cluster prior to analysing the patterns. First the parameters must be initalized via:
```
biophysical_model/initialize_network_params.py
```
This creates the file params_file.pkl, which is used to run array jobs on an external cluster (skip this step if using pre-simulated data). After initialisation, the following script simulates the model:
```
biophysical_model/run_mf_grc_network.py
```
This saves the GC and MF spike times as .dat files for each simulation. To save memory and analyse further, after finishing the simulations run:
```
biophysical_model/save_samples_as_txt.py
```
to convert the spike times to spike count activity patterns, and remove the .dat files. 

Note that simulating the biophysical model requires NeuroML2, jNeuroML, and pyNeuroML (https://github.com/NeuroML). The folder grc_lemsDefinitions includes NeuroML2 and XML files necessary to run the biophysical model. 

### Analysis
Activity pattern properties (variance, covariance, population sparseness) can be computed using the following code:
```
analytical_model/get_var_cov.m
```
or
```
biophysical_model/get_cov_randompatterns.py
```

To run backpropagation, use either:
```
analytical_model/test_mf_grc_backprop.py
```
or
```
biophysical_model/test_mf_grc_backprop_biophys.py
```

For the analytical model, the .m files above will automatically produce relevant plots. For the biophysical model, after running the above .py files, use:
```
biophysical_model/analyze_simulated_data.m
```
