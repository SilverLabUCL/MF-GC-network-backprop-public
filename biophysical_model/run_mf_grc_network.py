## Runs biophysical simulations of MF-GC network
# Designed to write files to be run in parallel 
# on a server or cluster, but can also be used to run
# individually
# python run_mf_grc_network.py runID sigma NADT
# Generates xml and nml files for model, saved in tempdata
# To simulate, switch run to True, spike times saved as .dat

# Flow for biophysical model:
# python initialize_network_params.py to generate params_file.pkl for a specific sigma, NADT
# python run_mf_grc_network.py runID sigma NADT  to run files for specific sigma, NADT -- run for ALL runID in params_file.pkl
# python save_samples_as_txt.py basedir to convert .dat files of spikeimes to .txt files of activity patterns
# Then can run test_mf_grc_backprop_biophys.py, etc.

import neuroml as nml
from pyneuroml import pynml
from pyneuroml.lems.LEMSSimulation import LEMSSimulation
import lems.api as lems

import random
import numpy as np
import os
import pickle as pkl
import scipy.io as io
import sys
from datetime import datetime

random.seed()

def generate_grc_layer_network(runID,
                               correlationRadius,
                               NADT,
                               duration, 
                               dt, 
                               minimumISI, # ms
                               ONRate, # Hz
                               OFFRate, # Hz
                               run=False):
    ########################################
    # Load parameters for this run
    file = open('../params_file.pkl','r')
    p = pkl.load(file)
    N_syn = p['N_syn'][int(runID)-1]
    f_mf = p['f_mf'][int(runID)-1]
    run_num = p['run_num'][int(runID)-1]
    file.close()
    #################################################################################
    # Get connectivity matrix between cells
    file = open('../../network_structures/GCLconnectivity_'+str(N_syn)+'.pkl')
    p = pkl.load(file); conn_mat = p['conn_mat']
    N_mf, N_grc = conn_mat.shape
    assert (np.all(conn_mat.sum(axis=0)==N_syn)), 'Connectivity matrix is incorrect.'
    # Get MF activity pattern
    if correlationRadius == 0: # Activate MFs randomly
        N_mf_ON = int(N_mf*f_mf)
        mf_indices_ON = random.sample(range(N_mf),N_mf_ON); mf_indices_ON.sort()
    elif correlationRadius>0: # Spatially correlated MFs
        f_mf_range = np.linspace(.05,.95,19); f_mf_ix = np.where(f_mf_range == f_mf)[0][0]
        p = io.loadmat('../../input_statistics/mf_patterns_r'+str(correlationRadius)+'.mat')
        R = p['Rs'][:,:,f_mf_ix]; g = p['gs'][f_mf_ix]
        t = np.dot(R.transpose(),np.random.randn(N_mf))
        S = (t>-g*np.ones(N_mf))
        mf_indices_ON = np.where(S)[0]
        N_mf_ON = len(mf_indices_ON)
    #
    N_mf_OFF = N_mf - N_mf_ON
    mf_indices_OFF = [x for x in range(N_mf) if x not in mf_indices_ON]; mf_indices_OFF.sort()
    #################################################################################
    # load NeuroML components, LEMS components and LEMS componentTypes from external files
    # Spike generator (for Poisson MF spiking)
    spike_generator_file_name = "../../grc_lemsDefinitions/spikeGenerators.xml"
    spike_generator_doc = pynml.read_lems_file(spike_generator_file_name)
    # Integrate-and-fire GC model
    # if NADT = 1, loads model GC
    iaf_nml2_file_name = "../../grc_lemsDefinitions/IaF_GrC.nml" if NADT == 0 else "../../grc_lemsDefinitions/IaF_GrC_"+'{:.2f}'.format(f_mf)+".nml"
    iaF_GrC_doc = pynml.read_neuroml2_file(iaf_nml2_file_name)
    iaF_GrC = iaF_GrC_doc.iaf_ref_cells[0]
    # AMPAR and NMDAR mediated synapses
    ampa_syn_filename="../../grc_lemsDefinitions/RothmanMFToGrCAMPA_"+str(N_syn)+".xml"
    nmda_syn_filename="../../grc_lemsDefinitions/RothmanMFToGrCNMDA_"+str(N_syn)+".xml"
    rothmanMFToGrCAMPA_doc = pynml.read_lems_file(ampa_syn_filename)
    rothmanMFToGrCNMDA_doc = pynml.read_lems_file(nmda_syn_filename)
    #
    # Define components from the componentTypes we just loaded
    # Refractory poisson input -- representing active MF
    spike_generator_ref_poisson_type = spike_generator_doc.component_types['spikeGeneratorRefPoisson']
    lems_instances_doc = lems.Model()
    spike_generator_on = lems.Component("mossySpikerON", spike_generator_ref_poisson_type.name)
    spike_generator_on.set_parameter("minimumISI", "%s ms"%minimumISI)
    spike_generator_on.set_parameter("averageRate", "%s Hz"%ONRate)
    lems_instances_doc.add(spike_generator_on)
    # Refractory poisson input -- representing silent MF
    spike_generator_off = lems.Component("mossySpikerOFF", spike_generator_ref_poisson_type.name)
    spike_generator_off.set_parameter("minimumISI", "%s ms"%minimumISI)
    spike_generator_off.set_parameter("averageRate", "%s Hz"%OFFRate)
    lems_instances_doc.add(spike_generator_off)
    # Synapses
    rothmanMFToGrCAMPA = rothmanMFToGrCAMPA_doc.components['RothmanMFToGrCAMPA'].id
    rothmanMFToGrCNMDA = rothmanMFToGrCNMDA_doc.components['RothmanMFToGrCNMDA'].id
    #
    # Create ON MF, OFF MF, and GC populations
    GrCPop = nml.Population(id="GrCPop",component=iaF_GrC.id,size=N_grc)
    mossySpikersPopON = nml.Population(id=spike_generator_on.id+"Pop",component=spike_generator_on.id,size=N_mf_ON)
    mossySpikersPopOFF = nml.Population(id=spike_generator_off.id+"Pop",component=spike_generator_off.id,size=N_mf_OFF)
    #
    # Create network and add populations
    net = nml.Network(id="network")
    net_doc = nml.NeuroMLDocument(id=net.id)
    net_doc.networks.append(net)
    net.populations.append(GrCPop)
    net.populations.append(mossySpikersPopON)
    net.populations.append(mossySpikersPopOFF)
    #
    # MF-GC connectivity
    # First connect ON MFs to GCs
    for mf_ix_ON in range(N_mf_ON):
        mf_ix = mf_indices_ON[mf_ix_ON]
        # Find which GCs are neighbors
        innervated_grcs = np.where(conn_mat[mf_ix,:]==1)[0]
        for grc_ix in innervated_grcs:
            # Add AMPAR and NMDAR mediated synapses
            for synapse in [rothmanMFToGrCAMPA, rothmanMFToGrCNMDA]: 
                connection = nml.SynapticConnection(from_='{}[{}]'.format(mossySpikersPopON.id,mf_ix_ON),
                                                    synapse=synapse,
                                                    to='GrCPop[{}]'.format(grc_ix))
                net.synaptic_connections.append(connection)
    #
    # Now connect OFF MFs to GCs
    for mf_ix_OFF in range(N_mf_OFF):
        mf_ix = mf_indices_OFF[mf_ix_OFF]
        # Find which GCs are neighbors
        innervated_grcs = np.where(conn_mat[mf_ix,:]==1)[0]
        for grc_ix in innervated_grcs:
            # Add AMPAR and NMDAR mediated synapses
            for synapse in [rothmanMFToGrCAMPA, rothmanMFToGrCNMDA]:
                connection = nml.SynapticConnection(from_='{}[{}]'.format(mossySpikersPopOFF.id,mf_ix_OFF),
                                                    synapse=synapse,
                                                    to='GrCPop[{}]'.format(grc_ix))
                net.synaptic_connections.append(connection)
    #
    # Write network to file
    net_file_name = 'generated_network_'+runID+'.net.nml'
    pynml.write_neuroml2_file(net_doc, net_file_name)
    # Write LEMS instances to file
    lems_instances_file_name = 'instances_'+runID+'.xml'
    pynml.write_lems_file(lems_instances_doc, lems_instances_file_name,validate=False)
    # Create a LEMSSimulation to manage creation of LEMS file
    ls = LEMSSimulation('sim_'+runID, duration, dt, lems_seed = int(np.round(1000*random.random())))
    # Point to network as target of simulation
    ls.assign_simulation_target(net.id)
    # Include generated/existing NeuroML2 files
    ls.include_neuroml2_file(iaf_nml2_file_name)
    ls.include_lems_file(spike_generator_file_name, include_included=False)
    ls.include_lems_file(lems_instances_file_name)
    ls.include_lems_file(ampa_syn_filename, include_included=False)
    ls.include_lems_file(nmda_syn_filename, include_included=False)
    ls.include_neuroml2_file(net_file_name)
    # Specify Displays and Output Files
    # Details for saving output files
    basedir = '../data_r'+str(correlationRadius)+'/' if NADT==0 else '../data_r'+str(correlationRadius)+'_NADT/'
    end_filename = str(N_syn)+'_{:.2f}'.format(f_mf)+'_'+str(run_num) # Add parameter values to spike time filename
    # Save MF spike times under basedir + MF_spikes_ + end_filename
    eof0 = 'MFspikes_file'
    ls.create_event_output_file(eof0, basedir+"MF_spikes_"+end_filename+".dat")
    # ON MFs
    for i in range(mossySpikersPopON.size):
        ls.add_selection_to_event_output_file(eof0, mf_indices_ON[i], "%s[%i]"%(mossySpikersPopON.id, i), 'spike')
    # OFF MFs
    for i in range(mossySpikersPopOFF.size):
        ls.add_selection_to_event_output_file(eof0, mf_indices_OFF[i], "%s[%i]"%(mossySpikersPopOFF.id, i), 'spike')
    # Save GC spike times under basedir + GrC_spikes_ + end_filename
    eof1 = 'GrCspikes_file'
    ls.create_event_output_file(eof1, basedir+"GrC_spikes_"+end_filename+".dat")
    #    
    for i in range(GrCPop.size):
        ls.add_selection_to_event_output_file(eof1, i, "%s[%i]"%(GrCPop.id, i), 'spike')
    #
    lems_file_name = ls.save_to_file()
    #
    if run:
        results = pynml.run_lems_with_jneuroml(lems_file_name,max_memory="8G", nogui=True, load_saved_data=False, plot=False)
        
        return results

if __name__ == '__main__':
    #
    startTime = datetime.now()
    #
    runID = sys.argv[1] # Determines N_syn, f_mf 
    sigma = int(sys.argv[2]) # size of cluster
    NADT = float(sys.argv[3]) # =0 if no NADT, =1 if NADT
    #
    # Move working dir to tempdata to hide xml and nml files
    os.chdir('tempdata')
    #
    generate_grc_layer_network(runID = runID,
                               correlationRadius = sigma,
                               NADT = NADT,
                               duration = 180, 
                               dt = 0.05,
                               minimumISI = 2,
                               ONRate = 50,
                               OFFRate = 0,
                               run = False) # Change to True to run
    #
    # Move back to original directory
    os.chdir('..')
    print datetime.now() - startTime

