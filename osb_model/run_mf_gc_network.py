
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

spike_generator_file_name = "spikeGenerators.xml"

def generate_grc_layer_network(p_mf_ON,
                               duration, 
                               dt, 
                               minimumISI, # ms
                               ONRate, # Hz 
                               OFFRate, # Hz
                               run=False):

    # Load connectivity matrix
    
    file = open('GCLconnectivity.pkl')
    p = pkl.load(file); conn_mat = p['conn_mat']
    N_mf, N_grc = conn_mat.shape
    assert (np.all(conn_mat.sum(axis=0)==4)), 'Connectivity matrix is incorrect.'
    
    # Load GrC and MF rosette positions
    
    grc_pos = p['grc_pos']
    glom_pos = p['glom_pos']
        
    # Choose which mossy fibers are on, which are off
    
    N_mf_ON = int(N_mf*p_mf_ON)
    mf_indices_ON = random.sample(range(N_mf),N_mf_ON); mf_indices_ON.sort()


    N_mf_OFF = N_mf - N_mf_ON
    mf_indices_OFF = [x for x in range(N_mf) if x not in mf_indices_ON]; mf_indices_OFF.sort()
    
    # load NeuroML components, LEMS components and LEMS componentTypes from external files
    
    spike_generator_doc = pynml.read_lems_file(spike_generator_file_name)

    iaF_GrC = nml.IafRefCell(id="iaF_GrC",
                   refract="2ms",
                   C="3.22pF",
                   thresh = "-40mV",
                   reset="-63mV",
                   leak_conductance="1.498nS",
                   leak_reversal="-79.67mV")

    ampa_syn_filename="RothmanMFToGrCAMPA.xml"
    nmda_syn_filename="RothmanMFToGrCNMDA.xml"

    rothmanMFToGrCAMPA_doc = pynml.read_lems_file(ampa_syn_filename)
    rothmanMFToGrCNMDA_doc = pynml.read_lems_file(nmda_syn_filename)

    # define some components from the componentTypes we just loaded
    spike_generator_ref_poisson_type = spike_generator_doc.component_types['spikeGeneratorRefPoisson']
    lems_instances_doc = lems.Model()

    spike_generator_on = lems.Component("mossySpikerON", spike_generator_ref_poisson_type.name)
    spike_generator_on.set_parameter("minimumISI", "%s ms"%minimumISI)
    spike_generator_on.set_parameter("averageRate", "%s Hz"%ONRate)
    lems_instances_doc.add(spike_generator_on)

    spike_generator_off = lems.Component("mossySpikerOFF", spike_generator_ref_poisson_type.name)
    spike_generator_off.set_parameter("minimumISI", "%s ms"%minimumISI)
    spike_generator_off.set_parameter("averageRate", "%s Hz"%OFFRate)
    lems_instances_doc.add(spike_generator_off)

    rothmanMFToGrCAMPA = rothmanMFToGrCAMPA_doc.components['RothmanMFToGrCAMPA'].id
    rothmanMFToGrCNMDA = rothmanMFToGrCNMDA_doc.components['RothmanMFToGrCNMDA'].id

    # create populations
    GrCPop = nml.Population(id=iaF_GrC.id+"Pop",component=iaF_GrC.id,type="populationList",size=N_grc)
    mossySpikersPopON = nml.Population(id=spike_generator_on.id+"Pop",component=spike_generator_on.id,type="populationList",size=N_mf_ON)
    mossySpikersPopOFF = nml.Population(id=spike_generator_off.id+"Pop",component=spike_generator_off.id,size=N_mf_OFF)

    # create network and add populations
    net = nml.Network(id="network")
    net_doc = nml.NeuroMLDocument(id=net.id)
    net_doc.networks.append(net)
    net_doc.iaf_ref_cells.append(iaF_GrC)
    net.populations.append(GrCPop)
    net.populations.append(mossySpikersPopON)
    net.populations.append(mossySpikersPopOFF)

    #net_doc.includes.append(nml.IncludeType(href=iaf_nml2_file_name))

    # Add locations for GCs

    for grc in range(N_grc):
        inst = nml.Instance(id=grc)
        GrCPop.instances.append(inst)
        inst.location = nml.Location(x=grc_pos[grc,0], y=grc_pos[grc,1], z=grc_pos[grc,2])

    # ON MFs: locations and connectivity

    ONprojectionAMPA = nml.Projection(id="ONProjAMPA", presynaptic_population=mossySpikersPopON.id, postsynaptic_population=GrCPop.id, synapse=rothmanMFToGrCAMPA)
    ONprojectionNMDA = nml.Projection(id="ONProjNMDA", presynaptic_population=mossySpikersPopON.id, postsynaptic_population=GrCPop.id, synapse=rothmanMFToGrCNMDA)
    net.projections.append(ONprojectionAMPA)
    net.projections.append(ONprojectionNMDA)

    ix = 0
    for mf_ix_ON in range(N_mf_ON):
        mf_ix = mf_indices_ON[mf_ix_ON]
        inst = nml.Instance(id=mf_ix_ON)
        mossySpikersPopON.instances.append(inst)
        inst.location = nml.Location(x=glom_pos[mf_ix,0], y=glom_pos[mf_ix,1], z=glom_pos[mf_ix,2])
        # find which granule cells are neighbors
        innervated_grcs = np.where(conn_mat[mf_ix,:]==1)[0]
        for grc_ix in innervated_grcs:
            for synapse in [rothmanMFToGrCAMPA, rothmanMFToGrCNMDA]:
                connection = nml.Connection(id=ix, 
                        pre_cell_id='../{}/{}/{}'.format(mossySpikersPopON.id,mf_ix_ON,spike_generator_on.id), 
                        post_cell_id='../{}/{}/{}'.format(GrCPop.id,grc_ix,iaF_GrC.id))
                ONprojectionAMPA.connections.append(connection)
                ONprojectionNMDA.connections.append(connection)
                ix=ix+1
    
    # OFF MFs: locations and connectivity
    
    OFFprojectionAMPA = nml.Projection(id="OFFProjAMPA", presynaptic_population=mossySpikersPopOFF.id, postsynaptic_population=GrCPop.id, synapse=rothmanMFToGrCAMPA)
    OFFprojectionNMDA = nml.Projection(id="OFFProjNMDA", presynaptic_population=mossySpikersPopOFF.id, postsynaptic_population=GrCPop.id, synapse=rothmanMFToGrCNMDA)
    net.projections.append(OFFprojectionAMPA)
    net.projections.append(OFFprojectionNMDA)

    ix = 0
    for mf_ix_OFF in range(N_mf_OFF):
        mf_ix = mf_indices_OFF[mf_ix_OFF]
        inst = nml.Instance(id=mf_ix_OFF)
        mossySpikersPopOFF.instances.append(inst)
        inst.location = nml.Location(x=glom_pos[mf_ix,0], y=glom_pos[mf_ix,1], z=glom_pos[mf_ix,2])
        # find which granule cells are neighbors
        innervated_grcs = np.where(conn_mat[mf_ix,:]==1)[0]
        for grc_ix in innervated_grcs:
            for synapse in [rothmanMFToGrCAMPA, rothmanMFToGrCNMDA]:
                connection = nml.Connection(id=ix, 
                        pre_cell_id='../{}/{}/{}'.format(mossySpikersPopOFF.id,mf_ix_OFF,spike_generator_on.id), 
                        post_cell_id='../{}/{}/{}'.format(GrCPop.id,grc_ix,iaF_GrC.id))
                OFFprojectionAMPA.connections.append(connection)
                OFFprojectionNMDA.connections.append(connection)
                ix=ix+1


    # Write network to file
    net_file_name = 'OSBnet.nml'
    pynml.write_neuroml2_file(net_doc, net_file_name)

    # Write LEMS instances to file
    lems_instances_file_name = 'instances.xml'
    pynml.write_lems_file(lems_instances_doc, lems_instances_file_name,validate=False)
    
    # Create a LEMSSimulation to manage creation of LEMS file
    ls = LEMSSimulation('sim', duration, dt, lems_seed = 123)# int(np.round(1000*random.random())))
    
    # Point to network as target of simulation
    ls.assign_simulation_target(net.id)
    
    # Include generated/existing NeuroML2 files
    ls.include_lems_file(spike_generator_file_name, include_included=False)
    ls.include_lems_file(lems_instances_file_name)
    ls.include_lems_file(ampa_syn_filename, include_included=False)
    ls.include_lems_file(nmda_syn_filename, include_included=False)
    ls.include_neuroml2_file(net_file_name)
    
    # Specify Displays and Output Files
    
    basedir = ''

    eof0 = 'Volts_file'
    ls.create_event_output_file(eof0, basedir+"MF_spikes.dat")

    for i in range(mossySpikersPopON.size):
        ls.add_selection_to_event_output_file(eof0, mf_indices_ON[i], '{}/{}/{}'.format(mossySpikersPopON.id,i,spike_generator_on.id), 'spike')

    for i in range(mossySpikersPopOFF.size):
        ls.add_selection_to_event_output_file(eof0, mf_indices_OFF[i], '{}/{}/{}'.format(mossySpikersPopOFF.id,i,spike_generator_on.id), 'spike')

    eof1 = 'GrCspike_file'
    ls.create_event_output_file(eof1, basedir+"GrC_spikes.dat")
    
    for i in range(GrCPop.size):
        ls.add_selection_to_event_output_file(eof1, i, '{}/{}/{}'.format(GrCPop.id,i,iaF_GrC.id), 'spike')
        
    lems_file_name = ls.save_to_file()

    if run:
        results = pynml.run_lems_with_jneuroml(lems_file_name,max_memory="8G", nogui=True, load_saved_data=False, plot=False)
        
        return results

if __name__ == '__main__':

    startTime = datetime.now()

    generate_grc_layer_network(p_mf_ON = 1,
                               duration = 30, 
                               dt = 0.05,
                               minimumISI = 2,
                               ONRate = 50,
                               OFFRate = 0,
                               run = True)
    
    print datetime.now() - startTime


