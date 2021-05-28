#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import h5py

def getNeuronMatrix(folder):
    '''Given the path to a network folder 'Snudda/networks/XXXXX' where there is a 
    network-neuron-position.hdf5 file, returns a Matrix where the content of of the
    first column is the neuron_ID and the content of the second column is the neuron 
    type.
    ie. neuron_matrix[0][0]= 0  and  neuron_matrix[0][1] = "dspn
    "means that the neuron with neuron_ID=0 is a dspn '''
    
    path = folder + '/network-neuron-positions.hdf5'
    
    with h5py.File(path, 'r') as hdf5:
        neurons = (hdf5.get('network')).get('neurons')
        neuron_ID = list(neurons.get('neuronID'))
        neuron_type = list(neurons.get('morphology'))

    neuron_type = removePath(neuron_type)
    
    neuron_matrix = []
    neuron_matrix.append(neuron_ID)
    neuron_matrix.append(neuron_type)
    
    neuron_matrix = matrixTranspose(neuron_matrix)

    
    return(neuron_matrix)

def getNeuronSubMatrixes(matrix):
    '''Given the neuron matrix, returns 1 array for each neuron type, with the neuron_IDs of neurons of this type
    Assumes there are only the 5 neuron types we see in the example (ispn, dpsn, lts chin and fs)
     '''
    
    dspn = []
    ispn = []
    lts = []
    fs = []
    chin = []
    
    for i in range (0, len(matrix)):
        if matrix[i][1] == "dspn":
            dspn.append(matrix[i][0])
        elif matrix[i][1] == "ispn":
            ispn.append(matrix[i][0])
        elif matrix[i][1] == "lts":
            lts.append(matrix[i][0])
        elif matrix[i][1] == "fs":
            fs.append(matrix[i][0])
        elif matrix[i][1] == "chin":
            chin.append(matrix[i][0])

    return(dspn, ispn, lts, fs, chin)

def getSpikes(file_path):
    
    '''
    Given the path to a spike .txt archive, copies the information onto a dictionary
    The keys are the IDs of each of the spiking neurons, and the corresponding value is an array
    with that neuron's spike times (in ns)
    
    Here I ask for the full path because there are multiple output files (for the diff simulation durations)
     on each network folder
    '''
        
    with open('Snudda/networks/tinySim/simulation/network-output-spikes-666.txt', 'r') as file:
        lines = file.readlines()

    spikes = {}
    
    for line in lines :
        
        split_line = np.array(line.split("\t"),float)
        nID = int(split_line[1])
        spike_time = split_line[0]
        if(nID not in spikes.keys()):
            spikes[nID] = []            
        spikes[nID].append(spike_time)
            
    return(spikes)

def getSubSpikes(spikes, matrix):
    
    dspn_spikes = {}
    ispn_spikes = {}
    lts_spikes = {}
    fs_spikes = {}
    chin_spikes = {}
    
    for key in spikes.keys():
        
        if matrix[key][1] == "dspn":
            if(key not in dspn_spikes.keys()):
                dspn_spikes[key] = []            
            dspn_spikes[key].extend(spikes[key])
            
        elif matrix[key][1] == "ispn":
            if(key not in ispn_spikes.keys()):
                ispn_spikes[key] = []            
            ispn_spikes[key].extend(spikes[key])
            
        elif matrix[key][1] == "lts":
            if(key not in lts_spikes.keys()):
                lts_spikes[key] = []            
            lts_spikes[key].extend(spikes[key])
            
        elif matrix[key][1] == "fs":
            if(key not in fs_spikes.keys()):
                fs_spikes[key] = []            
            fs_spikes[key].extend(spikes[key])
            
        elif matrix[key][1] == "chin":
            if(key not in chin_spikes.keys()):
                chin_spikes[key] = []            
            chin_spikes[key].extend(spikes[key])

    return(dspn_spikes)

def getVolts(file_path):
    
    '''
    Given the path to a volt.txt archive, copies the information onto a matrix and an array
    The matrix has one line for each neuron (line number = neuron_ID) and the values of the voltage of said neuron
    at each time instant on the different columns.
    The array has the time values for each voltage measure.
    Here I ask for the full path because there are multiple output files (for the diff simulation durations)
     on each network folder
    '''
        
    with open(file_path, 'r') as file:
        volt = file.readlines()

    time = np.array(volt[0].split(","),float)
    time = time[1:] #cuts the first column (just an index)

    volt = volt[1:]
    traces =[]
    for i in range(0,len(volt)):
        traces.append(np.array(volt[i].split(","),float))
        traces[i] = traces[i][1:] #cuts the first column (just an index)
        
    return(traces, time)

def removePath(ntype_array):
    '''Receives a list containing the morphology file pathway for each neuron, returns a new array
    that only states the neuron type.'''
    
    new_ntype_array =[]

    for i in range (0, len(ntype_array)) :
        new_ntype_array.append(str(ntype_array[i]))

    for i in range (0, len(new_ntype_array)):
        if "dspn" in new_ntype_array[i]:
            new_ntype_array[i] = "dspn"
        elif ('ispn' in new_ntype_array[i]) :
            new_ntype_array[i] = "ispn"
        elif 'lts' in new_ntype_array[i] :
            new_ntype_array[i] = "lts"
        elif 'fs' in new_ntype_array[i] :
            new_ntype_array[i] = "fs"
        elif 'chin' in new_ntype_array[i] :
            new_ntype_array[i] = "chin"
        
    return(new_ntype_array)

def matrixTranspose(matrix):
    if not matrix: return ([])
    return ([[row[i] for row in matrix] for i in range(len(matrix[0]))])

