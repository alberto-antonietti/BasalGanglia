import numpy as np
import h5py
import pandas as pd


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
    Assumes there are only the 5 neuron types we see in the example (dspn, ipsn, lts chin and fs)
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


def  getOrderedIDs(dspn, ispn, lts, fs, chin):
    
    ordered_raster = []
    
    for fs_id in fs:
        ordered_raster.append(fs_id)
    
    for dspn_id in dspn:
        ordered_raster.append(dspn_id)
        
    for lts_id in lts:
        ordered_raster.append(lts_id)
    
    for ispn_id in ispn:
        ordered_raster.append(ispn_id)
        
    for chin_id in chin:
        ordered_raster.append(chin_id)        
    
    return(ordered_raster)

def getNeuronPositions(net):
    ''' Receives the network size: net_100, net_1000 or net_10000'''
    
    neuron_positions = []
    
    path = '/home/ubuntu/BasalGanglia/NEURON-data/' + net + '/network-neuron-positions.hdf5'

    
    with h5py.File(path, 'r') as hdf5:

        position = hdf5.get('network').get('neurons').get('position')

        for i in range(0,len(position)):
            single_neuron = []
            for j in range(0, len(position[0])):
                single_neuron.append(position[i][j])                    
            neuron_positions.append(single_neuron)
    
    return(neuron_positions)

def getSynapses(net):
    
    path = '/home/ubuntu/BasalGanglia/NEURON-data/' + net + '/network-pruned-synapses.hdf5'
    
    with h5py.File(path, 'r') as file:
        synapses = file.get('network').get('synapses')

        synaptic_matrix = []
        for i in range(0,len(synapses)):
            synaptic_matrix.append([synapses[i][0], synapses[i][1]]) #Pre_ID | Pos_ID

    return(synaptic_matrix)

def old_getWeightedSynapses(pre_array, post_array):
    
    syn_matrix = []
    
    for i in range(0, len(pre_array)):
        syn_matrix.append([pre_array[i], post_array[i]])
        
    syn_df = pd.DataFrame(data= syn_matrix , columns=["source", "target"])

    new = syn_df.groupby(['source', 'target']).size().reset_index()
    wgt_syn_matrix = new.to_numpy()
    print(wgt_syn_matrix)
    sources = wgt_syn_matrix[:,0]
    targets = wgt_syn_matrix[:,1]
    weights = wgt_syn_matrix[:,2]
    
    return(wgt_syn_matrix)

def getWeightedSynapses(pre_array, post_array, neuron_matrix):
    
    syn_matrix = []
    
    for i in range(0, len(pre_array)):
        syn_matrix.append([pre_array[i], post_array[i]])
        
    syn_df = pd.DataFrame(data= syn_matrix , columns=["source", "target"])

    grouped_syn_df = syn_df.groupby(['source', 'target']).size().reset_index()
    summed_syn_matrix = grouped_syn_df.to_numpy()
    
    conductances = []
    
    for i in range(0,len(summed_syn_matrix)):

        neuron_type = neuron_matrix[summed_syn_matrix[i][0]-1][1]

        if (neuron_type == 'dspn' or neuron_type == 'ispn' ):
            #"conductance": [2.4e-10, 1e-10]
            cond = np.random.normal(2.4e-10,  1e-10, 1)

            if(cond<2.4e-11): #capping at 10% of mean
                cond = 2.4e-11
                
        elif (neuron_type == 'lts'):
            # conductance mean = 3e-09, std deviation =  0
            cond = 3e-09

        elif (neuron_type == 'fs'):
            #"conductance": [1.1e-09, 1.5e-09],
            cond = np.random.normal(1.1e-09, 1.5e-09, 1)

            if(cond<1.1e-10): #capping at 10% of mean
                cond = 1.1e-10

        conductances.append(cond)
    
    weights = []

    for i in range(0, len(conductances)):
        weights.append(-1*conductances[i]*summed_syn_matrix[i][2])

    sources = summed_syn_matrix[:,0]
    targets = summed_syn_matrix[:,1]
    
    return(sources, targets, weights)

def getInput(net, net_size):
    
    path = '/home/ubuntu/BasalGanglia/NEURON-data/' + net + '/input-spikes.hdf5'
    
    with h5py.File(path, 'r') as file:
    
        inputs = file.get('input')
        input_matrix = []

        for num in range(0, net_size-1):

            neuronID = str(num)

            neuron_i = inputs.get(neuronID)
            input_neuron_i = []

            source_list = list(neuron_i.keys())

            for source in source_list:
                spike_train = np.array(neuron_i.get(source).get('spikes'))

                for line in spike_train:

                    for spike_time in line:

                        if (spike_time!=-1):              
                            input_neuron_i.append(np.ceil((spike_time*1000) * 10) / 10)

            input_neuron_i.sort()
            input_matrix.append(input_neuron_i)
            
    return(input_matrix)



def getSpikes(path, size):
    
    '''
    Given the path to a spike .txt archive and the size of the network, copies the information onto a dictionary
    The keys are the IDs of each of the spiking neurons, and the corresponding value is an array
    with that neuron's spike times (in ns)
    
    '''
        
    with open(path  + '/spikes_2s.txt', 'r') as file:
        lines = file.readlines()

    spikes = {}
    
    for line in lines :
        
        split_line = np.array(line.split("\t"),float)
        nID = int(split_line[1])
        spike_time = split_line[0]
        if(nID not in spikes.keys()):
            spikes[nID] = []            
        spikes[nID].append(spike_time)
    
    for i in range(0, size):
        if(i not in spikes.keys()):
            spikes[i] = []
            
    return(spikes)

def getSubSpikes(spikes, matrix):
    ''' Receives the spike dict(with the IDs and times of spike) and neuron matrix(with IDs and neuron type), checks the type
        of each spiking neuron and adds the event into a dictionary for the specific type. The output are 5 dicts, each containing
        the IDs and spike times of one type of neuron (dspn, ipsn, lts chin and fs)
    
    '''
    
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

    return(dspn_spikes, ispn_spikes, lts_spikes, fs_spikes, chin_spikes)

def getVolts(file_path):
    
    '''
    Given the path to a volt.txt archive, copies the information onto a matrix and an array
    The matrix has one line for each neuron (line number = neuron_ID) and the values of the voltage of said neuron
    at each time instant on the different columns.
    The array has the time values for each voltage measure.
    Here I ask for the full path because there are multiple output files (for the diff simulation durations)
     on each network folder
    '''
        
    with open(file_path + '/volt_2s.txt', 'r') as file:
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

