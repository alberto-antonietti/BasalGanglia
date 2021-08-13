###
####
###

import pandas as pd
import matplotlib
import random
from ipywidgets import *
import matplotlib.pyplot as plt
from getData import *

def plotNeuronalDistribution(folder_path):
    '''
    Given the path to a folder containing the network-neuron-position.hdf5 file of a network
    Plots 2 pie charts representing the neuron type distribution
    => path is '/NEURON data/net_X'
    '''
    matrix = getNeuronMatrix(folder_path)
    
    matrix_t = matrixTranspose(matrix)
    structure_df = {'neuronID': matrix_t[0][:], 'neuron_type': matrix_t[1][:]}
    df = pd.DataFrame(data=structure_df)
    cond = (df['neuron_type'] == 'lts') | (df['neuron_type'] == 'fs') | (df['neuron_type'] == 'chin')

    df_interneurons = df[cond]
    
    fig = plt.figure()

    ax = fig.add_subplot(1,4,1) # 2,1,1 means: 2:two rows, 1: one column, 1: first plot
    graph_1 = df.mask(cond,'Interneurons')['neuron_type'].value_counts().plot(kind='pie',autopct='%1.1f%%', radius=2)

    ax2 = fig.add_subplot(1,4,4) # 2,1,2 means: 2:two rows, 1: one column, 1: second plot
    graph_2 = df_interneurons['neuron_type'].value_counts().plot(kind='pie',autopct='%1.1f%%', radius = 2)

def plotTraces(folder_path, mode, spike_dict = None):
    '''
    Given the time array, voltage matrix (obtained through getVolts fction),
               neuron_submatrix (for the desired neuron type) and mode
    Plots the traces of 5 neurons of each type (if there are less than 5 available, plots all )
    
    MODES : R - randomly selected from the whole population of neurons
            F - randomly selected from those who fired
           

    '''
    
    voltage_matrix, time = getVolts(folder_path)
    
    neuron_matrix = getNeuronMatrix(folder_path)
    
    dspn_ID, ispn_ID, lts_ID, fs_ID, chin_ID = getNeuronSubMatrixes(neuron_matrix)
    submatrixes = [dspn_ID, ispn_ID, lts_ID, fs_ID, chin_ID]
    
    for x in submatrixes:

        if mode=='R' :

            if len(x)<5: #if there are less than 5 neurons available, we plot them all.
                plt.figure()
                for i in range(0, len(x)):
                    plt.plot(time, voltage_matrix[x[i]])
                plt.show()

            else:
                #in case there are at least 5 neurons
                random_IDs = random.sample(x, 5)  #we ramdomly select 5 values from the neuron_submatrix

                print("Plotting the traces of the following neurons : ", random_IDs)
                plt.figure()
                for i in range(0,5):
                    plt.plot(time, voltage_matrix[random_IDs[i]])
                plt.show()


        if mode == "F" :

            if spike_dict == None :
                print("The spike dictionary must be provided to plot random firing neurons ")
                return()

            else : 

                #get the IDs of the neurons that have fired from the keys of the dict
                firing_IDs = list(spike_dict.keys())

                if len(firing_IDs)<5: #if there are less than 5 neurons available, we plot them all.

                    plt.figure()
                    for i in range(0, len(x)):
                        plt.plot(time, voltage_matrix[x[i]])
                    plt.show()

                else:
                    #in case there are at least 5 firing neurons
                    random_IDs = random.sample(firing_IDs, 5)  #we ramdomly select 5 values from the keys

                    print("Plotting the traces of the following neurons : ", random_IDs)
                    plt.figure()
                    for i in range(0,5):
                        plt.plot(time, voltage_matrix[random_IDs[i]])
                    plt.show() 
    
    return()


