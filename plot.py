import pandas as pd
import matplotlib
import random
from ipywidgets import *
import matplotlib.pyplot as plt
from getData import *

chin_color='xkcd:orange'
ispn_color='xkcd:blue purple'
lts_color='xkcd:bright purple'
dspn_color='xkcd:sky blue'
fs_color='xkcd:dark blue'

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
    
    fig.suptitle('Neuron Type Distribution', fontsize=14, fontweight='bold')
    
    ax = fig.add_subplot(1,4,1) # 2,1,1 means: 2:two rows, 1: one column, 1: first plot
    graph_1 = df.mask(cond,'Interneurons')['neuron_type'].value_counts().plot(kind='pie',ylabel='',autopct='%1.1f%%', radius=2, colors = [dspn_color, ispn_color, 'xkcd:grey'])

    ax2 = fig.add_subplot(1,4,4) # 2,1,2 means: 2:two rows, 1: one column, 1: second plot
    graph_2 = df_interneurons['neuron_type'].value_counts().plot(kind='pie', ylabel='',autopct='%1.1f%%',  textprops={'color':"w"}, radius = 2, colors = [fs_color, chin_color, lts_color])
    #ax2.legend(loc=3, labels=df_interneurons.index)


def plotTraces(net, mode, spike_dict = None):
    '''
    Given the time array, voltage matrix (obtained through getVolts fction),
               neuron_submatrix (for the desired neuron type) and mode
    Plots the traces of 5 neurons of each type (if there are less than 5 available, plots all )
    
    MODES : R - randomly selected from the whole population of neurons
            F - randomly selected from those who fired
           

    '''
    
    voltage_matrix, time = getVolts(net)
    for i in range (0, len(time)):
        time[i] = time[i]/1000
        
    neuron_matrix = getNeuronMatrix(net)
    
    dspn_ID, ispn_ID, lts_ID, fs_ID, chin_ID = getNeuronSubMatrixes(neuron_matrix)
    submatrixes = [dspn_ID, ispn_ID, lts_ID, fs_ID, chin_ID]
    colors = [dspn_color, ispn_color, lts_color, fs_color, chin_color]
    
    neuron_types =  ['dSPN', 'iSPN', 'LTS', 'FSN', 'ChIN']
    
    for i in range(0, len(submatrixes)):
        
        title = "Traces of Random " + neuron_types[i] +  " Cells"

        if mode=='R' :

            if len(submatrixes[i])<5: #if there are less than 5 neurons available, we plot them all.
                fig = plt.figure(figsize=(5,7))
                ax = fig.add_subplot(211)
                ax.set_title(title)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(' Voltage [mV]')
                for j in range(0, len(submatrixes[i])):
                    plt.plot(time, voltage_matrix[submatrixes[i][j]], color = colors[i])
                plt.show()

            else:
                #in case there are at least 5 neurons
                random_IDs = random.sample(submatrixes[i], 5)  #we ramdomly select 5 values from the neuron_submatrix

                print("Plotting the traces of the following neurons : ", random_IDs)
                fig = plt.figure(figsize=(5,7))
                ax = fig.add_subplot(211)
                ax.set_title(title)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(' Voltage [mV]')
                for j in range(0,5):
                    plt.plot(time, voltage_matrix[random_IDs[j]], color = colors[i])
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

def plotSpikes(events, id_array, raster_order, color, label, ax):
    
    label_done = False
    
    for neuron_id in id_array:
        
        spikes = events[neuron_id] #array of the spike times of neuron i
        
        index = raster_order.index(neuron_id)
        if not label_done:
            ax.plot(spikes, np.full_like(spikes, index), ms = 3, marker=".", label=label, color=color, linestyle="None")

            #this full_like function generates an array that has the same size as events[i], with the value index
            #on every position (so we #have the same number of x and ys for plotting)
            label_done = True
        else:
            ax.plot(spikes, np.full_like(spikes, index), ms = 3, marker=".", color=color, linestyle="None")
    return();


def plotRaster(net, size):
    ''' Given a net 'net_XXX', generates the raster plot for the corresponding simulation
    '''    
    spikes_dict = getSpikes(net, size)
    
    n_ids = np.array(list(spikes_dict.keys()), dtype=int)
    n_ids.sort() #organize the neurons by id

    neuron_matrix = getNeuronMatrix(net)
    dspn_ID, ispn_ID, lts_ID, fs_ID, chin_ID = getNeuronSubMatrixes(neuron_matrix)
    
    orderedIDs = getOrderedIDs(dspn_ID, ispn_ID, lts_ID, fs_ID, chin_ID)


    fig_handle = plt.figure()
    ax = fig_handle.add_subplot(111)
    ax.set_xlabel('$t$ (ms)')
    ax.axes.yaxis.set_visible(False)

    plotSpikes(spikes_dict, fs_ID, orderedIDs, fs_color, 'fs', ax)
    plotSpikes(spikes_dict, dspn_ID, orderedIDs,dspn_color, 'dspn', ax)
    plotSpikes(spikes_dict, lts_ID, orderedIDs, lts_color, 'lts', ax)
    plotSpikes(spikes_dict, ispn_ID, orderedIDs,ispn_color , 'ispn', ax)
    plotSpikes(spikes_dict, chin_ID, orderedIDs, chin_color, 'chin', ax)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def ComputeFiringRates(net):

    neuron_IDs = getNeuronMatrix(net)

    file_path = '/home/ubuntu/BasalGanglia/NEURON-data/' + net + '/spikes.txt'

    dspn_spks = []
    ispn_spks = []
    lts_spks = []
    chin_spks = []
    fs_spks = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    spikes = {}

    for line in lines :

        split_line = np.array(line.split("\t"),float)
        nID = int(split_line[1])
        spike_time = split_line[0]

        if neuron_IDs[nID][1] == 'dspn':
            dspn_spks.append(spike_time)

        elif neuron_IDs[nID][1] == 'ispn':
            ispn_spks.append(spike_time)

        elif neuron_IDs[nID][1] == 'fs':
            fs_spks.append(spike_time)

        elif neuron_IDs[nID][1] == 'chin':
            chin_spks.append(spike_time)

        elif neuron_IDs[nID][1] == 'lts':
            lts_spks.append(spike_time)

    dspn_spikes_dt = []
    ispn_spikes_dt = []
    chin_spikes_dt = []
    lts_spikes_dt = []
    fs_spikes_dt = []

    for d_time in range (0, 1981, 20):

        start = d_time
        stop = d_time + 20
        spikes_dt_aux = [0,0,0,0,0]

        for elem in dspn_spks:
            if (start < elem <= stop):
                spikes_dt_aux[0] +=1
        for elem in ispn_spks:
            if (start < elem <= stop):
                spikes_dt_aux[1] +=1
        for elem in chin_spks:
            if (start < elem <= stop):
                spikes_dt_aux[2] +=1
        for elem in lts_spks:
            if (start < elem <= stop):
                spikes_dt_aux[3] +=1
        for elem in fs_spks:
            if (start < elem <= stop):
                spikes_dt_aux[4] +=1

        dspn_spikes_dt.append(spikes_dt_aux[0])
        ispn_spikes_dt.append(spikes_dt_aux[1])
        chin_spikes_dt.append(spikes_dt_aux[2])
        lts_spikes_dt.append(spikes_dt_aux[3])
        fs_spikes_dt.append(spikes_dt_aux[4])

    dspn, ispn, lts, fs, chin = getNeuronSubMatrixes(neuron_IDs)
    N_dspn = len(dspn)
    N_ispn = len(ispn)
    N_chin = len(chin)
    N_lts= len(lts)
    N_fs = len(fs)

    dspn_fr = []
    ispn_fr = []
    chin_fr = []
    lts_fr = []
    fs_fr = []

    for elem in dspn_spikes_dt:
        dspn_fr.append(elem*50/(N_dspn))
    for elem in ispn_spikes_dt:
        ispn_fr.append(elem*50/(N_ispn))
    for elem in chin_spikes_dt:
        chin_fr.append(elem*50/(N_chin))
    for elem in lts_spikes_dt:
        lts_fr.append(elem*50/(N_lts))
    for elem in fs_spikes_dt:
        fs_fr.append(elem*50/(N_fs))

    time = []
    for d_time in range (0, 1981, 20):
        time.append(d_time/1000)
        
    return(time, dspn_fr, ispn_fr, chin_fr, lts_fr, fs_fr)


def plotFR(net):
    
    time, dspn_fr, ispn_fr, chin_fr, lts_fr, fs_fr = ComputeFiringRates(net)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axes.xaxis.set_visible(False)

    
    plt.step(time, lts_fr, color = lts_color, linewidth=0.5)
    plt.step(time, chin_fr, color = chin_color, linewidth=0.5)
    plt.step(time, fs_fr, color = fs_color, linewidth=0.5)
    plt.step(time, dspn_fr, color = dspn_color, linewidth=1)
    plt.step(time, ispn_fr, color = ispn_color, linewidth=1)
    
    ax.set_title("Firing Rates")
    ax.set_ylabel(' Firing Rate [Hz]')

    plt.show()
    
    #------------------------------------------------------------#
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0,0,1,1])

    plt.step(time, lts_fr, color = lts_color, linewidth=0.5)
    plt.step(time, chin_fr, color = chin_color, linewidth=0.5)
    plt.step(time, fs_fr, color = fs_color, linewidth=0.5)
    
    ax1.set_title("Firing Rates of the Interneurons")
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(' Firing Rate [Hz]')
    
    plt.show()
    
    #------------------------------------------------------------#    
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0,0,1,1])
    plt.step(time, dspn_fr, color = dspn_color, linewidth=1)
    plt.step(time, ispn_fr, color = ispn_color, linewidth=1)
    ax2.set_title("Firing Rates of dSPN and iSPN")
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel(' Firing Rate [Hz]')
    
    plt.show()
    
    return
