import numpy as np
import matplotlib.pyplot as plt
from synthetic_signal_utils import *

nsamples = 64               #No. of samples in each signal
train_cnt = 3000            #No. training of samples
maxbw=30                    #Maximum BW
minbw=7                     #Minimum BW
fcnt=6                      #No. of frequency bins      (Freq_bin = F_sample/n_samples)
scnt = nsamples * fcnt      #Sampling frequency
predict=False
noiseval = 0.01               
mindb=5                     #Minimum SNR    (in dB)
maxdb=20                    #Maximum SNR    (in dB)
plotenable=False

########################################################################################################
################## Types of synthetically generated signal data ########################################
#ltraffic=["single_cont", "mult_cont", "single_rshort", "mult_rshort", "det_hop"]
ltraffic=["single_cont", "single_rshort", "mult_cont", "det_hop"]
#ltraffic=["single_cont", "single_rshort", "det_hop"]
# dl=[]
# for el in ltraffic:
#     dl += [el]+["dummy"]*2
#ltraffic=dl
#########################################################################################################


#Initializing lists
datatype = "float32"
train_data = np.zeros((1,fcnt,nsamples),dtype=datatype)         #To store training data
train_labels = np.zeros((1,len(ltraffic)),dtype=datatype)       #To store training labels
bw_labels = []                                                  #Position labels
pos_labels = []                                                 #Position labels
nsig_labels = []                                                #No. of modulation signals labels

def gaussian(cnt):
    '''Generates gaussian noise signal and performs signal processing operations
    
    cnt: Int
        No. of samples in each signal.
    '''
    #complex Numpy array of shape (cnt, fcnt, nsamples)
    indata = noiseval/np.sqrt(2)* (np.random.normal(size=(cnt,fcnt,nsamples)) + 1j* np.random.normal(size=(cnt,fcnt,nsamples)))
    
    #FFT -> normalization (by nsamples) -> convert amplitude into dB
    outdata = 20*np.log10(np.abs(np.fft.fft(indata)/float(nsamples)))
    
    #Shifts zero frequency component ti centre along third axis 
    outdata = np.fft.fftshift(outdata,axes=(2,))
    
    
    #outdata = -np.min(outdata)+outdata
    #outdata = (outdata- np.min(outdata))/(1 - np.min(outdata))
    return outdata

if predict:
	pred_data = np.zeros((1,pcnt,nsamples),dtype=datatype)                      #Initializing array for storing prediction data.

def gendata(noise=True, normalize=True):
    '''Generated synthetic data for different signal types.
    noise : Boolean
        If true, noise will be added to the generated signal.
    normalize : Boolean
        If true, signal is converted to dB and then normalized.
    '''
    global train_data, train_labels, bw_labels, pos_labels, nsig_labels         #Use global variables
    idx=0
    for traffic in ltraffic:                                                    
        if traffic == "single_cont":                                            #Continuous single tone modulation
           gen_single_cont(train_cnt, plotenable)
        elif traffic == "mult_cont":                                            #Multiple continuous single tone modulation
           gen_multi_cont(train_cnt, plotenable)
        elif traffic == "single_rshort":                                        #Short duration random binary patterns
           gen_single_rshort(train_cnt, plotenable)
        elif traffic == "mult_rshort":                                          #Multiple short duration random binary patterns
           gen_multi_rshort(train_cnt, plotenable)
        elif traffic == "det_hop":                                              #Deterministic hop signal
           gen_det_hop(train_cnt, plotenable)
        else:
            print("Traffic not defined")
        idx+=1
    
    
    #Removing first rows of each array (was initialised with zeroes as a placeholder)
    train_data = np.delete(train_data,0,0)                                          
    train_labels = np.delete(train_labels,0,0)
    bw_labels = np.reshape(bw_labels,(-1,1))
    pos_labels = np.reshape(pos_labels,(-1,1))
    nsig_labels = np.reshape(nsig_labels,(-1,1))
    if predict:
        pred_data = np.delete(pred_data,0,0)                        #removing first row of zeroes, if initialized. 
        print("Predict data:",pred_data.shape)                      #prints shape of pred data
    print("--"*50)
    print("Training data:",train_data.shape)                        #prints shape of training data
    print("--"*50)
    #print len(train_data), len(train_labels), len(bw_labels), len(pos_labels)
    
    if noise:                                                       #add noise if True
        #train_data_nn = train_data
        #train_data = train_data+gaussian(train_data.shape[0])
        train_data = train_data+ noiseval
        
    train_data_org = np.copy(train_data)
    train_data = 20 * np.log10(train_data)                          #Convert to dB
    nmin = np.min(train_data)                                       #store minimum value
    train_data = -np.min(train_data)+train_data                     #shift to non-negative values (by subtracting most negative value)
    nmax = np.max(train_data)                                       #store maximum value
    train_data = (train_data- np.min(train_data))/(np.max(train_data) - np.min(train_data))         #normalization  
    
    return train_data, train_labels, bw_labels, pos_labels, nmin, nmax,  train_data_org

