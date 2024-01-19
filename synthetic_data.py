import numpy as np
import matplotlib.pyplot as plt

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
########################### Types of synthetically generated signal data ###############################
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
            for i in range(train_cnt):                                          #Iterate over each training samples
                sigbw = np.random.randint(minbw,maxbw)                          #Random modulation signal BW
                sigpos = np.random.randint(0, nsamples-sigbw)                   #Random modulation signal position
                bw_labels.append((sigbw/float(nsamples)))                       #Calculate and save BW label
                pos_labels.append((sigpos+(sigbw/2.0))/float(nsamples))         #Calculate and save modulation signal position label
                nsig_labels.append(1)                                           #Single tone (1 modulation signal)
                sigon1=np.concatenate((np.ones((fcnt,sigbw)),np.zeros((fcnt,nsamples-sigbw))),axis=1)       #Binary signal
                sigon1 = np.roll(sigon1,sigpos,axis=1).reshape(1,fcnt,nsamples) #time shift (as per signal position) and reshape the signal
                snr = np.random.uniform(mindb,maxdb)                            #random generated SNR 
                rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw            #calculates modulation signal intensity (received) from generated SNR
                sigon1 =  sigon1 * rintens #+ gaussian()                        
                train_data   = np.vstack((train_data,sigon1))                   #stacking of generated signal data and label
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:                                                      #plot
                plt.figure(1)
                # print pos_labels[0]*nsamples
                # print bw_labels[0]*nsamples
                plt.imshow(sigon1[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "mult_cont":                                            #Multiple continuous single tone modulation
            for i in range(train_cnt):                                          #Iterate over training samples
                nsig = np.random.randint(2,5)                                   #No of modulation signals
                actsig=np.zeros((fcnt,nsamples))                                #place holder     
                posum=0
                for i in range(nsig):                                           #Iterate over each signal (perform same operations as continuous single tone modulation)
                    sigbw = np.random.randint(minbw,maxbw/2)
                    sigpos = np.random.randint(0, nsamples-sigbw)
                    posum += sigpos+(sigbw/2.0)
                    sigon1=np.concatenate((np.ones((fcnt,sigbw)),np.zeros((fcnt,nsamples-sigbw))),axis=1)
                    sigon1 = np.roll(sigon1,sigpos,axis=1).reshape(1,fcnt,nsamples)
                    snr = np.random.uniform(mindb,maxdb)
                    rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                    sigon1 =  sigon1 * rintens 
                    actsig = actsig + sigon1                                    #Sum all modulation signals
                orgbw = len(np.where(actsig[0]>0)[0])                           #calculate total occupied BW
                bw_labels.append((orgbw/float(nsamples)))                       #Calculate and save BW label
                pos_labels.append(posum/float(nsamples)/nsig)                   #Calculate and save signal position label
                nsig_labels.append(nsig)                                        #Calculate and save no. of modulation signals label
                actsig = actsig #+ gaussian()  
                train_data   = np.vstack((train_data,actsig))                   #stack generated signal data and label
            dlabels=np.zeros((train_cnt,len(ltraffic)))             
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:                                                      #plot
                plt.figure(1)
                plt.imshow(actsig[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "single_rshort":                                        #Short duration random binary patterns
            for i in range(train_cnt):                                          #Iterate over each training sample   
                sigbw = np.random.randint(minbw,maxbw)                          #Random  signal BW
                sigpos = np.random.randint(0, nsamples-sigbw)                   #Random  signal BW
                bw_labels.append((sigbw/float(nsamples)))                       #Calculate and save BW label
                pos_labels.append((sigpos+(sigbw/2.0))/float(nsamples))         #Calculate and save modulation signal position label
                nsig_labels.append(1)                                           #Single tone (1 modulation signal)
                ipres = np.random.randint(2,size=fcnt).reshape(fcnt,1)          #Random binary column vector
                sig = np.repeat(ipres,sigbw,axis=1)                             #Repeat each value of column vector sigbw times
                sigon1=np.zeros((fcnt,nsamples))                                #place holder
                sigon1[:,sigpos:sigpos+sigbw] = sig                             #Insert random binary signal as per its 'sigpos' in the final signal
                snr = np.random.uniform(mindb,maxdb)                            #random generated SNR 
                rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw            #Calculate final signal intensity (received) from generated SNR
                sigon1 =  sigon1 * rintens #+ gaussian()  
                sigon1= sigon1.reshape(1,fcnt,nsamples)
                train_data   = np.vstack((train_data,sigon1))                   #stacking of generated signal data and label
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:                                                      #plot
                plt.figure(1)
                plt.imshow(sigon1[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "mult_rshort":                                          #Multiple short duration random binary patterns
            for i in range(train_cnt):                                          #Iterate over each training sample
                nsig = np.random.randint(2,5)                                   #No. of modulation signals
                actsig=np.zeros((fcnt,nsamples))                                #place holder
                posum=0
                for i in range(nsig):                                           #Iterate over each signal (perform same operations as short duration random binary patterns)
                    sigbw = np.random.randint(minbw,maxbw/2)
                    sigpos = np.random.randint(0, nsamples-sigbw)
                    posum += sigpos+(sigbw/2.0)
                    ipres = np.random.randint(2,size=fcnt).reshape(fcnt,1)
                    sig = np.repeat(ipres,sigbw,axis=1)
                    sigon1=np.zeros((fcnt,nsamples))
                    sigon1[:,sigpos:sigpos+sigbw] = sig
                    snr = np.random.uniform(mindb,maxdb)
                    rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                    sigon1 =  sigon1 * rintens 
                    actsig = actsig + sigon1                                    #Sum all modulation signals
                actsig = actsig.reshape(1,fcnt,nsamples)                        #Reshape the sum signal
                #maxpos = np.argmax(np.sum(actsig>0,axis=0))
                #orgbw = len(np.where(actsig[0][maxpos]>0)[0])
                orgbw = np.max(np.sum(actsig>0,axis=0))                         #calculate total occupied BW
                bw_labels.append((orgbw/float(nsamples)))                       #Calculate and save BW label
                pos_labels.append(posum/float(nsamples)/nsig)                   #Calculate and save signal position label
                nsig_labels.append(nsig)                                        #Calculate and save no. of modulation signals label
                actsig = actsig #+ gaussian()  
                train_data   = np.vstack((train_data,actsig))                   #stacking of generated signal data and label
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:                                                      #plot
                plt.figure(1)
                plt.imshow(actsig[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
        elif traffic == "det_hop":                                              #Deterministic hop signal
            for i in range(train_cnt):                                          #Iterate over each training sample  
                sigbw = np.random.randint(minbw,maxbw/2)                        #Random generated BW
                sigpos = np.random.randint(0, nsamples-sigbw)                   #Random generated signal position
                detshift = np.random.randint(1,maxbw/4)                         #Random generated hopping shift values
                initarr=np.zeros(nsamples)                                      #place holder
                initarr[sigpos:sigpos+sigbw] = 1                                #Binary signal with '1' at determined locations (sigpos to sigpo+sigbw)
                res=[]                                                          #place holder
                for j in range(fcnt):                                           #roll and save intermediate signal (repeated fcnt times) 
                    res.append(initarr)                                         
                    initarr= np.roll(initarr,detshift)                      
                res = np.reshape(res,(1,fcnt,nsamples))                         #reshape signal
                #orgbw = sigbw+detshift*fcnt
                orgbw = sigbw                                                   #Total signal bw
                bw_labels.append((orgbw/float(nsamples)))                       #Calculate and save BW label
                pos_labels.append((sigpos+(sigbw/2.0)+(fcnt*detshift/2))%nsamples/float(nsamples))      #Calculate and save signal position label
                snr = np.random.uniform(mindb,maxdb)                            #Random generated SNR
                rintens = 10**(snr/20.0) * noiseval * nsamples/sigbw 
                res = res * rintens #+ gaussian()                               #Calculate signal intensity (received) from generated SNR
                train_data = np.vstack((train_data,res))                        #stacking of generated signal data and label
                nsig_labels.append(1)
            dlabels=np.zeros((train_cnt,len(ltraffic)))
            dlabels[:,idx]=1
            train_labels = np.vstack((train_labels,dlabels))
            if plotenable:                                                      #plot
                plt.figure(1)
                plt.imshow(res[0], interpolation='none')
                plt.colorbar(orientation='vertical')
                plt.show()
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

