import numpy as np
import matplotlib.pyplot as plt

def gen_single_cont(train_data,train_labels,train_cnt, idx, plotenable):
    '''Generates synthetic signal of type 'Continuous single tone modulation'. 
    
    train_data : NumPy Nd Array
        Signal for training
    train_labels : NumPy Nd Array
        Signal labels for training
    train_cnt : Int
        No. of training samples to generate.
    idx : Int
        Signal index.
    plotenable : Boolean
        If True, signal plot is generated.
    '''
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
    
    return train_data, train_labels
    
def gen_multi_rshort(train_data,train_labels,train_cnt, idx,  plotenable):
    '''Generates synthetic signal of type 'Multiple short duration random binary pattern'. 

    train_data : NumPy Nd Array
        Signal for training
    train_labels : NumPy Nd Array
        Signal labels for training 
    train_cnt : Int
        No. of training samples to generate.
    idx : Int
        Signal index.
    plotenable : Boolean
        If True, signal plot is generated.
    '''
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
    
    return train_data, train_labels

def gen_single_rshort(train_data,train_labels,train_cnt, idx,  plotenable):
    '''Generates synthetic signal of type 'Short duration random binary pattern'. 

    train_data : NumPy Nd Array
        Signal for training
    train_labels : NumPy Nd Array
        Signal labels for training
    train_cnt : Int
        No. of training samples to generate.
    idx : Int
        Signal index.
    plotenable : Boolean
        If True, signal plot is generated.
    '''
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
    
    return train_data, train_labels
def gen_multi_cont(train_data,train_labels,train_cnt, idx,  plotenable):
    '''Generates synthetic signal of type 'Multiple continuous single tone modulation'. 
    
    train_data : NumPy Nd Array
        Signal for training
    train_labels : NumPy Nd Array
        Signal labels for training
    train_cnt : Int
        No. of training samples to generate.
    idx : Int
        Signal index.
    plotenable : Boolean
        If True, signal plot is generated.
    '''
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
        
    return train_data, train_labels

def gen_det_hop(train_data,train_labels,train_cnt, idx, plotenable):
    '''Generates synthetic signal of type 'Deteministic hop signal'. 
    
    train_data : NumPy Nd Array
        Signal for training
    train_labels : NumPy Nd Array
        Signal labels for training
    train_cnt : Int
        No. of training samples to generate.
    idx : Int
        Signal index.
    plotenable : Boolean
        If True, signal plot is generated.
    '''   
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
        
    return train_data, train_labels

def gen_dt(train_data,train_labels,train_cnt, plotenable):
    pass
    # return train_data, train_labels
    