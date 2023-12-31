
# saife



Wireless spectrum anomaly detector

  

Non-documented SAFE base code used for the following papers

  

1. SAIFE: Unsupervised wireless spectrum anomaly detection with interpretable features

2. Unsupervised Wireless Spectrum Anomaly Detection With Interpretable Features

3. Crowdsourced wireless spectrum anomaly detection

  

Adapted from: https://github.com/hwalsuklee/tensorflow-mnist-AAE .

Thanks to musyoku's pytorch implementaion for all the unsupervised setup and quick answers (https://github.com/musyoku/adversarial-autoencoder).

  

Requirements: Tested on

  

Python 3.6.9, Tensorflow 1.12.0, Tflearn

  
  

Sample run:

  

python spectrum_semsup_cat.py --prior_type normal --num_epochs=510 --PMLR_n_samples=500 --learn_rate=0.05e-3 --dimz=50

  

More documentation and examples will be updated soon.

## Setup
Execute following commands on Terminal:

 1. `conda create -n saife python=3.6.9`
 
 2. `conda activate saife`
 
 3. `pip install tensorflow==1.15`

4. `pip install scipy==1.2`

5. `pip install tflearn`

6. `pip install matplotlib pandas`

### Sample Run: 
Run following command: 

    python spectrum_semsup_cat.py --prior_type normal --num_epochs=10 --PMLR_n_samples=500 --learn_rate=0.05e-3 --dimz=50
