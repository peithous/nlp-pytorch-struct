### POS Prediction
wsj data splits from: `git clone -q http://github.com/srush/temp` \
- remove empty lines at the end of `.conllx` files

#### HMM 1-hot MLE: 
`python hmm-mle.py` \
- ~ 93.9% accurate on test sentences (38992 train sentences, 2416 test sentences) \
- ~4 hours on CPU with 100 sample batches

#### Linear Chain CRF 1-hot: 
~

#### Linear Chain CRF Bert embeddomgs: 
~

#### EK HMM: 


### Dependency Prediction

