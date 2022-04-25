#### Get some conllx data (i.e. universal dep format)
wsj data splits from: `git clone -q http://github.com/srush/temp`  
- NB: remove empty lines at the end of `.conllx` files
>
## POS prediction
- to read data in, use `from torch_struct.data import ConllXDatasetPOS'
- class def is modified from orignial/current torch-struct 

### non-neural baselines

#### 1. hmm-1hot-analytic-counts(-sup): 
`python hmm-mle.py`  
- 0.1614 inacc on test sentences 
    - 1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7
    - 174 secds CPU
>
w/ more data
- 93.9% accurate on test sentences (38992 train sentences, 2416 held out test sentences)  
    - w/out hidden state space restriction 
    - 3min on CPU with 100 sample batches

#### 2. hmm-gradient-based(-sup): 
`python hmm-grad.py` with "loss"
- 0.2650 inacc on test sentences at 21 epochs, Adam: lr=0.01, weight_decay=0.2
- 0.2621 inacc at 201, Adam: lr=0.001, weight_decay=0.2
    - no clip_grad_norm
    - 1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7

#### 3. hmm-grad-based-direct-max-marg-loglik(-unsup): 
`python hmm-grad.py` with "loss1"
-  0.8604 inacc on test sentences at 61 epochs (if and when, very high variance)
    - Adam: lr=0.1, weight_decay=0.5
    - clip_grad_norm: 1.0
    - 1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7

#### 4. hmm-grad-based-direct-max-marg-loglik-reconstruction: 
`python hmm-reconstruct.py` 
- inacc on test sentences at  epochs (very high variance)
    - Adam: lr=0.01, weight_decay=0.5
    - clip_grad_norm: 1.0
    - 1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7

#### 5. hmm-em-analytic(-unsup): 
`python hmm-em.py` 
- convereges i.e. <img src="https://render.githubusercontent.com/render/math?math=loglik|_{\theta^{old}}-loglik|_{\theta^{old}}"> goes to 0 
- 0.8557 inacc on test, at 100 epochs (passes over train data)
- high var, rand init might be better than converged test inacc
    - 1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7

#### 6. hmm-grad-based-em-analytic-reconstruction: 


>

#### 7. linear-chain-CRF-1hot(-sup): 

#### 8. linear-chain-CRF-1hot-direct-max-marg-loglik(-unsup): 


#### 9. linear-chain-CRF-1hot-direct-max-marg-loglik-reconstruction: 


#### 10. linear-chain-CRF-1hot-em-analytic-reconstruction: 



### bert pretrained embeddings:  

#### 1. linear-chain-CRF-bert(-sup): 
`python bert-pos.py`  
?? rerun
- 0.2203 inacc at 21 epochs on test sentences 
    - 1174 train sentences, 45 held out test sentences
    - starts overfitting at 31 eps
    - min_freq = 10, max_size=7
    - 28095.17342400551 secds for 52 eps

#### 2. linear-chain-CRF-bert-direct-max-marg-loglik(-unsup): 
`python bert-pos.py` with "loss1"
?? rerun
- 0.8870 inacc at 31 epochs on test sentences
    - 1174 train sentences, 45 held out test sentences

    - starts overfitting by 41 eps: 0.9497 inacc
    - 0.9478 inacc at 21 epochs on test sentences 
    - min_freq = 10, max_size=7
    - 27461.224514961243 for 52 eps

#### 3. linear-chain-CRF-bert-direct-max-marg-loglik-reconstruction: 


#### 4. linear-chain-CRF-bert-em-analytic-reconstruction: 


> 
##### ?? neuralized-hmms

>

## Dependency prediction (parent given child)
- to read in, use `from torch_struct.data import ConllXDataset'

#### dep-CRF: 