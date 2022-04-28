#### Get some conllx data (i.e. universal dep format)
wsj data splits from: `git clone -q http://github.com/srush/temp`  
- NB: remove empty lines at the end of `.conllx` files
>
## POS prediction
- to read data in, use `from torch_struct.data import ConllXDatasetPOS'
- class def is modified from orignial/current torch-struct 

>
- Ammar et al. compare models simlar resp to 4 and 9 below (eval on p(y| x, \hat x) argamx)

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
`python hmm-grad.py` with "lik"

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7: 

- 0.2650 inacc on test sentences at 21 epochs, Adam: lr=0.01, weight_decay=0.2
- 0.2621 inacc at 201, Adam: lr=0.001, weight_decay=0.2
    - no clip_grad_norm

#### 3. hmm-grad-based-direct-max-marg-loglik(-unsup): 
`python hmm-grad.py` with "lik_u"

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7: 

- 0.8604 inacc on test sentences at 61 epochs (if and when, very high variance), Adam: lr=0.1, weight_decay=0.5
    - clip_grad_norm: 1.0

#### 4. hmm-grad-based-direct-max-marg-loglik-reconstruction: 
`python hmm-grad-rec.py` 

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7: 

- eval on p(y| x, \hat x) argamx
    - 0.7502 inacc on test at 81 epochs; Adam: lr=0.001, weight_decay=3.0,
        - clip_grad_norm: 1.0

- eval on z encoder.argmax 
    - 0.7502 inacc on test at 71 epochs (high var), Adam: lr=0.01, weight_decay=0.5
        - clip_grad_norm: 1.0


#### 5. hmm-em-analytic(-unsup): 
`python hmm-em.py` 

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7:

- convereges i.e. <img src="https://render.githubusercontent.com/render/math?math=loglik|_{\theta^{old}}-loglik|_{\theta^{old}}"> goes to 0 
- 0.7920 inacc on test, at convergence ~ 100 epochs (passes over train data)
- high var, rand init might be better than converged test inacc


#### 6. hmm-grad-based-em-analytic-reconstruction: 

>

#### 7. linear-chain-CRF-1hot(-sup): 
`python lincrf.py` 

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7:

- 0.3666 at 31, Adam: lr=0.1, weight_decay=0.5,
    - no clip_grad_norm

#### 8. linear-chain-CRF-1hot-direct-max-marg-loglik(-unsup): 
`python lincrf.py` with "loss1"

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7: 

- 0.7778 at 21, 0.6258 at 51, Adam:  lr=0.1, weight_decay=0.5,
    - clip_grad_norm: 1.0

#### 9. linear-chain-CRF-1hot-direct-max-marg-loglik-reconstruction: 
`python lincrf-rec.py` 

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7:

- eval on p(y| x, \hat x) argamx
    - 0.5109 inacc at 131, Adam: lr=0.001, weight_decay=5.0
        - clip_grad_norm: 1.0

- eval on z.argmax
    - 0.4986 inacc at 71, Adam: lr=0.001, weight_decay=3, 
    - 0.7160 inacc on test sentences at 51 epochs (high variance), Adam: lr=0.01, weight_decay=2
    - 0.5964 inacc at 41, Adam: lr=0.01, weight_decay=3,
        - clip_grad_norm: 1.0

#### 10. linear-chain-CRF-1hot-em-analytic-reconstruction: 

1174 train sentences, 45 held out test sentences; min_freq = 5, max_size=7:

- eval on p(y| x, \hat x) argamx
    -0.8623 at 55, lr=0.001, weight_decay=3.0,

- Zhang et al. Semi-sup


### bert pretrained embeddings:  

#### 11. linear-chain-CRF-bert(-sup): 
`python bert-pos.py`  

1174 train sentences, 45 held out test sentences: 

?? rerun
- 0.2203 inacc at 21 epochs on test sentences 
    - 
    - starts overfitting at 31 eps
    - min_freq = 10, max_size=7
    - 28095.17342400551 secds for 52 eps

#### 12. linear-chain-CRF-bert-direct-max-marg-loglik(-unsup): 
`python bert-pos.py` with "loss1"

1174 train sentences, 45 held out test sentences:

?? rerun
- 0.8870 inacc at 31 epochs on test sentences
    - starts overfitting by 41 eps: 0.9497 inacc
    - 0.9478 inacc at 21 epochs on test sentences 
    - min_freq = 10, max_size=7
    - 27461.224514961243 for 52 eps

#### 13. linear-chain-CRF-bert-direct-max-marg-loglik-reconstruction: 
- eval on p(y| x, \hat x) argamx

#### 14. linear-chain-CRF-bert-em-analytic-reconstruction: 

> 

##### ?? neuralized-hmms

>

#### POS (N)CRF-AE References 
- Ammar et al. 2014 https://arxiv.org/pdf/1411.1147.pdf
- Zhang X. et al. 2017 "Semi-Supervised Structured Prediction with Neural CRF Autoencoder" https://www.cs.purdue.edu/homes/dgoldwas/papers/ZJPTG_EMNLP17.pdf



## Dependency prediction (parent given child)
- to read in, use `from torch_struct.data import ConllXDataset'

#### 15. dep-CRF: 



####
