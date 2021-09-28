### POS Prediction
wsj data splits from: `git clone -q http://github.com/srush/temp`  
- NB: Remove empty lines at the end of `.conllx` files

- "[(Wu and Dredze, 2019)] found [POS tagging] accuracies over 97% across 15 languages from the Universal Dependency (UD) treebank. Accuracies on various English treebanks are also 97% (no matter the algorithm; HMMs, CRFs, BERT perform similarly). This 97% number is also about the human performance on this task, at least for English (Manning, 2011)." (Jurasky, 3rd ed. , 8.2)  

#### HMM 1-hot MLE: 
`python hmm-mle.py`  
- 93.9% accurate on test sentences (38992 train sentences, 2416 held out test sentences)  
- ~4 hours on CPU with 100 sample batches

#### Linear Chain CRF 1-hot: 
~ 

#### Linear Chain CRF Bert embeddomgs: 
~

#### EM HMM: 


### Dependency Prediction

