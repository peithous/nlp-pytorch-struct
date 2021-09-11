from torch.utils.tensorboard import SummaryWriter
import torchtext.data as data
from torchtext.data import BucketIterator
import torch
from torch.distributions import Categorical, Uniform
from torch_struct import HMM
import matplotlib.pyplot as plt
from torch_struct.data.trees import ConllXDatasetPOS
#writer = SummaryWriter(log_dir="hmm-1hot")

WORD = data.Field(pad_token=None, eos_token='<eos>') #init_token='<bos>', 
POS = data.Field(include_lengths=True, pad_token=None, eos_token='<eos>') 

fields = (('word', WORD), ('pos', POS), (None, None))
train = ConllXDatasetPOS('data/wsj.train0.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 50) #en_ewt-ud-train.conllu
test = ConllXDatasetPOS('data/test0.conllx', fields)
print('total train sentences', len(train))

WORD.build_vocab(train) # min_freq = 5
POS.build_vocab(train)
train_iter = BucketIterator(train, batch_size=10, device='cpu', shuffle=False)
test_iter = BucketIterator(test, batch_size=10, device='cpu', shuffle=False)

C = len(POS.vocab)
V = len(WORD.vocab)
print('C', C, POS.vocab.itos, '\n', 'V', V, WORD.vocab.itos)

# estimate mle's 
tags = [] # prior
bigrams = [] # transition
word_tag_counts = [] # enission
for ex in train_iter:
#    print(ex.pos)
    words = ex.word
    label, lengths = ex.pos
    for batch in range(label.shape[1]):
#        print(' '.join([WORD.vocab.itos[i] for i in words[:lengths[batch], batch]]))        
        tags.append(label[:lengths[batch], batch])
        bigrams.append(label[:lengths[batch], batch].unfold(0, 2, 1)) #dimension, size, step      
        for i, t in enumerate(label[:lengths[batch], batch]):
            word_tag_counts.append(torch.tensor((t.item(), words[i, batch].item())))
tags = torch.cat(tags, 0)
bigrams = torch.cat(bigrams, 0)
word_tag_counts = torch.stack(word_tag_counts)

# priors tensor
init = torch.zeros(C).long() 
init.index_put_((tags,), torch.tensor(1), accumulate=True)
assert init[POS.vocab.stoi['<eos>']] == len(train)
init = init.float() / init.sum()
assert torch.isclose(init.sum(), torch.tensor(1.))# \sum_C p_c = 1
init = init.log().long().float()

# transition tensor
transition = torch.zeros((C, C)).long() 
transition[0, :] = 1 # p(c_i | c_i-1 = unk) = 1/|C| 
transition.index_put_((bigrams[:, 0], bigrams[:, 1]), torch.tensor(1), accumulate=True)
transition = (transition.float() / transition.sum(-1, keepdim=True)).transpose(0, 1) 
for tminus1 in range(transition.shape[1]): # Q1: is there a better way to correct the nan's?
    if tminus1==POS.vocab.stoi['<eos>']: # p(. | eos) = 0
        transition[:, tminus1] = 0 
assert torch.isclose(transition.sum(0, keepdim=True).sum(), torch.tensor(C-1.)) # for all x \in C-{eos}, \sum_C  p(c, x) = 1
transition = transition.log().long().float()
#print(transition[POS.vocab.stoi['<unk>'], :]) # Q2: p(c_i = unk | c_i-1 ) != 1/|C| 

# emission tensor
emission = torch.zeros((C, V)).long()
emission[0, :] = 1 # p(v_i | c_i-1 = unk) = 1/|V| 
emission.index_put_((word_tag_counts[:, 0], word_tag_counts[:, 1]), torch.tensor(1), accumulate=True)
emission = (emission.float() / emission.sum(-1, keepdim=True)).transpose(0, 1)
assert torch.isclose(emission.sum(0, keepdim=True).sum(), torch.tensor(C, dtype=torch.float)) # for all c \in C, \sum_V p_c (v) = 1
emission = emission.log().long().float()

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

def trn(train_iter):
    losses = []
    for ex in train_iter:
        label, lengths = ex.pos
        observations = torch.LongTensor(ex.word).transpose(0, 1).contiguous()  

        dist = HMM(transition.type(torch.FloatTensor), emission.type(torch.FloatTensor) , init.type(torch.FloatTensor) , observations, lengths=lengths) # CxC, VxC, C, bxN -> b x (N-1) x C x C 
        labels = HMM.struct.to_parts(label.transpose(0, 1) \
                         .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) 
        # print(HMM.struct.from_parts(dist.argmax)[0][0])

        # print('label', label.transpose(0, 1)[0])  
        # show_chain(dist.argmax[0])  
        # plt.show()

        loss = dist.log_prob(labels).sum()
        losses.append(loss.detach())
        print(torch.tensor(losses).mean())

    def test(iters):
        losses = []
        total = 0
        incorrect_edges = 0 
        #model.eval()
        for i, ex in enumerate(iters):   
            observations = torch.LongTensor(ex.word).transpose(0, 1).contiguous()            
            label, lengths = ex.pos

            dist = HMM(transition, emission, init, observations, lengths=lengths) # CxC, VxC, C, bxN -> b x (N-1) x C x C 
            labels = HMM.struct.to_parts(label.transpose(0, 1) \
                        .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) 
            # print('label', label.transpose(0, 1)[0])  
            # show_chain(dist.argmax[0])  
            # plt.show()
            
            loss = dist.log_prob(labels).sum()
            losses.append(loss.detach())
            incorrect_edges += (dist.argmax.sum(-1) - labels.sum(-1)).abs().sum() / 2.0
            total += labels.sum()        

        print(torch.tensor(losses).mean())        
        print(total, incorrect_edges)   
        #model.train()
        return incorrect_edges / total     
    #print(losses, len(losses))
    test(test_iter)

trn(train_iter) 



