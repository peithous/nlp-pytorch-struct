from torch.utils.tensorboard import SummaryWriter
import torchtext.data as data
from torchtext.data import BucketIterator
import torch
from torch_struct import HMM
import matplotlib.pyplot as plt
from torch_struct.data import ConllXDatasetPOS

WORD = data.Field(pad_token=None, eos_token='<eos>') #init_token='<bos>', 
POS = data.Field(include_lengths=True, pad_token=None, eos_token='<eos>') 

fields = (('word', WORD), ('pos', POS), (None, None))
train = ConllXDatasetPOS('data/wsj.train.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 50) #en_ewt-ud-train.conllu
test = ConllXDatasetPOS('data/wsj.test.conllx', fields)
print('total train sentences', len(train))
print('total test sentences', len(test))

WORD.build_vocab(train) # min_freq = 5
POS.build_vocab(train)
train_iter = BucketIterator(train, batch_size=50, device='cpu', shuffle=False)
test_iter = BucketIterator(test, batch_size=50, device='cpu', shuffle=False)

C = len(POS.vocab)
V = len(WORD.vocab)

# counts for mle's 
tags = [] # prior
bigrams = [] # transition
word_tag_counts = [] # emission
for ex in train_iter:
#    print(ex.pos)
    words = ex.word
    label, lengths = ex.pos
    for batch in range(label.shape[1]):
    #    print(' '.join([WORD.vocab.itos[i] for i in words[:lengths[batch], batch]]))        
        tags.append(label[:lengths[batch], batch])
        bigrams.append(label[:lengths[batch], batch].unfold(0, 2, 1)) #dimension, size, step      
        for i, t in enumerate(label[:lengths[batch], batch]):
            word_tag_counts.append(torch.tensor((t.item(), words[i, batch].item())))
tags = torch.cat(tags, 0)
bigrams = torch.cat(bigrams, 0)
word_tag_counts = torch.stack(word_tag_counts)

# prior
init = torch.ones(C).long() # add-1 smoothing
init.index_put_((tags,), torch.tensor(1), accumulate=True)
assert init[POS.vocab.stoi['<eos>']] == len(train)+1
init = init.float() / init.sum()
assert torch.isclose(init.sum(), torch.tensor(1.))# \sum_C p_c = 1
init = init.log()

# transition
transition = torch.ones((C, C)).long() # p(. | eos) = 1/C
transition.index_put_((bigrams[:, 0], bigrams[:, 1]), torch.tensor(1), accumulate=True)
transition = (transition.float() / transition.sum(-1, keepdim=True)).transpose(0, 1) 
assert torch.isclose(transition.sum(0, keepdim=True).sum(), \
        torch.tensor(C, dtype=torch.float)) # should be for x in C-{eos}, sum_C  p(c, x) = 1?
transition = transition.log()

# emission 
emission = torch.ones((C, V)).long()
emission.index_put_((word_tag_counts[:, 0], word_tag_counts[:, 1]), torch.tensor(1), accumulate=True)
emission = (emission.float() / emission.sum(-1, keepdim=True)).transpose(0, 1)
assert torch.isclose(emission.sum(0, keepdim=True).sum(), \
        torch.tensor(C, dtype=torch.float)) # sum_V p(v | c) = 1
emission = emission.log()

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

def test(iters):
    losses = []
    total = 0
    incorrect_edges = 0 
    #model.eval()
    for i, ex in enumerate(iters):   
        
        observations = torch.LongTensor(ex.word).transpose(0, 1).contiguous()            
        label, lengths = ex.pos

        dist = HMM(transition, emission, init, observations, lengths=lengths) 
        labels = HMM.struct.to_parts(label.transpose(0, 1) \
                .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor)    
        # print(HMM.struct.from_parts(dist.argmax)[0][0])
        # print('label', label.transpose(0, 1)[0])  
        # show_chain(dist.argmax[0])  
        # plt.show()

        loglik = dist.log_prob(labels).sum()
        losses.append(loglik.detach()/label.shape[1])

        incorrect_edges += (dist.argmax.sum(-1) - labels.sum(-1)).abs().sum() / 2.0
        total += labels.sum()        

    print('inaccurate', incorrect_edges / total) 
    return torch.tensor(losses).mean()

print('train-log-lik', test(train_iter))
print('test-log-lik', test(test_iter))



