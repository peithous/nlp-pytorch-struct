from torch.utils.tensorboard import SummaryWriter
import torchtext.data as data
from torchtext.data import BucketIterator
import torch
from torch.distributions import Categorical
from torch_struct import HMM
import matplotlib.pyplot as plt

writer = SummaryWriter(log_dir="hmm-1hot")
class ConllXDataset(data.Dataset):
    def __init__(self, path, fields, encoding='utf-8', separator='\t', **kwargs):
        examples = []
        columns = [[], []]
        column_map = {1: 0, 3: 1}
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == '':
                    examples.append(data.Example.fromlist(columns, fields))
                    columns = [[], []]
                else:
                    for i, column in enumerate(line.split(separator)):
                        if i in column_map:
                            columns[column_map[i]].append(column)
            examples.append(data.Example.fromlist(columns, fields))
        super(ConllXDataset, self).__init__(examples, fields, **kwargs)
WORD = data.Field(pad_token=None, eos_token='<eos>') #init_token='<bos>', 
POS = data.Field(include_lengths=True, pad_token=None, eos_token='<eos>') 
fields = (('word', WORD), ('pos', POS), (None, None))
train = ConllXDataset('wsj.train0.conllx', fields)
#test = ConllXDataset('samIam-data-copies.conllu', fields)
WORD.build_vocab(train, min_freq = 5) 
POS.build_vocab(train, min_freq = 5)
train_iter = BucketIterator(train, batch_size=20, device='cpu', shuffle=False)
#test_iter = BucketIterator(test, batch_size=20, device='cpu', shuffle=False)
C = len(POS.vocab)
V = len(WORD.vocab)
print('C', C, POS.vocab.itos, '\n', 'V', V, WORD.vocab.itos)

tags = []
bigrams = []
word_tag_counts = []
for ex in train_iter:
    words = ex.word
    label, lengths = ex.pos
    for batch in range(label.shape[1]):
        bigrams.append(label[:lengths[batch], batch].unfold(0, 2, 1))
        tags.append(label[:lengths[batch], batch])
        for i, t in enumerate(label[:lengths[batch], batch]):
            word_tag_counts.append(torch.tensor((t.item(), words[i, batch].item())))
tags = torch.cat(tags, 0)
bigrams = torch.cat(bigrams, 0)
word_tag_counts = torch.stack(word_tag_counts)

init = torch.zeros(C).long() 
init.index_put_((tags,), torch.tensor(1), accumulate=True)
init = torch.div(init.float(), init.sum())
assert init.sum() == 1 # \sum_C p_c = 1

transition = torch.zeros((C, C)).long() 
transition.index_put_((bigrams[:, 0], bigrams[:, 1]), torch.tensor(1), accumulate=True)
transition = transition.type(torch.FloatTensor) 
#transition = transition / transition.sum(-1, keepdim=True) 
#print(transition.type())
for row in range(transition.shape[0]):
    if row!=POS.vocab.stoi['<eos>']:  
        transition[row, :] = torch.div(transition[row, :].float(), transition[row, :].sum()) #).transpose(0, 1)
        #transition[row, :] = Categorical(transition[row, :].float()).probs # normalize counts
transition = transition.transpose(0, 1)
assert transition.sum(0, keepdim=True).sum() == C-1 # for all x \in C-{eos}, \sum_C  p_{x,c} = 1

emission = torch.zeros((C, V)).long()
emission.index_put_((word_tag_counts[:, 0], word_tag_counts[:, 1]), torch.tensor(1), accumulate=True)
emission = torch.div(emission.float(), emission.sum(-1, keepdim=True)).transpose(0, 1)
assert emission.sum(0, keepdim=True).sum() == C # for all c \in C, \sum_V p_c (v) = 1

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

def trn(train_iter):
    losses = []
    for ex in train_iter:
        label, lengths = ex.pos
        observations = torch.LongTensor(ex.word).transpose(0, 1).contiguous()  

        dist = HMM(transition, emission, init, observations, lengths=lengths) # CxC, VxC, C, bxN -> b x (N-1) x C x C 
        labels = HMM.struct.to_parts(label.transpose(0, 1) \
                         .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) 
        #print(HMM.struct.from_parts(dist.argmax)[0][0])

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
    test(train_iter)

trn(train_iter) 



