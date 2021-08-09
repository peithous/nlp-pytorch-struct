import torchtext.data as data
from torchtext.data import BucketIterator
import torch
from torch.distributions import Categorical
from torch_struct import HMM
import matplotlib.pyplot as plt

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

WORD = data.Field(pad_token=None)
POS = data.Field(include_lengths=True, pad_token=None) 
fields = (('word', WORD), ('pos', POS), (None, None))

train = ConllXDataset('samIam.conllu', fields)

WORD.build_vocab(train) 
POS.build_vocab(train)
print(POS.vocab.stoi)

train_iter = BucketIterator(train, batch_size=2, device='cpu', shuffle=False)

C = len(POS.vocab)
V = len(WORD.vocab)
print('C', C, 'V', V)

class Model():
    def __init__(self):
        super().__init__()
        self.trnsn_prms = {}
        self.emssn_prms = {}

    def update_a(self, tag_sq, length):
        for i, x in enumerate(tag_sq[:length]): 
            if i!=0: 
                if (tag_sq[i-1].item(), x.item()) not in self.trnsn_prms:
                    self.trnsn_prms[(tag_sq[i-1].item(), x.item())] = 1
                elif (tag_sq[i-1].item(), x.item()) in self.trnsn_prms:
                    self.trnsn_prms[(tag_sq[i-1].item(), x.item())] += 1

    def update_b(self, tag_sq, sent, length):
        for i, t in enumerate(tag_sq[:length]): 
            x = (t.item(), sent[i].item())
            if x not in self.emssn_prms:
                self.emssn_prms[x] = 1
            elif x in self.emssn_prms:
                self.emssn_prms[x] += 1

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

def trn(train_iter, model):
    for ex in train_iter:
        words = ex.word
        label, lengths = ex.pos
        for x in range(words.shape[1]):
            print(' '.join([WORD.vocab.itos[i] for i in words[:, x]]))
            print(' '.join([POS.vocab.itos[i] for i in label[:, x]]))

        for b in range(label.shape[1]):
            model.update_a(label[:, b], lengths[b])
            model.update_b(label[:, b], words[:, b], lengths[b])

    transition = torch.zeros((C, C)) 
    for x in model.trnsn_prms:
        transition[x[0], x[1]] = model.trnsn_prms[x] # counts (pos_n-1, pos_n)
    for row in range(transition.shape[0]):
        if row!=POS.vocab.stoi['PUNCT']: # p(z_n | z_n-1 = punct) = 0 
            transition[row, :] = Categorical(transition[row, :]).probs # normalize counts
    transition = transition.transpose(0, 1) # p(z_n| z_n-1) 

    init = torch.zeros(C)
    for x in range(C):
        init[x] = POS.vocab.freqs[POS.vocab.itos[x]]
    init = Categorical(init).probs
   
    emission = torch.zeros((C, V)) 
    for x in model.emssn_prms:  
        emission[x[0], x[1]] = model.emssn_prms[x]
    for row in range(emission.shape[0]):
        emission[row, :] = Categorical(emission[row, :]).probs # p(w_i = ·|PUNCT) = 1, (Eisenstein: 148)
    emission = emission.transpose(0,1) # p(x_n| z_n)
   
    losses = []
    for ex in train_iter:
        
        label, lengths = ex.pos
        observations = torch.LongTensor(ex.word).transpose(0, 1).contiguous()
        dist = HMM(transition, emission, init, observations, lengths=lengths) # CxC, VxC, C, bxN -> b x (N-1) x C x C 
        print('label', label.transpose(0, 1)[0])  
        show_chain(dist.argmax[0])  
        plt.show()

        labels = HMM.struct.to_parts(label.transpose(0, 1) \
                         .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) 
        print(1, dist.log_prob(labels))
        print(2, dist.log_prob(dist.argmax))
        
        loss = dist.log_prob(labels).sum()
        losses.append(loss.detach())
        print(torch.tensor(losses).mean())


model = Model()
trn(train_iter, model)

