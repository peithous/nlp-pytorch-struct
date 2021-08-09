
import torchtext.data as data
from torchtext.data import BucketIterator
import torch
from torch.distributions import Categorical
#from torch import nn
from torch_struct import HMM
#import matplotlib
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

WORD = data.Field(pad_token=None) # instead of pad: length masking for param estim 
POS = data.Field(include_lengths=True, pad_token=None) # if included in class vocab, p (z_t = pad| z_t-1) > 0; init params instead of <bos> init_tok
fields = (('word', WORD), ('pos', POS), (None, None))

train = ConllXDataset('samIam.conllu', fields)
train_DATA = ConllXDataset('samIam-data-copies.conllu', fields)
test = ConllXDataset('test.conllu', fields)

WORD.build_vocab(train_DATA) # include 'was' in vocab with <unk> as its POS
POS.build_vocab(train_DATA) 
print(POS.vocab.stoi)

train_iter = BucketIterator(train, batch_size=2, device='cpu', shuffle=False)
train_iter_DATA = BucketIterator(train_DATA, batch_size=2, device='cpu', shuffle=False)

test_iter = BucketIterator(test, batch_size=2, device='cpu', shuffle=False)

C = len(POS.vocab)
V = len(WORD.vocab)
print('C', C, 'V', V)

# class Model(nn.Module):
class Model():
    def __init__(self):
        super().__init__()
        self.trnsn_prms = {}
        self.emssn_prms = {}

    def update_a(self, tag_sq, length):
        for i, x in enumerate(tag_sq[:length]): 
            # print(POS.vocab.itos[x.item()])
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
        # for x in range(words.shape[1]):
            # print(' '.join([WORD.vocab.itos[i] for i in words[:, x]]))
            # print(' '.join([POS.vocab.itos[i] for i in label[:, x]]))

        for b in range(label.shape[1]):
            model.update_a(label[:, b], lengths[b])
            model.update_b(label[:, b], words[:, b], lengths[b])
    # print(model.trnsn_prms)
    #print([(POS.vocab.itos[x[0]], POS.vocab.itos[x[1]], model.trnsn_prms[x]) for x in model.trnsn_prms])
    #print([(POS.vocab.itos[x[0]], WORD.vocab.itos[x[1]], model.emssn_prms[x]) for x in model.emssn_prms])

    transition = torch.zeros((C, C)) 
    for x in model.trnsn_prms:
        transition[x[0], x[1]] = model.trnsn_prms[x] # populate with counts: (pos_n-1, pos_n)
    #print(transition)
    for row in range(transition.shape[0]):
        if row!=POS.vocab.stoi['PUNCT']: # 0 probs at p(z_n | z_n-1 = punct)
            transition[row, :] = Categorical(transition[row, :]).probs # normalize counts, don't use logits, since 0 prob and they will be re-normalized in the lin-chain crf hmm
    #print(transition.shape, '\n', transition)
    transition = transition.transpose(0, 1) # p(z_n| z_n-1) 
    #print('transition', '\n', transition)

    init = torch.zeros(C)
    for x in range(C):
        init[x] = POS.vocab.freqs[POS.vocab.itos[x]]
    #print(init)
    init = Categorical(init).probs
    #print(init)
   
    emission = torch.zeros((C, V)) 
    for x in model.emssn_prms:  
        emission[x[0], x[1]] = model.emssn_prms[x]
    #print(emission)
    for row in range(emission.shape[0]):
        emission[row, :] = Categorical(emission[row, :]).probs # p(w_i = ·|PUNCT) = 1, (Eisenstein: 148)
    #print(emission)
    emission = emission.transpose(0,1) # p(x_n| z_n)
    #print('emission', '\n', emission)

    for ex in train_iter:
        label, lengths = ex.pos
        observations = torch.LongTensor(ex.word).transpose(0, 1).contiguous()

        dist = HMM(transition, emission, init, observations, lengths=lengths) # CxC, VxC, C, bxN -> b x (N-1) x C x C 
        # print('train')

        # print('label train', label.transpose(0, 1)[0])

        # print('marginals', dist.marginals[0].sum(-1))
        # print('argmax', dist.argmax[0].sum(-1))
        # print(dist.argmax.shape) # 
        
        # show_chain(dist.argmax[0])
        # plt.show()

    losses = []
    total = 0
    incorrect_edges = 0 
    for i, ex in enumerate(test_iter):
        label, lengths = ex.pos
        #print(label.shape)
        observations = torch.LongTensor(ex.word).transpose(0, 1).contiguous()

        #print('words', observations[0])
        print('label test', i, label.transpose(0, 1))

        dist = HMM(transition, emission, init, observations, lengths=lengths) # CxC, VxC, C, bxN
        #print(dist.log_potentials[0].sum(-1))   
        print('argmax', dist.argmax.sum(-1))

        labels = HMM.struct.to_parts(label.transpose(0, 1) \
                    .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) 

        print('label', labels.sum(-1))

        print(1, (dist.argmax.sum(-1) - labels.sum(-1)).abs())

        loss = dist.log_prob(labels).sum()
        losses.append(loss.detach())
        #print(loss)
        incorrect_edges += (dist.argmax.sum(-1) - labels.sum(-1)).abs().sum() / 2.0
        total += labels.sum()       
        print(torch.tensor(losses).mean())        
        print(total, incorrect_edges)   

        # print(dist.marginals[0].sum(-1))     
        # print(dist.argmax[0].sum(-1))        

        # show_chain(dist.argmax[0])
        # plt.show()

model = Model()
trn(train_iter, model)
# UNK PUNCT is observed once; p(w=unk| UNK) = 1
# probs: p('was'| UNK/AUX) = 0 -> max p(z| Sam I was.) = PRON AUX PUNCT
# probs: max p(z| unk unk unk unk.) = UNK UNK UNK PUNCT

print('data copies') # ie repeat PRON AUX PUNCT
model1 = Model()
trn(train_iter_DATA, model1)
# UNK PUNCT is observed twice; p(w=unk| UNK) = 1
# probs: p('was'| UNK) = 0.3, p('was'| AUX) = 0 -> max p(z| Sam I was.) = PRON AUX PUNCT
# probs: p(z| unk unk unk unk.) =  UNK PRON AUX PUNCT 

# https://homes.cs.washington.edu/~nasmith/slides/LXMLS-6-16-18.pdf
# https://www.cs.cmu.edu/~nasmith/papers/smith.tut04a.pdf 