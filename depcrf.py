import torchtext
import torch
import torch.nn as nn
from torch_struct import DependencyCRF
import torch_struct.data 
import torchtext.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt

def batch_num(nums):
    lengths = torch.tensor([len(n) for n in nums]).long()
    n = lengths.max()
    out = torch.zeros(len(nums), n).long()
    for b, n in enumerate(nums):
        out[b, :len(n)] = torch.tensor(n)
    return out, lengths
HEAD = data.RawField(preprocessing= lambda x: [int(i) for i in x],
                     postprocessing=batch_num)
HEAD.is_target = True

WORD = data.Field(pad_token=None)
WORD.is_target = False

train = torch_struct.data.ConllXDataset("samIam.conllu", (('word', WORD), ('head', HEAD)),
                     ) #filter_pred=lambda x: 5 < len(x.word) < 40

WORD.build_vocab(train)
train_iter = data.BucketIterator(train, batch_size=2, device='cpu', shuffle=False)

V = len(WORD.vocab.itos)


class Model(nn.Module):
    def __init__(self, hidden):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(F.one_hot(torch.arange(V)).type(torch.FloatTensor), freeze=True) #one hot 

        self.linear = nn.Linear(V, V)
        self.bilinear = nn.Linear(V, V)

        self.root = nn.Parameter(torch.rand(V))
        
    def forward(self, words):
        #out = self.dropout(self.base_model(words)[0])
        out = self.embedding(words) # (b x N ) -> (b x N x V)
        final2 = self.linear(out) # (b x N x V) (V x V) -> (b x N x V)
        final = torch.einsum("bnh,hg,bmg->bnm", out, self.bilinear.weight, final2) # (N x V) (V x V) (V x N) -> (N, N)
        #print('ein3', final.shape)
        root_score = torch.einsum("bnh,h->bn", out, self.root)
        #print('root', root_score)

        N = final.shape[1]
        final[:, torch.arange(N), torch.arange(N)] += root_score
        #print('f2', final.shape)
        return final

model = Model(V)

def show_deps(tree):
    plt.imshow(tree.detach())

def trn(train_iter, model):

    for i, ex in enumerate(train_iter):
        print(i)
        # print(ex.word.shape) # sq x b
        # print(ex.head[0].shape) # b x sq
        # print('lens', ex.head[1])
        
        words = ex.word.transpose(0,1)
        #print(words.shape, mapper.shape)
        label, lengths = ex.head
        #batch, _ = label.shape
        
        final = model(words)
        print(final)
        dist = DependencyCRF(final, lengths=lengths)
        # dist.multiroot=False
        # show_deps(dist.argmax[0])
        # plt.show()

        labels = dist.struct.to_parts(label, lengths=lengths).type_as(final)
        #print('labels', labels.shape)

        gold = DependencyCRF(labels, lengths=lengths)
        show_deps(gold.argmax[0])
        plt.show()

        log_prob = dist.log_prob(labels)
        print(log_prob.shape)

        loss = log_prob.sum()
        (-loss).backward()

trn(train_iter, model)