import torchtext.data as data
from torchtext.data import BucketIterator

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torch_struct import LinearChainCRF
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

#to do: add bos
WORD = data.Field() # init_token='<bos>', eos_token='<eos>'
POS = data.Field(is_target=True)
fields = (('word', WORD), ('pos', POS))

train = ConllXDataset('sam.conllu', fields)

WORD.build_vocab(train)
POS.build_vocab(train)
#print(vars(POS.vocab))

train_iter = BucketIterator(train, batch_size=2, device='cpu', shuffle=False)

C = len(POS.vocab.itos)
V = len(WORD.vocab.itos)

class Model(nn.Module):
    def __init__(self, voc_size, num_pos_tags):
        super().__init__()
        self.linear = nn.Linear(voc_size, num_pos_tags)

    def forward(self, count_mat):
        return F.log_softmax(self.linear(count_mat), dim=-1)     

model = Model(V, C)
opt = optim.SGD(model.parameters(), lr=0.001)

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

#def make_count_mat(sent, poss):

def trn(train_iter):
    model.train()
    
    for epoch in range(200):
        losses = []
        for batch in train_iter:
            #model.zero_grad()
            opt.zero_grad() 

            dims_fake = torch.ones(batch.pos.shape[1], C, V)
            #model_in = make_count_mat(batch.word, batch.pos)

            probs = model(dims_fake)
            #print(weights)

            chain = probs.unsqueeze(1).expand(-1, batch.word.shape[0]-1, -1, -1)  # batch, N, C, C 

            dist = LinearChainCRF(chain) # f(y) = \prod_{n=1}^N \phi(n, y_n, y_n{-1}) 
            #print('d', dist.marginals.shape, dist.marginals)
            # print(dist.argmax.shape) 
            # show_chain(dist.argmax[0])
            # plt.show()

            label = batch.pos
            labels = LinearChainCRF.struct.to_parts(label.transpose(0, 1) \
                            .type(torch.LongTensor), C).type(torch.FloatTensor) # b x N, C -> b x (N-1) x C x C 
            #print('l', labels.shape, labels)

            loss = dist.log_prob(labels).sum() # (*sample_shape x batch_shape x event_shape*) -> (*sample_shape x batch_shape*)
            #print(loss)

            (-loss).backward()
            opt.step()
            losses.append(loss.detach())
        print(sum(losses))
            
            # losses.append(loss.detach())

trn(train_iter)
