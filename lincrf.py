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

# if pad is included in class vocab, then p (z_t = pad| z_t-1) > 0 
# add eos bc '.' might not always be the eos 
WORD = data.Field(init_token='<bos>', eos_token='<eos>', pad_token=None) 
POS = data.Field(init_token='<bos>', eos_token='<eos>', pad_token=None, include_lengths=True) #


fields = (('word', WORD), ('pos', POS), (None, None))

train = ConllXDataset('test0.conllx', fields)
#train = ConllXDataset('samIam.conllu', fields)
test = ConllXDataset('wsj.train0.conllx', fields)

WORD.build_vocab(train) 
POS.build_vocab(train) 

train_iter = BucketIterator(train, batch_size=10, device='cpu', shuffle=False)
test_iter = BucketIterator(test, batch_size=10, device='cpu', shuffle=False)

C = len(POS.vocab.itos)
V = len(WORD.vocab.itos)
print(C, V)
# print(POS.vocab.stoi)
# print(WORD.vocab.stoi)

class Model(nn.Module):
    def __init__(self, voc_size, num_pos_tags):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.eye(voc_size).type(torch.FloatTensor), freeze=True) #one hot 
        self.linear = nn.Linear(voc_size, num_pos_tags) # batch x C x V -> batch x C_t x C_t-1
        self.transition = nn.Linear(num_pos_tags, num_pos_tags)
        
    def forward(self, words):
        out = self.embedding(words) # (b x N ) -> (b x N x V)
        final = self.linear(out) # (b x N x V) (V x C) -> (b x N x C)
        batch, N, C = final.shape
        vals = final.view(batch, N, C, 1)[:, 1:N] + self.transition.weight.view(1, 1, C, C)
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0] 
        return vals

model = Model(V, C)
opt = optim.SGD(model.parameters(), lr=0.01)

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

def validate(iter):
    incorrect_edges = 0
    total = 0 
    model.eval()
    for i, batch in enumerate(test_iter):
        sents = batch.word.transpose(0,1)
        label, lengths = batch.pos

        log_potentials = model(sents)
        
        dist = LinearChainCRF(log_potentials, lengths=lengths) 

        labels = LinearChainCRF.struct.to_parts(label.transpose(0, 1) \
                        .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) # b x N x C -> b x (N-1) x C x C  
        
        incorrect_edges += (dist.argmax.sum(-1) - labels.sum(-1)).abs().sum() / 2.0
        total += dist.argmax.sum()        
    
    print(total, incorrect_edges)   
    model.train()
    return incorrect_edges / total 

def trn(train_iter):   

    for epoch in range(100):
        model.train()
        losses = []
        for i, ex in enumerate(train_iter):
            opt.zero_grad() 
            
            sents = ex.word.transpose(0,1)
            label, lengths = ex.pos

            log_potentials = model(sents)
            # print(log_potentials.shape)

            dist = LinearChainCRF(log_potentials, lengths=lengths) # f(y) = \prod_{n=1}^N \phi(n, y_n, y_n{-1}) 
            #print('d', dist.marginals.shape, dist.marginals)
            #print(dist.argmax.shape) 
            #show_chain(dist.argmax[0])
            #plt.show()

            labels = LinearChainCRF.struct.to_parts(label.transpose(0, 1) \
                        .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) # b x N x C -> b x (N-1) x C x C 
            #print('l', labels.shape) #labels         
            #print(dist.log_prob(labels))

            loss = dist.log_prob(labels).sum() # (*sample_shape x batch_shape x event_shape*) -> (*sample_shape x batch_shape*)
            #print(loss)
            (-loss).backward()
            opt.step()
            losses.append(loss.detach())
           
        if epoch % 10 == 1:            
            print(epoch, -torch.tensor(losses).mean(), sents.shape)
            losses = []
            # show_deps(dist.argmax[0])

            val_loss = validate(test_iter)
            print('val', val_loss)     

            # incorrect_edges = validate(test_iter)  
            # writer.add_scalar('incorrect_edges', incorrect_edges, epoch)      
            # show_deps(gold.argmax[0])
            # plt.show()

trn(train_iter)


