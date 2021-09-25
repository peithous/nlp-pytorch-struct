import time
from torch.utils.tensorboard import SummaryWriter
import torchtext.data as data
from torchtext.data import BucketIterator
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch_struct import LinearChainCRF
#import matplotlib
import matplotlib.pyplot as plt
from torch_struct.data import ConllXDatasetPOS

start_time = time.time()
# writer = SummaryWriter(log_dir="lincrf")

WORD = data.Field(eos_token='<eos>', pad_token=None) # add eos bc '.' might not always be the eos 
POS = data.Field(eos_token='<eos>', pad_token=None, include_lengths=True) # init_token='<bos>',

fields = (('word', WORD), ('pos', POS), (None, None))
train = ConllXDatasetPOS('data/wsj.train0.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 10) #en_ewt-ud-train.conllu
test = ConllXDatasetPOS('data/wsj.test0.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 10)
print('total train sentences', len(train))
print('total test sentences', len(test))

WORD.build_vocab(train) 
POS.build_vocab(train) 

train_iter = BucketIterator(train, batch_size=100, device='cpu', shuffle=False)
test_iter = BucketIterator(test, batch_size=100, device='cpu', shuffle=False)

C = len(POS.vocab.itos)
V = len(WORD.vocab.itos)

class Model(nn.Module):
    def __init__(self, voc_size, num_pos_tags):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
                torch.eye(voc_size).type(torch.FloatTensor), # 1-hot 
                freeze=True) 
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
opt = optim.SGD(model.parameters(), lr=0.2)

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

def validate(iter):
    losses = []
    incorrect_edges = 0
    total = 0 
    model.eval()
    for i, batch in enumerate(test_iter):
        sents = batch.word.transpose(0,1)
        label, lengths = batch.pos
        scores = model(sents)   
        dist = LinearChainCRF(scores, lengths=lengths) 
        labels = LinearChainCRF.struct.to_parts(label.transpose(0, 1)\
                .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) # b x N x C -> b x (N-1) x C x C  
        
        # print(labels.shape)
        incorrect_edges += (dist.argmax.sum(-1) - labels.sum(-1)).abs().sum() / 2.0
        total += labels.sum()        

        loss = dist.log_prob(labels).sum()
        # print(loss)
        losses.append(loss.detach()/label.shape[1])
    
    acc = incorrect_edges / total
    print('test-loss', torch.tensor(losses).mean())   
    model.train()
    return acc

def trn(train_iter):   
    losses = []
    val_inacc = []
    
    for epoch in range(100):
        model.train()
        epoch_loss = []
        for i, ex in enumerate(train_iter):
            opt.zero_grad() 
            
            sents = ex.word.transpose(0,1)
            label, lengths = ex.pos

            scores = model(sents)

            dist = LinearChainCRF(scores, lengths=lengths) # f(y) = \prod_{n=1}^N \phi(n, y_n, y_n{-1}) 
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
            # writer.add_scalar('loss', -loss, epoch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            epoch_loss.append(loss.detach()/label.shape[1])

        losses.append(torch.tensor(epoch_loss).mean())
           
        if epoch % 10 == 1:            
            print(epoch, 'train-loss', losses[-1])
            val_acc = validate(test_iter)
            print(val_acc)
            val_inacc.append(val_acc.item())
            # print('inac', val_inacc) 

            # writer.add_scalar('val_loss', val_loss, epoch)      

    plt.plot(losses)
    plt.plot(val_inacc)

trn(train_iter)

print("--- %s seconds ---" % (time.time() - start_time))



