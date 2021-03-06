import time
from torch.utils.tensorboard import SummaryWriter
# from torchtext.legacy import data
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
device='cpu'
# writer = SummaryWriter(log_dir="lincrf")

WORD = data.Field(init_token='<bos>', eos_token='<eos>', pad_token=None) 
POS = data.Field(init_token='<bos>', eos_token='<eos>', pad_token=None, include_lengths=True) 

fields = (('word', WORD), ('pos', POS), (None, None))
train = ConllXDatasetPOS('data/wsj.train0.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 50) #en_ewt-ud-train.conllu
test = ConllXDatasetPOS('data/wsj.test0.conllx', fields)

print('total train sentences', len(train))
print('total test sentences', len(test))

WORD.build_vocab(train, min_freq=10) # min_freq = 10
POS.build_vocab(train, min_freq=10, max_size=5) # min_freq = 10, max_size=7

train_iter = BucketIterator(train, batch_size=20, device=device, shuffle=False)
test_iter = BucketIterator(test, batch_size=20, device=device, shuffle =False)

C = len(POS.vocab.itos)
V = len(WORD.vocab.itos)
# print(C)

class Model(nn.Module):
    def __init__(self, voc_size, num_pos_tags):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
                torch.eye(voc_size).type(torch.FloatTensor), # 1-hot 
                freeze=True) 
        self.linear = nn.Linear(voc_size, num_pos_tags) 
        self.transition = nn.Linear(num_pos_tags, num_pos_tags)
        self.rec_emission = nn.Linear(voc_size, num_pos_tags) 

    def forward(self, words):
        out = self.embedding(words) # (b x N) -> (b x N x V)
        final = self.linear(out) # (b x N x V) (V x C) -> (b x N x C)
        # final = torch.einsum("bnh,ch->bnc", out, self.linear.weight) # (N x H) (H x C) -> N x C
        batch, N, C = final.shape
        vals = final.view(batch, N, C, 1)[:, 1:N] + self.transition.weight.view(1, 1, C, C) # -> (b x N-1 x C x C)      
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0] # 1st tag prob

        rec_emission_probs = F.log_softmax(self.rec_emission.weight.transpose(0,1), 0)

        return vals, rec_emission_probs

model = Model(V, C)

def show_chain(chain):
    plt.imshow(chain.detach().sum(-1).transpose(0, 1))

def validate(iter):
    losses = []
    incorrect_edges = 0
    total = 0 
    model.eval()
    for i, batch in enumerate(iter):
        sents = torch.LongTensor(batch.word).transpose(0, 1).contiguous() 
        label, lengths = batch.pos     
        
        scores, rec_emission = model(sents)

        dist = LinearChainCRF(scores, lengths=lengths) 
        # argmax = dist.argmax     

        batch, N = sents.shape
        rec_obs = rec_emission[sents.view(batch*N), :]
        u_scores = dist.log_potentials + rec_obs.view(batch, N, C, 1)[:, 1:]
        u_scores[:, 0, :, :] +=  rec_obs.view(batch, N, 1, C)[:, 0]
        u = LinearChainCRF(u_scores, lengths=lengths)

        argmax = u.argmax
        gold = LinearChainCRF.struct.to_parts(label.transpose(0, 1)\
                .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) # b x N x C -> b x (N-1) x C x C  
        
        incorrect_edges += (argmax.sum(-1) - gold.sum(-1)).abs().sum()/2.0
        total += argmax.sum()  

    print(total, incorrect_edges)           
    model.train()    
    return incorrect_edges / total   

def trn(train_iter):   
    # opt = optim.SGD(model.parameters(), lr=0.1)
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=5.0,  ) # weight_decay=0.1 
    
    losses = []
    test_acc = []
    for epoch in range(202):
        model.train()
        epoch_loss = []
        for i, ex in enumerate(train_iter):
            opt.zero_grad()      
            observations = torch.LongTensor(ex.word).transpose(0,1).contiguous()            

            label, lengths = ex.pos
           
            scores, rec_emission = model(observations)
            dist = LinearChainCRF(scores, lengths=lengths) # f(y) = \prod_{n=1}^N \phi(n, y_n, y_n{-1}) 
        
# direct max of log marginal lik 
            z = dist.partition
            batch, N = observations.shape

            rec_obs = rec_emission[observations.view(batch*N), :]

            u_scores = dist.log_potentials + rec_obs.view(batch, N, C, 1)[:, 1:]
            u_scores[:, 0, :, :] +=  rec_obs.view(batch, N, 1, C)[:, 0]
            u = LinearChainCRF(u_scores, lengths=lengths).partition            
            
            loss = -u + z  # -log lik 
            loss.sum().backward()
            # writer.add_scalar('loss', -loss, epoch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss.append(loss.sum().detach()/label.shape[1])

        losses.append(torch.tensor(epoch_loss).mean())


        if epoch % 10 == 1:            
            print(epoch, 'train-loss', losses[-1])
            imprecision = validate(test_iter)
            print(imprecision)
            #test_acc.append(val_acc.item())

            # print('l', label.transpose(0, 1)) #labels         
            # show_chain(dist.argmax[0])
            # plt.show()

            # writer.add_scalar('val_loss', val_loss, epoch)      

    # plt.plot(losses)
    # plt.plot(test_acc)

trn(train_iter)

print("--- %s seconds ---" % (time.time() - start_time))