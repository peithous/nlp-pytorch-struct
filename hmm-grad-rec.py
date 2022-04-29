import time
from torch.utils.tensorboard import SummaryWriter
import torchtext.data as data
from torchtext.data import BucketIterator
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch_struct import HMM, LinearChainCRF
import matplotlib.pyplot as plt
from torch_struct.data import ConllXDatasetPOS

start_time = time.time()
device='cpu'

WORD = data.Field(init_token='<bos>', pad_token=None, eos_token='<eos>') #init_token='<bos>', 
POS = data.Field(init_token='<bos>', include_lengths=True, pad_token=None, eos_token='<eos>') 

fields = (('word', WORD), ('pos', POS), (None, None))
train = ConllXDatasetPOS('data/wsj.train0.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 50) #en_ewt-ud-train.conllu
test = ConllXDatasetPOS('data/wsj.test0.conllx', fields)
print('total train sentences', len(train))
print('total test sentences', len(test))

WORD.build_vocab(train,  min_freq = 5,) #  min_freq = 5,
POS.build_vocab(train, min_freq = 5, max_size=7)
train_iter = BucketIterator(train, batch_size=20, device=device, shuffle=False)
test_iter = BucketIterator(test, batch_size=20, device=device, shuffle=False)

C = len(POS.vocab)
V = len(WORD.vocab)
# print(C, V)

class Model(nn.Module):
    def __init__(self, voc_size, num_pos_tags):
        super().__init__()
        self.emission = nn.Linear(voc_size, num_pos_tags, bias=False) 
        self.transition = nn.Linear(num_pos_tags, num_pos_tags, bias=False)
        self.init = nn.Linear(num_pos_tags, 1, bias=False)
        self.rec_emission = nn.Linear(voc_size, num_pos_tags) 
        
    def forward(self):
        transition_probs = F.log_softmax(self.transition.weight, dim=0)
        emission_probs = F.log_softmax(self.emission.weight.transpose(0,1), dim=0)
        init_probs = F.log_softmax(self.init.weight.transpose(0,1), dim=0)
        rec_emission_probs = F.log_softmax(self.rec_emission.weight.transpose(0,1), 0)

        return emission_probs, transition_probs, init_probs,\
                rec_emission_probs

model = Model(V, C)

def validate(iter):
    losses = []
    incorrect_edges = 0
    total = 0 
    model.eval()
    for i, batch in enumerate(iter):
        observations = torch.LongTensor(batch.word).transpose(0, 1).contiguous()            
        label, lengths = batch.pos
        
        emission, transition, init, rec_emission = model.forward()

        dist = HMM(transition, emission, init, observations, lengths=lengths) 

        batch, N = observations.shape
        rec_obs = rec_emission[observations.view(batch*N), :]
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
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=3.0,) # weight_decay=0.1
    losses = []
    test_acc = []
    
    for epoch in range(102):
        model.train()
        epoch_loss = []
        for i, ex in enumerate(train_iter):
            opt.zero_grad()      
            # sents = ex.word.transpose(0,1)
            observations = torch.LongTensor(ex.word).transpose(0,1).contiguous()            
            label, lengths = ex.pos
           
            # scores = model(sents)
            emission, transition, init, rec_emission = model.forward()
            dist = HMM(transition, emission, init, observations, lengths=lengths) 
            
# direct max of log marginal lik 
            z = dist.partition

            batch, N = observations.shape
            rec_obs = rec_emission[observations.view(batch*N), :]
            u_scores = dist.log_potentials + rec_obs.view(batch, N, C, 1)[:, 1:]
            u_scores[:, 0, :, :] +=  rec_obs.view(batch, N, 1, C)[:, 0]
            u = LinearChainCRF(u_scores, lengths=lengths).partition
            loss = -u + z

            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # writer.add_scalar('loss', -loss, epoch)
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


