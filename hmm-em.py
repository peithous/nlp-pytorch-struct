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
train_iter = BucketIterator(train, batch_size=2000, device=device, shuffle=False)
test_iter = BucketIterator(test, batch_size=200, device=device, shuffle=False)
C = len(POS.vocab)
V = len(WORD.vocab)
# print(C, V)

class Model(nn.Module):
    def __init__(self, voc_size, num_pos_tags):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
                torch.eye(voc_size).type(torch.FloatTensor), # 1-hot 
                freeze=True) 
        self.emission = nn.Linear(voc_size, num_pos_tags, bias=False) 
        self.transition = nn.Linear(num_pos_tags, num_pos_tags, bias=False)
        self.init = nn.Linear(num_pos_tags, 1, bias=False)
        
    def forward(self):
        transition_probs = F.log_softmax(self.transition.weight, dim=0)
        emission_probs = F.log_softmax(self.emission.weight.transpose(0,1), dim=0)
        # print(emission_probs.shape)
        init_probs = F.log_softmax(self.init.weight.transpose(0,1), dim=0)

        return emission_probs, transition_probs, init_probs,\
                self.embedding.weight

model = Model(V, C)

def validate(iter, emission, transition, init):
    incorrect_edges = 0
    total = 0 
    for i, batch in enumerate(iter):
        observations = torch.LongTensor(batch.word).transpose(0,1).contiguous()            
        label, lengths = batch.pos
        
        dist = HMM(transition, emission, init, observations, lengths=lengths) 
        argmax = dist.argmax
        gold = LinearChainCRF.struct.to_parts(label.transpose(0,1)\
                .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) # b x N x C -> b x (N-1) x C x C  
        
        incorrect_edges += (argmax.sum(-1) - gold.sum(-1)).abs().sum() / 2.0
        total += argmax.sum()        

    print(total, incorrect_edges)           
    model.train()    
    return incorrect_edges / total   

def trn(iter):   
    # losses = []
    # test_acc = []
    emission, transition, init, one_hot = model.forward() # could also be tensors instead of nn.lin.weight?
    for epoch in range(100): 
        print(epoch)

        for i, ex in enumerate(iter): # actually reads in 1 batch with all train data (not a real for-loop)
            observations = torch.LongTensor(ex.word).transpose(0,1).contiguous() # b x N (b = all train data)
            _, lengths = ex.pos     
            batch, N = observations.shape
            # print(batch)

            dist = HMM(transition, emission, init, observations, lengths=lengths)  # using theta^{old}
            old = dist.partition.sum() 

            # print(old)

# E: marginals of posterior p(latent_vars|obs)
            pair_marg = dist.marginals # xi's: (b x N-1 x C x C)
            unary_marg = pair_marg.sum(dim=-1) # gamma's: sum_{j=1}^C p[z_{t-1} =j, z_t =i] 
            init_marg = pair_marg[:,0,:,:].sum(dim=-2) # gamma_1: sum_{i=1}^C  p[z_1 =j, z_t =i] 

# M: theta^{new} 
            # div prob of getting to a state at t from each state at t-1 by prob of leaving each state at t-1
            # i.e. div each transition mat row by 1 x C vec corresponding to prob of leaving each prev state
            # b x C_{t-1} x C_t /  b x 1 x C_{t-1}
            transition = pair_marg.sum(dim=-3).sum(dim=0)/pair_marg.sum(-2).sum(-2).sum(0).unsqueeze(0) #.log() # denom = sum_b sum_t^{N-1} xi_{t,j -> k}/sum_{l=1}^C sum_t^{N-1} xi_{t,j -> l}
            assert torch.isclose(transition.sum(0, keepdim=True).sum(), \
                                    torch.tensor(C, dtype=torch.float)) # should be for x in C-{eos}, sum_C  p(c, x) = 1?
            transition = transition.log()

            init = init_marg.sum(0)/init_marg.sum().sum(0) #.log()
            assert torch.isclose(init.sum(), torch.tensor(1.))# \sum_C p_c = 1
            init = init.log()

            gamma = torch.cat((init_marg.unsqueeze(1), unary_marg), dim=1) # b x N x C
            v_x_n = one_hot[observations.view(batch*N), :].view(batch, N, V).transpose(1,2)
            #  V x N * N x C -> V x C; V x C/1 x C
            emission = torch.matmul(v_x_n, gamma).sum(0)/gamma.sum(1).sum(0).unsqueeze(0) #.log()
            # print(emission[:5])
            assert torch.isclose(emission.sum(0, keepdim=True).sum(), \
                                    torch.tensor(C, dtype=torch.float)) # sum_V p(v | c) = 1
            emission = emission.log()

            dist_new = HMM(transition, emission, init, observations, lengths=lengths)             
            lik = dist_new.partition.sum() # at theta^{new} 

            diff = torch.abs(old -lik)
            print('diff', diff)

            inacc = validate(test_iter, emission, transition, init)
            print(inacc)

    return transition, emission, init, observations

    # print('l', label.transpose(0, 1)) #labels         
    # show_chain(dist.argmax[0])
    # plt.show()

    # plt.plot(losses)
    # plt.plot(test_acc)

trn(train_iter)

print("--- %s seconds ---" % (time.time() - start_time))


