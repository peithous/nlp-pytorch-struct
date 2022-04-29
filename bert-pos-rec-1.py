import time
from torch.utils.tensorboard import SummaryWriter
import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_struct import LinearChainCRF
import torch_struct.data
import torchtext.data as data
from pytorch_transformers import *
from torch_struct.data import ConllXDatasetPOS

start_time = time.time()
#writer = SummaryWriter(log_dir="bert-pos")
config = {"bert": "bert-base-cased", "H" : 768, "dropout": 0.2}
model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, config["bert"]
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)    
# print(vars(tokenizer).keys())

WORD = torch_struct.data.SubTokenizedField(tokenizer)
UD_TAG = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", include_lengths=True)

# train, val, test = torchtext.datasets.UDPOS.splits(
#     fields=(('word', WORD), ('udtag', UD_TAG), (None, None)), 
#     filter_pred=lambda ex: len(ex.word[0]) < 200 )
fields=(('word', WORD), ('udtag', UD_TAG), (None, None))
train = ConllXDatasetPOS('data/wsj.train0.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 50) #en_ewt-ud-train.conllu
test = ConllXDatasetPOS('data/wsj.test0.conllx', fields)
UD_TAG.build_vocab(train.udtag, min_freq = 5, max_size=7)

#train_iter = torch_struct.data.TokenBucket(train, 20, device="cpu")
train_iter = torchtext.data.BucketIterator(train, batch_size=20, device="cpu", shuffle=False)
val_iter = torchtext.data.BucketIterator(test, batch_size=20, device="cpu", shuffle=False)

C = len(UD_TAG.vocab)
V = len(tokenizer.vocab)

class Model(nn.Module):
    def __init__(self, hidden, classes):
        super().__init__()
        self.base_model = model_class.from_pretrained(pretrained_weights)
        self.linear = nn.Linear(hidden, C)
        self.transition = nn.Linear(C, C)
        self.dropout = nn.Dropout(config["dropout"])
        self.rec_emission = nn.Linear(V, classes) 

    def forward(self, words, mapper):
        out = self.dropout(self.base_model(words)[0]) # N x H
        out = torch.einsum("bca,bch->bah", mapper.float(), out) #.cuda() # (N x N) (N x H) -> N x H
        final = torch.einsum("bnh,ch->bnc", out, self.linear.weight) # (N x H) (H x C) -> N x C
        batch, N, C = final.shape
        vals = final.view(batch, N, C, 1)[:, 1:N] + self.transition.weight.view(1, 1, C, C)
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0] 
        
        rec_emission_probs = F.log_softmax(self.rec_emission.weight.transpose(0,1), 0)

        return vals, rec_emission_probs

model = Model(config["H"], C)
#wandb.watch(model)
#model.cuda()

def validate(iter):
    incorrect_edges = 0
    total = 0 
    model.eval()
    for i, ex in enumerate(iter):
        words, mapper, _ = ex.word
        label, lengths = ex.udtag
        # observations = torch.LongTensor(words).transpose(0,1).contiguous() 
        batch, _, N = mapper.shape  
        
        scores, rec_emission = model(words, mapper)

        dist = LinearChainCRF(scores, lengths=lengths) 
        # argmax = dist.argmax     

        obs = torch.matmul(words.unsqueeze(dim=1), mapper).squeeze(dim=1)            

        rec_obs = rec_emission[obs.view(batch*N), :]
        u_scores = dist.log_potentials + rec_obs.view(batch, N, C, 1)[:, 1:]
        u_scores[:, 0, :, :] +=  rec_obs.view(batch, N, 1, C)[:, 0]
        u = LinearChainCRF(u_scores, lengths=lengths)

        argmax = u.argmax
        # print(argmax.shape)
        gold = LinearChainCRF.struct.to_parts(label.transpose(0, 1)\
                .type(torch.LongTensor), C, lengths=lengths).type(torch.FloatTensor) # b x N x C -> b x (N-1) x C x C  
        
        incorrect_edges += (argmax.sum(-1) - gold.sum(-1)).abs().sum()/2.0
        total += argmax.sum()  

    print(total, incorrect_edges)           
    model.train()    
    return incorrect_edges / total  
    
def train(train_iter, val_iter, model):
    opt = AdamW(model.parameters(), lr=1e-4, eps=1e-8)
    # scheduler = WarmupLinearSchedule(opt, warmup_steps=20, t_total=2500) #t_total=2500


    for epoch in range(52):
    model.train()
    losses = []

    for i, ex in enumerate(train_iter):
        opt.zero_grad()
        words, mapper, _ = ex.word
        label, lengths = ex.udtag
        # observations = torch.LongTensor(words).transpose(0,1).contiguous() 
        batch, _, N = mapper.shape

        # Model
        log_potentials, rec_emission = model(words, mapper) #.cuda()
        if not lengths.max() <= log_potentials.shape[1] + 1:
            print("fail")
            continue
        dist = LinearChainCRF(log_potentials, lengths=lengths) #lengths.cuda()   
# direct max of log marginal lik 
        z = dist.partition

        # print(words.shape, mapper.shape)

        obs = torch.matmul(words.unsqueeze(dim=1), mapper).squeeze(dim=1)            
        # print('obs', obs.shape)
        rec_obs = rec_emission[obs.view(batch*N), :]

        # print(dist.log_potentials.shape)
        # print(rec_obs.view(batch, N, C, 1)[:, 1:].shape)

        u_scores = dist.log_potentials + rec_obs.view(batch, N, C, 1)[:, 1:]            
        u_scores[:, 0, :, :] +=  rec_obs.view(batch, N, 1, C)[:, 0]
        u = LinearChainCRF(u_scores, lengths=lengths).partition            
        loss = (-u + z).sum() # nll
        loss.backward()
            # writer.add_scalar('loss', -loss, epoch)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # scheduler.step()

        losses.append(loss.detach())
       
        # if epoch % 10 == 1:            
        print(i, -torch.tensor(losses).mean(), words.shape)
        imprecision = validate(val_iter)
        print(imprecision)
            # writer.add_scalar('val_loss', val_loss, epoch)      

            #wandb.log({"train_loss":-torch.tensor(losses).mean(), 
            #           "val_loss" : val_loss})
    
train(train_iter, val_iter, model) #.cuda()

print("--- %s seconds ---" % (time.time() - start_time))
