import time
from torch.utils.tensorboard import SummaryWriter
import torchtext
import torch
import torch.nn as nn
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
WORD = torch_struct.data.SubTokenizedField(tokenizer)
UD_TAG = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", include_lengths=True)
# train, val, test = torchtext.datasets.UDPOS.splits(
#     fields=(('word', WORD), ('udtag', UD_TAG), (None, None)), 
#     filter_pred=lambda ex: len(ex.word[0]) < 200 )
fields=(('word', WORD), ('udtag', UD_TAG), (None, None))
train = ConllXDatasetPOS('data/wsj.train0.conllx', fields, 
                filter_pred=lambda x: len(x.word) < 50) #en_ewt-ud-train.conllu
test = ConllXDatasetPOS('data/wsj.test0.conllx', fields)
#WORD.build_vocab(train.word, min_freq=3)
UD_TAG.build_vocab(train.udtag, min_freq = 10, max_size=7)

train_iter = torch_struct.data.TokenBucket(train, 20, device="cpu")
# train_iter = torchtext.data.BucketIterator(train, batch_size=20, device="cpu", shuffle=False)
val_iter = torchtext.data.BucketIterator(test, batch_size=20, device="cpu", shuffle=False)
C = len(UD_TAG.vocab)

class Model(nn.Module):
    def __init__(self, hidden, classes):
        super().__init__()
        self.base_model = model_class.from_pretrained(pretrained_weights)
        self.linear = nn.Linear(hidden, C)
        self.transition = nn.Linear(C, C)
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, words, mapper):
        out = self.dropout(self.base_model(words)[0]) # N x H
        out = torch.einsum("bca,bch->bah", mapper.float(), out) #.cuda() # (N x N) (N x H) -> N x H
        final = torch.einsum("bnh,ch->bnc", out, self.linear.weight) # (N x H) (H x C) -> N x C
        batch, N, C = final.shape
        #print(final.view(batch, N, C, 1).shape)
        #print(final.view(batch, N, C, 1)[:, 1:N].shape)
        vals = final.view(batch, N, C, 1)[:, 1:N] + self.transition.weight.view(1, 1, C, C)
        #print(vals.shape)
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0] 
        return vals

model = Model(config["H"], C)
#wandb.watch(model)
#model.cuda()

def validate(itera):
    incorrect_edges = 0 
    total = 0 
    model.eval()
    for i, ex in enumerate(itera):
        words, mapper, _ = ex.word
        label, lengths = ex.udtag
        dist = LinearChainCRF(model(words, mapper), #.cuda()
                              lengths=lengths)        
        argmax = dist.argmax
        gold = LinearChainCRF.struct.to_parts(label.transpose(0, 1), C,
                                              lengths=lengths).type_as(argmax)
        incorrect_edges += (argmax.sum(-1) - gold.sum(-1)).abs().sum() / 2.0
        total += argmax.sum()    

    print(total, incorrect_edges)           
    model.train()    
    return incorrect_edges / total   
    
def train(train_iter, val_iter, model):
    opt = AdamW(model.parameters(), lr=1e-4, eps=1e-8) # imported with pytorch_transformers
    scheduler = WarmupLinearSchedule(opt, warmup_steps=20, t_total=150) #t_total=2500
    
    model.train()
    losses = []

    for i, ex in enumerate(train_iter):
        opt.zero_grad()
        words, mapper, _ = ex.word
        label, lengths = ex.udtag
        # N_1, batch = label.shape

        # Model
        log_potentials = model(words, mapper) #.cuda()
        # if not lengths.max() <= log_potentials.shape[1] + 1:
        #     print("fail")
        #     continue
        dist = LinearChainCRF(log_potentials,
                                lengths=lengths) #lengths.cuda()   
        # labels = LinearChainCRF.struct.to_parts(label.transpose(0, 1), C, lengths=lengths) \
        #                     .type_as(dist.log_potentials)
        # loss = dist.log_prob(labels).sum()
        # (-loss).backward()
        # writer.add_scalar('loss', -loss, epoch)

# direct max of log marginal lik 
        loss1 = dist.partition.sum()
        (loss1).backward()
    
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        losses.append(loss1.detach())
 
        # if i % 10 == 1:            
        print(i, -torch.tensor(losses).mean(), words.shape)
        imprecision = validate(val_iter)
        print(imprecision)
            # writer.add_scalar('val_loss', val_loss, epoch)      

            #wandb.log({"train_loss":-torch.tensor(losses).mean(), 
            #           "val_loss" : val_loss})
    
train(train_iter, val_iter, model) #.cuda()

print("--- %s seconds ---" % (time.time() - start_time))
