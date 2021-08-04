from torch.utils.tensorboard import SummaryWriter
import torchtext
import torch
import torch.nn as nn
from torch_struct import DependencyCRF
import torch_struct.data
import torchtext.data as data
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import *
import matplotlib.pyplot as plt

writer = SummaryWriter(log_dir="bert-depcrf")


config = {"bert": "bert-base-cased", "H" : 768, "dropout": 0.2
         }

model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, config['bert']
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
WORD = torch_struct.data.SubTokenizedField(tokenizer)

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

train = torch_struct.data.ConllXDataset("test0.conllx", (('word', WORD), ('head', HEAD)),
                    ) #filter_pred=lambda x: 5 < len(x.word[0]) < 40
val = torch_struct.data.ConllXDataset("wsj.train0.conllx", (('word', WORD), ('head', HEAD)),
                    ) # filter_pred=lambda x: 5 < len(x.word[0]) < 40

train_iter = torchtext.data.BucketIterator(train, batch_size=20, device="cpu", shuffle=False)
#train_iter = torch_struct.data.TokenBucket(train, batch_size=10, device="cpu")
val_iter = torchtext.data.BucketIterator(val, batch_size=20, device="cpu", shuffle=False)

print(len(val))

H = config["H"]
class Model(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.base_model = model_class.from_pretrained(pretrained_weights)
        self.linear = nn.Linear(H, H)
        self.bilinear = nn.Linear(H, H)
        self.root = nn.Parameter(torch.rand(H))
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, words, mapper):
        out = self.dropout(self.base_model(words)[0])
        out = torch.einsum("bca,bch->bah", mapper.float(), out) #.cuda()
        final2 = torch.einsum("bnh,hg->bng", out, self.linear.weight)
        final = torch.einsum("bnh,hg,bmg->bnm", out, self.bilinear.weight, final2)
        root_score = torch.einsum("bnh,h->bn", out, self.root)
        final = final[:, 1:-1, 1:-1]
        N = final.shape[1]
        final[:, torch.arange(N), torch.arange(N)] += root_score[:, 1:-1]
        return final

model = Model(H)
#wandb.watch(model)
#model.cuda()

def show_deps(tree):
    plt.imshow(tree.detach())
    
def validate(val_iter):
    incorrect_edges = 0
    total_edges = 0
    model.eval()
    for i, ex in enumerate(val_iter):
        words, mapper, _ = ex.word
        label, lengths = ex.head
        batch, _ = label.shape

        final = model(words, mapper) #.cuda()
        for b in range(batch):
            final[b, lengths[b]-1:, :] = 0
            final[b, :, lengths[b]-1:] = 0
        dist = DependencyCRF(final, lengths=lengths)
        argmax = dist.argmax
        gold = dist.struct.to_parts(label, lengths=lengths).type_as(argmax)
        incorrect_edges += (argmax[:, :].cpu() - gold[:, :].cpu()).abs().sum() / 2.0
        total_edges += gold.sum()  

        gold1 = DependencyCRF(gold, lengths=lengths)

    print(total_edges, incorrect_edges)   
    model.train()
    return incorrect_edges #/total 

def train(train_iter, val_iter, model):
    opt = AdamW(model.parameters(), lr=1e-4, eps=1e-8)
    scheduler = WarmupLinearSchedule(opt, warmup_steps=20, t_total=2500)
    for epoch in range(100):
        #print(epoch)
        model.train()
        losses = []

        for i, ex in enumerate(train_iter):
            opt.zero_grad()
            words, mapper, _ = ex.word
            #print(words.shape, mapper.shape)
            label, lengths = ex.head
            batch, _ = label.shape

            # Model
            final = model(words, mapper) #.cuda()
            #print('final', final.shape)
    #         for b in range(batch):
    #             final[b, lengths[b]-1:, :] = 0
    #             final[b, :, lengths[b]-1:] = 0

    #       if not lengths.max() == final.shape[1]:
            if not lengths.max() <= final.shape[1] + 1:
                print("fail")
                continue

            dist = DependencyCRF(final, lengths=lengths)
            #print('dist', dist.argmax.shape)

            labels = dist.struct.to_parts(label, lengths=lengths).type_as(final)
            #print('labels', labels.shape)
            log_prob = dist.log_prob(labels)

            loss = log_prob.sum()
            (-loss).backward()
            writer.add_scalar('loss', -loss, epoch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            scheduler.step()
            losses.append(loss.detach())
        #print(-torch.tensor(losses).mean())

        if epoch % 10 == 1:            
            print(epoch, -torch.tensor(losses).mean(), words.shape)
            losses = []
#            show_deps(dist.argmax[0])
#            plt.show()
 
            incorrect_edges = validate(val_iter)  
            writer.add_scalar('incorrect_edges', incorrect_edges, epoch)      
                  
#            show_deps(gold.argmax[0])
#            plt.show()

train(train_iter, val_iter, model)