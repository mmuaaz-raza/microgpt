from pathlib import Path
import numpy as np
import json
import os
import torch 
import torch.nn as nn
from torch.nn import functional as F
from process_data import load_essentials,get_target_labels

block_size = 64
batch_size = 32
epsilon = 1e-7
beta1 = 0.9
beta2 = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ed = 128
heads = 4
qk_d = 32
v_d = 32
layers = 4
ffn_wd = 4 * ed
dropout= 0.3
lr = 1e-3
rng = np.random.default_rng(57)

FILE_NAME = input("model/history file name : ")
MODEL_FILE = "models/"+FILE_NAME+".pth"
LOSS_HISTORY_FILE = "history/"+FILE_NAME+".json"
historyDir = Path("history")
samplesDir = Path("samples")
historyDir.mkdir(parents=True, exist_ok=True)
samplesDir.mkdir(parents=True, exist_ok=True)


trainset,testset,itos,stoi,encode,decode = load_essentials()
vocab_size = len(itos)
class Transformer(nn.Module):
    positionalEncodings: torch.Tensor
    
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,ed)
        self.register_buffer("positionalEncodings",torch.from_numpy(self.positional_encoding()).float())
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(heads,qk_d) for _ in range(layers)],
                                     nn.LayerNorm(ed,eps=epsilon))
        self.Wu = nn.Linear(ed,vocab_size,bias=False)
        self._init_weights()
    
    def positional_encoding(self) -> np.ndarray:
            
            posEncodings   = np.zeros((block_size, ed))
            divisor   = 10000 ** (np.arange(0, ed, 2) / ed)
            positions = np.arange(block_size)[:, np.newaxis]
    
            posEncodings[:, 0::2] = np.sin(positions / divisor)
            posEncodings[:, 1::2] = np.cos(positions / divisor)
            return posEncodings
    
    def _init_weights(self):
        nrm = nn.init.normal_

        def zero_bias(layer):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nrm(self.embeddings.weight, 0.0, ed ** -0.5)
        nrm(self.Wu.weight, 0.0, ed ** -0.5)
        zero_bias(self.Wu)

        for block in self.blocks:
            if not isinstance(block, Block):
                continue  # skip the final LayerNorm
            att = block.attention
            for h in att.heads:
                for lin in (h.Wq, h.Wk, h.Wv):
                    nrm(lin.weight, 0.0, (ed / heads) ** -0.5)  # type: ignore
                    zero_bias(lin)
            nrm(att.Wp.weight, 0.0, (heads * v_d) ** -0.5)
            zero_bias(att.Wp)
            up, down = block.ffwd.net[0], block.ffwd.net[2]
            nrm(up.weight, 0.0, (2 / ed) ** 0.5)  # type: ignore
            nrm(down.weight, 0.0, (2 / ffn_wd) ** 0.5)  # type: ignore
            zero_bias(up)
            zero_bias(down)
    def forward(self,input,target=None):
        if isinstance(input,np.ndarray):
            input = torch.from_numpy(input).to(device)
        B,T  = input.shape 
        loss = None

        Xp = self.embeddings(input) + self.positionalEncodings[:T]
        Xp = self.dropout(Xp)
        Xfn = self.blocks(Xp)
        logits = Xv = self.Wu(Xfn)

        B,T ,C = logits.shape

        if target is not None:
            logits = logits.reshape(-1, C)
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target).to(device).long()
            target = target.reshape(-1)
            loss = F.cross_entropy(logits,target)
        

        return logits,loss

    def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Block(nn.Module):
    def __init__(self,num_heads,heads_dim):
        super().__init__()
        heads_dim = heads_dim//num_heads
        self.attention = MultiHeadAttention(head_dim=heads_dim,head_num=num_heads)
        self.ffwd = FFN()
        self.ln1 = nn.LayerNorm(ed,eps=epsilon)
        self.ln2 = nn.LayerNorm(ed,eps=epsilon)
    def forward(self,input):
        x = input+self.attention(self.ln1(input))
        x = x+self.ffwd(self.ln2(x))
        return x

        

class MultiHeadAttention(nn.Module):
    def __init__(self,head_dim,head_num):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_dim) for _ in range(head_num)])
        self.Wp = nn.Linear(head_num*v_d,ed,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,input):
        combined_att = torch.cat([h(input) for h in self.heads],dim=-1)
        return self.dropout(self.Wp(combined_att))
    

class Head(nn.Module):
    def __init__(self,qk_d) -> None:
        super().__init__()
        self.Wq = nn.Linear(ed, qk_d,bias=False)
        self.Wk = nn.Linear(ed, qk_d,bias=False)
        self.Wv = nn.Linear(ed, v_d,bias=False)

    def forward(self,input):
        B,T,C = input.shape
        Q = self.Wq(input)
        K = self.Wk(input)
        V = self.Wv(input)

        mask = torch.triu(torch.full((T, T), float('-inf'), device=input.device), diagonal=1)


        QK = (Q @ K.transpose(-2,-1)) / (Q.shape[-1]**0.5)
        QK += mask
        att = nn.functional.softmax(QK,dim=-1)
        attv= att @ V
        return attv


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(ed, ffn_wd, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(ffn_wd,ed, bias=True),
        nn.Dropout(dropout)
        )
    def forward(self,input):
        return self.net(input)

@torch.no_grad()
def estimate_loss(microgpt:Transformer):
    out = {}
    microgpt.eval() # toggling to evaluation
    for split in ['train', 'val']:
        losses = torch.zeros(20)
        for k in range(20):
            X,Y = get_target_labels(block_size, trainset if split =="train" else testset, batch_size,rng)
            logits, loss = microgpt(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().tolist()
    microgpt.train() # toggle to training mode again , in which some config will be changes such as dropout etc
    return out


microgpt = Transformer()
microgpt = microgpt.to(device)
optimizer = torch.optim.Adam(microgpt.parameters(),lr=lr)

epoch = 1
if os.path.exists(MODEL_FILE):
    checkpoint = torch.load(MODEL_FILE)
    microgpt.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = int(checkpoint['epoch'])

loss_history = []
if os.path.exists(LOSS_HISTORY_FILE):
    with open(LOSS_HISTORY_FILE, "r") as f:
        loss_history = json.load(f)


context = torch.zeros((1, 1), dtype=torch.long, device=device)


def train_network(
    number_iter: int,
    decay_rate: float,
    checkpoint_interval: float,
    tracking_interval: float,
    base_alpha: float,
    warmup_steps: int = 500,
    sample_interval: float = 0,
    start_iter:int = 1
) -> None:

    for step in range(start_iter, number_iter+1 ):
        if step < warmup_steps:
            alpha = base_alpha * (step + 1) / warmup_steps
        else:
            alpha = base_alpha / (1.0 + decay_rate * (step - warmup_steps))

        for param_group in optimizer.param_groups:
            param_group['lr'] = alpha
        
        x,y = get_target_labels(block_size,trainset,batch_size,rng)
        _ , loss = microgpt(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
     

        if step % tracking_interval == 0:
            losses = estimate_loss(microgpt)

            loss_history.append((step, losses["train"], losses["val"]))
            print(
                f"step {step}/{number_iter}  "
                f"lr={alpha:.6f}  "
                f"train_loss={losses["train"]:.4f}  "
                f"test_loss={losses["val"]:.4f}"
            )
        if step == number_iter or step % checkpoint_interval == 0:
            torch.save({
                'epoch': step,
                'model_state_dict': microgpt.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_FILE)
          
            with open(f"{LOSS_HISTORY_FILE}", "w") as f:
                json.dump(loss_history, f)

        if step % sample_interval == 0:
            
            with open(f"samples/{FILE_NAME}{step}.txt","w") as f:
                f.write(decode(microgpt.generate(context,max_new_tokens=250)[0].tolist()))


train_network(
    number_iter=5000,
    decay_rate=1e-3,
    checkpoint_interval=50,
    tracking_interval=100,
    base_alpha=lr,
    warmup_steps=500,
    sample_interval=1000,
    start_iter=epoch
)
