import numpy as np 
from process_data import load_essentials , get_target_labels

block_size = 8 
batch_size = 32
heads = 4
embedding_d,query_wei_d,value_wei_d = 16 , int(32/heads) ,int(64/heads)
train_set,test_set ,itos,stoi,encode , decode = load_essentials()
x,y = get_target_labels(block_size,train_set,block_size)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

# embedding table of the model (vocabulary size, embedding dimension)
token_embedding_table = np.random.randn(len(itos),embedding_d) * ( 1 / (embedding_d)**0.5)

# coverting tokens to embedding vectors
input = np.stack([token_embedding_table[xi] for xi in x[0]]) # dim(input) = number of tokens in input * embedding_dimension
attentions_blocks = []
rng = np.random.default_rng()
Wo = rng.standard_normal((value_wei_d*heads, embedding_d)) * (1 / value_wei_d**0.5)
Wq = rng.standard_normal((heads,embedding_d, query_wei_d)) * (1 / embedding_d**0.5)
Wk = rng.standard_normal((heads,embedding_d, query_wei_d)) * (1 / embedding_d**0.5)
Wv = rng.standard_normal((heads,embedding_d, value_wei_d)) * (1 / embedding_d**0.5)

for i in range(heads) :
    Q = input @ Wq[i]
    K = input @ Wk[i]
    V = input @ Wv[i]

    Qk = Q @ K.T
    Qk /= (query_wei_d)**0.5
    mask = np.tril(np.ones((block_size)))
    mask[mask==0] = -np.inf 
    mask[mask==1] = 0 
    Qk += mask 

    Attention = softmax(Qk) @ V 
    attentions_blocks.append(Attention)

attentions = np.concatenate(attentions_blocks,axis=1)    
final = attentions @ Wo
print(final.shape)





