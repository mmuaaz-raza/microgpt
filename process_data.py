import numpy as np 

def load_vocabulary(data):
    vocab = sorted(list(set("".join(data))))
    itos = {i:char for i,char in enumerate(vocab)}
    stoi = {char:i for i,char in enumerate(vocab)}
    return itos,stoi

def load_essentials():
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open("data2.txt") as file:
        data = file.read()
    n = int(0.9*len(list(data)))
    # 90% data ~ trainset 
    train_set = data[:]
                    #  n]
    test_set = data[n:]
    # index to string , string to index lookup tables
    
    itos,stoi = load_vocabulary(data)
  
    # index to string , string to index functions 
    encode = lambda x:[stoi[char] for char in "".join(x)]
    decode = lambda x:"".join([itos[index] for index in x])
    # encode the whole data set splits
    return np.array(encode(train_set),dtype=np.long),np.array(encode(test_set),dtype=np.long) ,itos,stoi ,encode,decode



def get_target_labels(block_size,data,batch_size):
    rng = np.random.default_rng()
    # 'batch size' number of random offset in dataset 
    indices = rng.integers(high=len(data)-block_size-1,low=0,size=batch_size)

    # stacking up all the data blocks produced by random indices
    
    x =np.stack([data[i:i+block_size] for i in indices])
    y =np.stack([data[i+1:i+block_size+1] for i in indices])
    # dim(x) = dim(y) = (batch_size=number of examples,block_size=number of tokens in a block per example)
    return x,y 



def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x_shifted)/(np.sum(np.exp(x_shifted),axis=1,keepdims=True))

def layer_norm_cal(x,epsilon):
    mean = np.mean(x,axis=1,keepdims=True)
    var = np.var(x,axis=1,keepdims=True)
    return (x-mean)/(var+epsilon)**0.5 , mean, var





def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x > 0



def gelu(x):
    inner = ((2 / np.pi) ** 0.5) * (x + 0.044715 * (x ** 3))
    return 0.5 * x * (1 + np.tanh(inner))

def dgelu(x):
    inner = ((2 / np.pi) ** 0.5) * (x + 0.044715 * (x ** 3))
    tanh_inner = np.tanh(inner)
    
    sech2_inner = 1 / (np.cosh(inner) ** 2)
    inner_derivative = ((2 / np.pi) ** 0.5) * (1 + 3 * 0.044715 * (x ** 2))
    
    term1 = 0.5 * (1 + tanh_inner)
    term2 = 0.5 * x * sech2_inner * inner_derivative
    
    return term1 + term2
