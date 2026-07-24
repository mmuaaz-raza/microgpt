import numpy as np 
from process_data import load_essentials , get_target_labels ,softmax , layer_norm_cal,relu, drelu

block_size = 8 
batch_size = 32
heads = 4

train_set,test_set,itos,stoi ,encode,decode = load_essentials()

class Transformer():
    def __init__(self,embedding_d,heads,qk_d,v_d,layers,ffn_w_d,nToken) -> None:
        self.params={}
        self.dimensions = {"embedding_d":embedding_d,"qk_d":int(qk_d/heads),"v_d":int(v_d/heads),"p_d":v_d,"ffn_w_d":ffn_w_d,"nToken":nToken,"layers":layers}
        self.train_set,self.test_set ,self.itos,self.stoi,self.encode , self.decode = load_essentials()
        # embedding table of the model (vocabulary size, embedding dimension)
        self.params["token_embedding_table"] = np.random.randn(len(self.itos),embedding_d) * ( 1 / (embedding_d)**0.5)
        rng = np.random.default_rng()
        self.params["attention"] = {}

        self.params["attention"]["Wp"] = rng.standard_normal((layers,self.dimensions["v_d"]*heads, embedding_d)) * (1 / self.dimensions["v_d"]**0.5)
        self.params["attention"]["epsilon"] = rng.random((layers))
        self.params["attention"]["gama"] = rng.standard_normal((layers,embedding_d))/(embedding_d**0.5)
        self.params["attention"]["beta"] = rng.standard_normal((layers,embedding_d))/(embedding_d**0.5) #dim = (layers, heads , embedding dimentsion of each token, Qk wieght dimension)
        self.params["attention"]["Wq"] = rng.standard_normal((layers,heads,embedding_d, self.dimensions["qk_d"])) * (1 / embedding_d**0.5) 
        self.params["attention"]["Wk"] = rng.standard_normal((layers,heads,embedding_d, self.dimensions["qk_d"])) * (1 / embedding_d**0.5)
        self.params["attention"]["Wv"] = rng.standard_normal((layers,heads,embedding_d, self.dimensions["v_d"])) * (1 / embedding_d**0.5)

        self.params["ffn"] = {}
        self.params["ffn"]["epsilon"] = rng.random((layers,1))
        self.params["ffn"]["gama"] = rng.standard_normal((layers,embedding_d))/(embedding_d**0.5)
        self.params["ffn"]["beta"] = rng.standard_normal((layers,embedding_d))/(embedding_d**0.5)
        self.params["ffn"]["Wi"] = rng.standard_normal((layers, embedding_d,self.dimensions["ffn_w_d"])) * (1 / self.dimensions["embedding_d"]**0.5)
        self.params["ffn"]["Wp"] = rng.standard_normal((layers, self.dimensions["ffn_w_d"],embedding_d)) * (1 / self.dimensions["ffn_w_d"]**0.5)
        self.params["final"] = {"Wu":rng.standard_normal((embedding_d,len(self.itos))) * (1 / embedding_d**0.5)}





    def init_step(self,x):
        input = np.stack([self.params["token_embedding_table"][xi] for xi in x]) # dim(input) = number of tokens in input * embedding_dimension
        # add absolute fixed positional encoding
        input = input + self.positional_encoding(input)
        return input

    def attention(self,x,layer): 
        # apply layer norm
        normx = layer_norm_cal(x,self.params["attention"]["epsilon"][layer]) * self.params["attention"]["gama"][layer] + self.params["attention"]["beta"][layer]
        attentions_blocks = []
        # run attention layer heads on norm x
        mask = np.tril(np.ones((block_size)))
        mask[mask==0] = -np.inf 
        mask[mask==1] = 0 
        for i in range(heads) :
            Q = normx @ self.params["attention"]["Wq"][layer][i]
            K = normx @ self.params["attention"]["Wk"][layer][i]
            V = normx @ self.params["attention"]["Wv"][layer][i]

            Qk = Q @ K.T
            Qk /= (self.dimensions["qk_d"])**0.5
            Qk += mask 
            Attention = softmax(Qk) @ V 
            attentions_blocks.append(Attention)

        attentions = np.concatenate(attentions_blocks,axis=-1)    
        final = attentions @ self.params["attention"]["Wp"][layer]

        # residual connection to initial input
        return x + final

    
    def positional_encoding(self,input): #input(number of token , dimension of embeddings)
            d_positions = positions = input.shape[0]
            d_model = input.shape[1]
            encoded = np.zeros((d_positions,d_model))

            divisor =  10000**(np.arange(0,d_model,2)/d_model)
            positions = np.arange(d_positions)[:,np.newaxis]
            encoded[:,0::2] = np.sin(positions/divisor)
            encoded[:,1::2] = np.cos(positions/divisor)

            return encoded 
    

    def feedforwardlayer(self,input,layer):
        normx = layer_norm_cal(input,self.params["ffn"]["epsilon"][layer]) * self.params["ffn"]["gama"][layer] + self.params["ffn"]["beta"][layer]
        Z0 = normx @ self.params["ffn"]["Wi"][layer]
        A0 = relu(Z0)
        Z1 = A0 @ self.params["ffn"]["Wp"][layer]
        return  Z1 + input
    
    def final_step (self,H):
        return softmax(H @ self.params["final"]["Wu"])
    
    def loss_calculation(self,probs,y):
        return -np.mean([np.log(probs[i,y[i]]+1e-9) for i in range(probs.shape[0])])
    
    def forward(self,x):
        self.dimensions["nToken"] = len(x)
        oFFn = self.init_step(x)
        for i in range(self.dimensions["layers"]):
            oAttention = self.attention(oFFn,i)
            oFFn = self.feedforwardlayer(oAttention,i)
        return self.final_step(oFFn)        
             
    def backward(self, probs,y,H):
        T = probs.shape[0]
        probs[np.arange(T),y] -= 1

        dLoss = probs * (1/T)  # (T,vocab_size)
        dWu = H.T @ dLoss 
        dH = dLoss @ self.params["final"]["Wu"].T




        





x,y = get_target_labels(block_size,train_set,batch_size)
print(x[0],y[0])




tinygpt = Transformer(embedding_d=64,heads=4,layers=2,qk_d=128,v_d=64,nToken=8,ffn_w_d=128)
input = tinygpt.encode("Whose en")
output = input
mean_loss= np.mean([tinygpt.loss_calculation(tinygpt.forward(x[i]),y[i]) for i in range(batch_size)])
print(mean_loss)
