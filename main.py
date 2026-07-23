from base64 import encode
from math import cos
import math

import numpy as np 
from process_data import load_essentials , get_target_labels ,softmax , layer_norm_cal

block_size = 8 
batch_size = 32
heads = 4
class Transformer():
    def __init__(self,embedding_d,heads,qk_d,v_d,layers) -> None:
        self.params={}
        self.dimensions = {"embedding_d":embedding_d,"qk_d":qk_d/heads,"v_d":v_d,"p_d":v_d*heads}
        self.train_set,self.test_set ,self.itos,self.stoi,self.encode , self.decode = load_essentials()
        # embedding table of the model (vocabulary size, embedding dimension)
        self.params["token_embedding_table"] = np.random.randn(len(self.itos),embedding_d) * ( 1 / (embedding_d)**0.5)
        rng = np.random.default_rng()
        self.params["attention"] = {}

        self.params["attention"]["Wp"] = rng.standard_normal((layers,self.dimensions["v_d"]*heads, embedding_d)) * (1 / self.dimensions["v_d"]**0.5)
        self.params["attention"]["epsilon"] = rng.standard_normal((layers))
        self.params["attention"]["gama"] = rng.standard_normal((layers,block_size))/(block_size**0.5)
        self.params["attention"]["beta"] = rng.standard_normal((layers,block_size))/(block_size**0.5)
        #dim = (layers, heads , embedding dimentsion of each token, Qk wieght dimension)
        self.params["attention"]["Wq"] = rng.standard_normal((layers,heads,embedding_d, self.dimensions["qk_d"])) * (1 / embedding_d**0.5) 
        self.params["attention"]["Wk"] = rng.standard_normal((layers,heads,embedding_d, self.dimensions["qk_d"])) * (1 / embedding_d**0.5)
        self.params["attention"]["Wv"] = rng.standard_normal((layers,heads,embedding_d, self.dimensions["v_d"])) * (1 / embedding_d**0.5)

    def forward(self,x,layer): 
        # coverting tokens to embedding vectors
        input = np.stack([self.params["token_embedding_table"][xi] for xi in x]) # dim(input) = number of tokens in input * embedding_dimension
        # add absolute fixed positional encoding
        input = input + self.positional_encoding(input)
        # apply layer norm
        normx = layer_norm_cal(input,self.params["attention"]["epsilon"][layer]) * self.params["attention"]["gama"][layer] + self.params["attention"]["beta"][layer]
        attentions_blocks = []
        # run attention layer heads on norm x
        for i in range(heads) :
            Q = normx @ self.params["attention"]["Wq"][layer][i]
            K = normx @ self.params["attention"]["Wk"][layer][i]
            V = normx @ self.params["attention"]["Wv"][layer][i]

            Qk = Q @ K.T
            Qk /= (self.dimensions["qk_d"])**0.5
            mask = np.tril(np.ones((block_size)))
            mask[mask==0] = -np.inf 
            mask[mask==1] = 0 
            Qk += mask 
            Attention = softmax(Qk) @ V 
            attentions_blocks.append(Attention)

        attentions = np.concatenate(attentions_blocks,axis=-1)    
        final = attentions @ self.params["attention"]["Wp"][layer]

        # residual connection to initial input
        return input + final

    
    def positional_encoding(self,input): #input(number of token , dimension of embeddings)
            d_positions = positions = input.shape[0]
            d_model = input.shape[1]
            encoded = np.zeros((d_positions,d_model))

            divisor =  10000**(np.arange(0,d_model,2)/d_model)
            positions = np.arange(d_positions)[:,np.newaxis]
            encoded[:,0::2] = np.sin(positions/divisor)
            encoded[:,1::2] = np.cos(positions/divisor)

            return encoded + input 
    







