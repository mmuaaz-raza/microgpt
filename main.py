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
        attentions_blocks = []
        for i in range(heads) :
            Q = input @ self.params["attention"]["Wq"][layer][i]
            K = input @ self.params["attention"]["Wk"][layer][i]
            V = input @ self.params["attention"]["Wv"][layer][i]

            Qk = Q @ K.T
            Qk /= (self.dimensions["qk_d"])**0.5
            mask = np.tril(np.ones((block_size)))
            mask[mask==0] = -np.inf 
            mask[mask==1] = 0 
            Qk += mask 
            Attention = softmax(Qk) @ V 
            attentions_blocks.append(Attention)

        attentions = np.concatenate(attentions_blocks,axis=1)    
        final = attentions @ self.params["attention"]["Wp"][layer]
        input += final 
        return layer_norm_cal(input,self.params["attention"]["epsilon"][layer]) * self.params["attention"]["gama"][layer] + self.params["attention"]["beta"][layer]

    
    def positional_encoding(self,input): #input(number of token , dimension of embeddings)
            encoded = np.array(input)
            d_model = input.shape[1]
            positions = input.shape[0]
            evens_pos = np.arange(0,d_model,2)
            odd_pos = np.arange(1,d_model,2)
            for pos in range(positions):
                for ei in evens_pos:
                    encoded[pos][ei] += math.sin(pos/(1e4**(ei/d_model)))
                for oi in odd_pos:
                    encoded[pos][oi] += math.cos(pos/(1e4**(oi/d_model)))
            return encoded









