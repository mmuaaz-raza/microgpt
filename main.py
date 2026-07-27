import numpy as np 
from process_data import load_essentials , get_target_labels ,softmax , layer_norm_cal,relu, drelu

block_size = 8 
batch_size = 32
heads = 4
alpha = 0.01
train_set,test_set,itos,stoi ,encode,decode = load_essentials()

class Transformer():
    def __init__(self,ed,heads,qk_d,v_d,layers,ffn_w_d,nToken) -> None:
        self.params={}
        self.dimensions = {"ed":ed,"qk_d":int(qk_d/heads),"v_d":int(v_d/heads),"p_d":v_d,"ffn_w_d":ffn_w_d,"nToken":nToken,"layers":layers}
        self.train_set,self.test_set ,self.itos,self.stoi,self.encode , self.decode = load_essentials()
        # embedding table of the model (vocabulary size, embedding dimension)
        self.params["token_embedding_table"] = np.random.randn(len(self.itos),ed) * ( 1 / (ed)**0.5)
        rng = np.random.default_rng()
        self.params["attention"] = {}

        self.params["attention"]["Wp"] = rng.standard_normal((layers,self.dimensions["v_d"]*heads, ed)) * (1 / self.dimensions["v_d"]**0.5)
        self.params["attention"]["epsilon"] = rng.random((layers))
        self.params["attention"]["gama"] = rng.standard_normal((layers,ed))/(ed**0.5)
        self.params["attention"]["beta"] = rng.standard_normal((layers,ed))/(ed**0.5) #dim = (layers, heads , embedding dimentsion of each token, Qk wieght dimension)
        self.params["attention"]["Wq"] = rng.standard_normal((layers,heads,ed, self.dimensions["qk_d"])) * (1 / ed**0.5) 
        self.params["attention"]["Wk"] = rng.standard_normal((layers,heads,ed, self.dimensions["qk_d"])) * (1 / ed**0.5)
        self.params["attention"]["Wv"] = rng.standard_normal((layers,heads,ed, self.dimensions["v_d"])) * (1 / ed**0.5)

        self.params["ffn"] = {}
        self.params["ffn"]["epsilon"] = rng.random((layers,1))
        self.params["ffn"]["gama"] = rng.standard_normal((layers,ed))/(ed**0.5)
        self.params["ffn"]["beta"] = rng.standard_normal((layers,ed))/(ed**0.5)
        self.params["ffn"]["W0"] = rng.standard_normal((layers, ed,self.dimensions["ffn_w_d"])) * (1 / self.dimensions["ed"]**0.5)
        self.params["ffn"]["W1"] = rng.standard_normal((layers, self.dimensions["ffn_w_d"],ed)) * (1 / self.dimensions["ffn_w_d"]**0.5)
        self.params["ffn"]["B0"] = rng.standard_normal((layers, 1,self.dimensions["ffn_w_d"])) * (1 / self.dimensions["ed"]**0.5)
        self.params["ffn"]["B1"] = rng.standard_normal((layers, 1,ed)) * (1 / self.dimensions["ffn_w_d"]**0.5)

        self.params["final"] = {"Wu":rng.standard_normal((ed,len(self.itos))) * (1 / ed**0.5)}
        self.params["final"]["epsilon"] = rng.random((1))
        self.params["final"]["gama"] = rng.standard_normal((ed))/(ed**0.5)
        self.params["final"]["beta"] = rng.standard_normal((ed))/(ed**0.5)

        t = self.forward_runtime = {}
        t["ffn"] = {"A0":[],"A1":[],"Z0":[],"Xn2":[],"Xhat":[],"Xlm":[],"Xlv":[]}




    def init_step(self,x):
        input = np.stack([self.params["token_embedding_table"][xi] for xi in x]) # dim(input) = number of tokens in input * embedding_dimension
        # add absolute fixed positional encoding
        input = input + self.positional_encoding(input)
        return input

    def attention(self,x,layer): 
        # apply layer norm
        normx = layer_norm_cal(x,self.params["attention"]["epsilon"][layer])[0] * self.params["attention"]["gama"][layer] + self.params["attention"]["beta"][layer]
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
        Xhat,mean,variance = layer_norm_cal(input,self.params["ffn"]["epsilon"][layer])[0]
        Xn2 = normx = Xhat  * self.params["ffn"]["gama"][layer] + self.params["ffn"]["beta"][layer]
        Z0 = normx @ self.params["ffn"]["W0"][layer] + self.params["ffn"]["B0"][layer]
        A0 = relu(Z0)
        A1 = A0 @ self.params["ffn"]["W1"][layer] + self.params["ffn"]["B1"][layer]
        t = self.forward_runtime 
        
        t["ffn"]["A0"].append(A0);
        t["ffn"]["A1"].append(A1);
        t["ffn"]["Z0"].append(Z0);
        t["ffn"]["Xn2"].append(Xn2);
        t["ffn"]["Xhat"].append(Xhat);
        t["ffn"]["Xlm"].append(mean);
        t["ffn"]["Xlv"].append(variance);

        Xfn = A1 + input 
        return Xfn
    
    def final_step (self,Xfn):
        Xhat, mean,variance = layer_norm_cal(Xfn,self.params["final"]["epsilon"])
        Xlm,Xlv = mean,variance
        Xlf  = Xhat  *   self.params["final"]["gama"] + self.params["final"]["beta"]
        Xv  = Xlf @ self.params["final"]["Wu"]
        Xf = softmax(Xv)
        return Xhat , Xlf, Xlm, Xlv , Xv , Xf 
    
    def loss_calculation(self,probs,y):
        return -np.mean([np.log(probs[i,y[i]]+1e-9) for i in range(probs.shape[0])])
    
    def forward(self,x):
        self.dimensions["nToken"] = len(x)
        oFFn = self.init_step(x)
        for i in range(self.dimensions["layers"]):
            oAttention = self.attention(oFFn,i)
            oFFn = self.feedforwardlayer(oAttention,i)
        self.forward_runtime["final"] = {}
        t = self.forward_runtime["final"]
        t["Xfn"] = oFFn
        t["Xhat"],t["Xlf"],t["Xlm"],t["Xlv"],t["Xv"],t["Xf"] = self.final_step(oFFn)
        return  t["Xf"]
             
    def backward(self,x,y):
        # store all the variables inside this->object variables
        probs = self.forward(x)
        t = self.forward_runtime

        T = probs.shape[0]
        probs[np.arange(T),y] -= 1
        dXv = probs * (1/T)  # (T,vocab_size)
        dWu = t["Xlf"].T @ dXv # (ed,vocab_size)

        #final layernorm gradients

        self.params["final"]["Wu"] -= alpha *dWu 

        dXlf = dXv @ self.params["final"]["Wu"].T # (T,ed)
        dγf = np.sum(dXlf * t["Xhat"],axis=0)

        self.params["final"]["gama"] -= alpha * dγf

        dβf = np.sum(dXlf ,axis=0)
        self.params["final"]["beta"] -= alpha * dβf

        dX_hat = dXlf * self.params["final"]["gama"]

        # d= derivative , m = mean , f= final layer norm
        dXhat_dmf = -1/((t["Xlv"]+self.params["final"]["epsilon"])**0.5) # m = mean
        dmf = dX_hat * dXhat_dmf
        dXhat_dvf = -1/2 * (t["Xfn"]-t["Xlm"])/((t["Xlv"]+self.params["final"]["epsilon"])**3/2) # v = variance
        dvf = dX_hat * dXhat_dvf
        dXhat_dXfn = -dXhat_dmf
        dm_dXfn = 1/(self.dimensions["ed"])
        dv_dXfn = (2/(self.dimensions["ed"])) * (t["Xfn"]-t["Xlm"])

        dXfn = dX_hat * dXhat_dXfn + dmf * dm_dXfn + dvf * dv_dXfn

        #! ffn layer

        d0A1ffn = dXfn  #? 0th layer's A1 layer of feed forward network
        d0Xatffn0 = dXfn #? 0th layer's Xat(original input (residual connection))

        #* ffn layer gradients

        d0A0ffn = d0A1ffn @ self.params["ffn"]["W1"][0].T
        d0W1ffn = t["ffn"]["A0"][0].T @ d0A1ffn
        self.params["ffn"]["W1"][0] -= alpha * d0W1ffn
        d0B1ffn = np.sum(d0A1ffn,axis=0)
        self.params["ffn"]["B1"][0] -= alpha * d0B1ffn

        d0Z0ffn = drelu(t["ffn"]["Z0"][0]) * d0A0ffn
        d0W0ffn = t["ffn"]["Xn2"][0].T @ d0Z0ffn
        self.params["ffn"]["W0"][0] -= alpha * d0W0ffn
        d0B0ffn =  np.sum(d0Z0ffn,axis=0)
        self.params["ffn"]["B0"][0] -= alpha * d0B0ffn
        d0Xn2ffn =  d0Z0ffn @ self.params["ffn"]["W0"][0].T

        #* ffn layer norm gradients
        
        
        d0γffn = np.sum(d0Xn2ffn * t["ffn"]["Xhat"],axis=0)
        self.params["ffn"]["gama"][0] -= alpha * d0γffn
        d0βffn = np.sum(d0Xn2ffn,axis=0)
        self.params["ffn"]["beta"][0]-= alpha * d0βffn
        
        d0Xhatffn = d0Xn2ffn * self.params["ffn"]["gama"][0]
        
        # d= derivative , m = mean , f= final layer norm
        d0Xhat_dmfffn = -1/((t["ffn"]["Xlv"][0]+self.params["ffn"]["epsilon"][0])**0.5) # m = mean
        dmf = d0Xhatffn * d0Xhat_dmfffn
        d0Xhat_dvfffn = -1/2 * (t["ffn"]["Xn2"][0]-t["ffn"]["Xlm"][0])/((t["ffn"]["Xlv"][0]+self.params["ffn"]["epsilon"][0])**3/2) # v = variance
        dvf = d0Xhatffn * d0Xhat_dvfffn
        dXhat_dXfn = -d0Xhat_dmfffn
        dm_dXfn = 1/(self.dimensions["ed"])
        dv_dXfn = (2/(self.dimensions["ed"])) * (t["ffn"]["Xn2"][0]-t["ffn"]["Xlm"][0])
        
        d0Xatffn1 = d0Xhatffn * dXhat_dXfn + dmf * dm_dXfn + dvf * dv_dXfn
        d0Xatffn = d0Xatffn1 + d0Xatffn0 # due to residual connection 
        














        





        





x,y = get_target_labels(block_size,train_set,batch_size)


tinygpt = Transformer(ed=64,heads=4,layers=2,qk_d=128,v_d=64,nToken=8,ffn_w_d=128)

input = x[0]
output = y[0]
print(tinygpt.backward(input,output))
