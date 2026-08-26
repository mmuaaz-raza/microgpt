import numpy as np
from process_data import load_essentials, get_target_labels, softmax, layer_norm_cal, relu, drelu

block_size = 8
batch_size = 32
heads = 4
alpha = 0.01
epsilon = 1e-7
train_set, test_set, itos, stoi, encode, decode = load_essentials()
# ? ed = embedding dimensions , heads = number of attention head in each attention block , qk_d = query and key dimensions at each attention head , v_d = value dimension for each attention head ,layers = number of attention+ffn blocks in a sequence/stacked manner , ffn_wd = weight matrix dimension at each hidden layer in ffn block , nToken = number of tokens (8 in mycase)


class Transformer():
    def __init__(self, ed, heads, qk_d, v_d, layers, ffn_wd, nToken) -> None:
        self.params = {}
        self.dimensions = {"ed": ed, "qk_d": int(qk_d/heads), "v_d": int(
            v_d/heads), "p_d": v_d, "ffn_wd": ffn_wd, "nToken": nToken, "layers": layers,"heads":heads}
        self.train_set, self.test_set, self.itos, self.stoi, self.encode, self.decode = load_essentials()
        # ? embedding table of the model (vocabulary size, embedding dimension)
        self.params["token_embedding_table"] = np.random.randn(
            len(self.itos), ed) * (1 / (ed)**0.5)
        rng = np.random.default_rng()
        #! Attention block
        self.params["attention"] = {}

        # ? layer norm trainable params
        self.params["attention"]["gama"] = rng.standard_normal(
            (layers, ed))/(ed**0.5)
        self.params["attention"]["beta"] = rng.standard_normal(
            (layers, ed))/(ed**0.5)

        # ? dimension : Wp=projection weight matrix : (layers,v_d*heads, ed)

        self.params["attention"]["Wp"] = rng.standard_normal(
            (layers, self.dimensions["v_d"]*heads, ed)) * (1 / self.dimensions["v_d"]**0.5)

        # ? dimension = (layers, heads , embedding dimentsion of each token, Qk/v wieght dimension)

        self.params["attention"]["Wq"] = rng.standard_normal(
            (layers, heads, ed, self.dimensions["qk_d"])) * (1 / ed**0.5)
        self.params["attention"]["Wk"] = rng.standard_normal(
            (layers, heads, ed, self.dimensions["qk_d"])) * (1 / ed**0.5)
        self.params["attention"]["Wv"] = rng.standard_normal(
            (layers, heads, ed, self.dimensions["v_d"])) * (1 / ed**0.5)

        #! FFN block
        self.params["ffn"] = {}

        # ? layer norm trainable params
        self.params["ffn"]["gama"] = rng.standard_normal(
            (layers, ed))/(ed**0.5)
        self.params["ffn"]["beta"] = rng.standard_normal(
            (layers, ed))/(ed**0.5)

        self.params["ffn"]["W0"] = rng.standard_normal(
            (layers, ed, self.dimensions["ffn_wd"])) * (1 / self.dimensions["ed"]**0.5)
        self.params["ffn"]["B0"] = rng.standard_normal(
            (layers, 1, self.dimensions["ffn_wd"])) * (1 / self.dimensions["ed"]**0.5)
        self.params["ffn"]["W1"] = rng.standard_normal(
            (layers, self.dimensions["ffn_wd"], ed)) * (1 / self.dimensions["ffn_wd"]**0.5)
        self.params["ffn"]["B1"] = rng.standard_normal(
            (layers, 1, ed)) * (1 / self.dimensions["ffn_wd"]**0.5)

        #! final block
        self.params["final"] = {}

        # ? Wu = unembedding weight matrix to project (nToken,ed) -> (nToken,len of vaocabulary)
        self.params["final"]["Wu"] = rng.standard_normal(
            (ed, len(self.itos))) / ed**0.5

        # ? layer norm trainable params
        self.params["final"]["gama"] = rng.standard_normal((1,ed))/(ed**0.5)
        self.params["final"]["beta"] = rng.standard_normal((1,ed))/(ed**0.5)

        #! Temporary storage for runtime params in forward Pass
        t = self.forward_runtime = {}
        t["ffn"] = {"A0": [], "A1": [], "Z0": [],
                    "Xfn": [], "Xhat": [], "Xlm": [], "Xlv": []}
        t["attention"] = {"V": [[] for _ in range(layers)], "Q": [[] for _ in range(layers)], 
                          "K": [[] for _ in range(layers)], "Qk": [[] for _ in range(layers)], "att": [[] for _ in range(layers)], "Xhat": [], "Xlm": [], "Xlv": [],"combined_att":[],"Xfn":[]}

    def init_step(self, x):
        # ? dim(input) = (number of tokens in input (nToken) , embedding_dimension)
        input = np.stack([self.params["token_embedding_table"][xi]
                         for xi in x])
        # ? add absolute fixed positional encoding
        input = input + self.positional_encoding(input)
        return input

    def attention(self, x, layer):  # dim(x) : (nToken,ed)
        # Apply layernorm to input(x)
        Xhat, mean, variance = layer_norm_cal(x, epsilon)
        Xn1 = Xhat * self.params["attention"]["gama"][layer] + self.params["attention"]["beta"][layer]

        # cummulative/concatented results from every attention head
        attentions_blocks = []
        # run attention layer heads on norm x
        mask = np.tril(np.ones((block_size)))
        mask[mask == 0] = -np.inf
        mask[mask == 1] = 0
        # storing runtime variables/calculated values
        t = self.forward_runtime["attention"]
        t["Xhat"].append(Xhat)
        t["Xlm"].append(mean)
        t["Xlv"].append(variance)
        t["Xfn"].append(Xn1)
        for i in range(heads):
            # dim(Q,K) : (nToken,qk_d)
            Q = Xn1 @ self.params["attention"]["Wq"][layer][i]
            K = Xn1 @ self.params["attention"]["Wk"][layer][i]
            # dim(V) : (nToken,v_d)
            V = Xn1 @ self.params["attention"]["Wv"][layer][i]
            # dim(QK) : (nToken,nToken)

            Qk = Q @ K.T
            # ! revision needed
            Qk /= (self.dimensions["qk_d"])**0.5
            # apply masking to restrict the attention to only attends to previous tokens
            Qk += mask

            att = softmax(Qk)
            # dim(attb_res) : (nToken,v_d)
            attv = att @ V
            attentions_blocks.append(attv)

            # storing runtime variables
            t["V"][layer].append(V)
            t["Q"][layer].append(Q)
            t["K"][layer].append(K)
            t["Qk"][layer].append(Qk)
            t["att"][layer].append(att)

        # dim(combined_att) : (nToken,heads*v_d)
        combined_att = np.concatenate(attentions_blocks, axis=-1)
        t["combined_att"].append(combined_att)

        # dim(final) : (nToken,ed)
        projected_att = combined_att @ self.params["attention"]["Wp"][layer]

        # residual connection to initial input (x)
        Xat = x + projected_att
        return Xat 

    # ? input(number of token , dimension of embeddings)
    def positional_encoding(self, input):
        d_positions = positions = input.shape[0]
        d_model = input.shape[1]  # d_model = ed
        encoded = np.zeros((d_positions, d_model))
        # PE(position of token = pos,ith value(col) in specific pos(row) of token matrix = 2i)
        # PE(pos,2i) = sin (pos / 10000 ^ 2i/d_model )
        # PE(pos,2i+1) = cos (pos/ 10000 ^ 2i/d_model)

        # ? dim(divisor) = (1,d_model/2)
        divisor = 10000**(np.arange(0, d_model, 2)/d_model)

        # ? dim(positions) = (number of tokens/d_positions , 1)
        positions = np.arange(d_positions)[:, np.newaxis]

        encoded[:, 0::2] = np.sin(positions/divisor)
        encoded[:, 1::2] = np.cos(positions/divisor)

        return encoded

    def feedforwardlayer(self, x, layer):  # dim(x) : (nToken,ed)
        # apply layernorm to x
        # dim(Xn2) = dim(x)
        Xhat, mean, variance = layer_norm_cal(x, epsilon)
        Xn2 = Xhat * self.params["ffn"]["gama"][layer] + \
            self.params["ffn"]["beta"][layer]
        # dim(Z0)=dim(A0) : (nToken,ffn_wd)
        Z0 =Xn2  @ self.params["ffn"]["W0"][layer] + \
            self.params["ffn"]["B0"][layer]
        A0 = relu(Z0)
        # dim(A1) : (nToken,ed)
        A1 = A0 @ self.params["ffn"]["W1"][layer] + \
            self.params["ffn"]["B1"][layer]

        # ? setting runtime variables
        t = self.forward_runtime
        # dim(t[ffn][var]) : (layer, value)
        t["ffn"]["A0"].append(A0)
        t["ffn"]["A1"].append(A1)
        t["ffn"]["Z0"].append(Z0)
        t["ffn"]["Xfn"].append(Xn2)
        t["ffn"]["Xhat"].append(Xhat)
        t["ffn"]["Xlm"].append(mean)
        t["ffn"]["Xlv"].append(variance)
        # residual connection to original input(x)
        Xfn = A1 + x
        return Xfn

    def final_step(self, Xfn):  # dim(Xfn) : (nToken,ed)
        # apply layer norm
        Xhat, mean, variance = layer_norm_cal(Xfn, epsilon)
        Xlm, Xlv = mean, variance
        # dim(Xlf) :(nToken , ed)
        Xlf = Xhat * self.params["final"]["gama"] + self.params["final"]["beta"]
        # dim(Xv) : (nToken,no. of tokens in vocabulary)
        Xv = Xlf @ self.params["final"]["Wu"]
        Xf = softmax(Xv)

        # setting runtime variables
        t = self.forward_runtime["final"]
        t["Xhat"], t["Xlf"], t["Xlm"], t["Xlv"], t["Xv"], t["Xf"] = [Xhat], [Xlf], [Xlm], [Xlv], [Xv], [Xf]

        return Xf

    def loss_calculation(self, probs, y):
        return -np.mean([np.log(probs[i, y[i]]+1e-9) for i in range(probs.shape[0])])

    def forward(self, x):
        self.dimensions["nToken"] = len(x)
        oFFn = self.init_step(x)
        for i in range(self.dimensions["layers"]):
            oAttention = self.attention(oFFn, i)
            oFFn = self.feedforwardlayer(oAttention, i)
        self.forward_runtime["final"] = {}
        t = self.forward_runtime["final"]
        t["Xfn"] = oFFn
        Xf = self.final_step(oFFn)
        return Xf
    def layernormGrad(self,dXlf,runtimeParams,modelParams,layerindex):
        

        dgamaf = np.sum(dXlf * runtimeParams["Xhat"][layerindex], axis=0) #(1,ed)
        modelParams["gama"][layerindex] -= alpha * dgamaf
        
        dbetaf = np.sum(dXlf, axis=0)
        modelParams["beta"][layerindex] -= alpha * dbetaf
        
        dX_hat = dXlf * modelParams["gama"][layerindex]
        # d= derivative , m = mean , f= final layer norm
        dXhat_dmf = -1/((runtimeParams["Xlv"][layerindex]+epsilon)**0.5)  # m = mean
        dmf = dX_hat * dXhat_dmf
        dXhat_dvf = -1/2 * (runtimeParams["Xfn"][layerindex]-runtimeParams["Xlm"][layerindex]) / ((runtimeParams["Xlv"][layerindex]+epsilon)**3/2)  # v = variance
        dvf = dX_hat * dXhat_dvf
        dXhat_dXfn = -dXhat_dmf
        dm_dXfn = 1/(self.dimensions["ed"])
        dv_dXfn = (2/(self.dimensions["ed"])) * (runtimeParams["Xfn"][layerindex]-runtimeParams["Xlm"][layerindex])
        dXfn = dX_hat * dXhat_dXfn + dmf * dm_dXfn + dvf * dv_dXfn
        return dXfn
    def softmaxGrad(self,input,dinput):
        return  input * (dinput - np.sum(dinput*input,axis=1,keepdims=True))
    def backward(self, x, y):
        Xf = self.forward(x)

        # All the runtime variables
        t = self.forward_runtime

        #! final step gradients
        T = Xf.shape[0]
        Xf[np.arange(T), y] -= 1
        dXv = Xf * (1/T)  # (T,vocab_size)

        dWu = t["final"]["Xlf"][0].T @ dXv  # (ed,vocab_size)
        self.params["final"]["Wu"] -= alpha * dWu
        


        dXlf = dXv @ self.params["final"]["Wu"].T  # (T,ed)

        # ? final step layer norm gradient
        dXfn = self.layernormGrad(dXlf,t["final"],self.params["final"],0)

        for layer in range(self.dimensions["layers"]-1,-1,-1):
            # ffn gradients 
            dA1 = dXfn 

            dW1 = t["ffn"]["A0"][layer].T @ dA1
            dB1 = np.sum(dA1,axis=0,keepdims=True)
            # print(dA1.shape,self.params["ffn"]["B1"][layer].shape)
            self.params["ffn"]["W1"][layer] -= alpha * dW1
            self.params["ffn"]["B1"][layer] -= alpha * dB1
            
            dA0 = dA1 @ self.params["ffn"]["W1"][layer].T
            dZ0 = dA0 * drelu(t["ffn"]["Z0"][layer])

            dW0 = t["ffn"]["Xfn"][layer].T @ dZ0
            dB0 = np.sum(dZ0,axis=0,keepdims=True)
            self.params["ffn"]["W0"][layer] -= alpha * dW0
            self.params["ffn"]["B0"][layer] -= alpha * dB0
            
            dXn2 = dZ0 @ self.params["ffn"]["W0"][layer].T
            dXat = 1*dXfn + self.layernormGrad(dXn2,t["ffn"],self.params["ffn"],layer) # due to residual connection

            #! Attention block grads
            dprojected_att = dXat

            dWp = t["attention"]["combined_att"][layer].T @ dprojected_att
            self.params["attention"]["Wp"][layer] -= alpha * dWp

            dcombined_att = dprojected_att @ self.params["attention"]["Wp"][layer].T
            dXn1 = np.zeros_like(t["attention"]["Xfn"][layer]) # to be initialized
            for head in range(self.dimensions["heads"]):
                sti = head * self.dimensions["v_d"]
                dattv = dcombined_att[:, sti : sti + self.dimensions["v_d"]]

                dV = t["attention"]["att"][layer][head].T @ dattv
                
                # print(dattv.shape)
                datt = dattv @ t["attention"]["V"][layer][head].T



                att =t["attention"]["att"][layer][head]
                dQK = self.softmaxGrad(att,datt)

                dQ = dQK @ t["attention"]["K"][layer][head]
                dK = dQK.T @ t["attention"]["Q"][layer][head]
                dWq = t["attention"]["Xfn"][layer].T @ dQ 
                dWk = t["attention"]["Xfn"][layer].T @ dK 
                dWv = t["attention"]["Xfn"][layer].T @ dV

                self.params["attention"]["Wv"][layer][head] -= alpha * dWv
                self.params["attention"]["Wq"][layer][head] -= alpha * dWq
                self.params["attention"]["Wk"][layer][head] -= alpha * dWk
                dXn1_ = dV @ self.params["attention"]["Wv"][layer][head].T + dQ @ self.params["attention"]["Wq"][layer][head].T + dK @ self.params["attention"]["Wk"][layer][head].T
                dXn1+= dXn1_

            dXp = 1*dXat + self.layernormGrad(dXn1,t["attention"],self.params["attention"],layer)
            





        




x, y = get_target_labels(block_size, train_set, batch_size)


tinygpt = Transformer(ed=64, heads=4, layers=8, qk_d=128,
                      v_d=64, nToken=8, ffn_wd=128)

input = x[0]
output = y[0]

print(tinygpt.backward(input,output))
# print(tinygpt.backward(input,output))
