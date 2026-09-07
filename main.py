import numpy as np
import os
import pickle
from process_data import load_essentials, get_target_labels, softmax, layer_norm_cal, gelu, dgelu
import copy
from param_init import ModelTrainableParams ,ModelDimensions
block_size = 64
batch_size = 16
epsilon = 1e-7
train_set, test_set, itos, stoi, encode, decode = load_essentials()


# ? ed = embedding dimensions , heads = number of attention head in each attention block , qk_d = query and key dimensions at each attention head , v_d = value dimension for each attention head ,layers = number of attention+ffn blocks in a sequence/stacked manner , ffn_wd = weight matrix dimension at each hidden layer in ffn block , nToken = number of tokens (8 in mycase)
class Transformer():
    params : ModelTrainableParams
    gradients : ModelTrainableParams
    dimensions : ModelDimensions
    def __init__(self, ed, heads, qk_d, v_d, layers, ffn_wd, nToken, savedModelFileName=None) -> None:
        isSavedParams = False
        self.train_set, self.test_set, self.itos, self.stoi, self.encode, self.decode = load_essentials()

        if savedModelFileName and os.path.exists(savedModelFileName):
            with open(savedModelFileName, "rb") as f:
                savedParams = pickle.load(f)
                self.params = copy.deepcopy(savedParams["params"])
                self.dimensions = copy.deepcopy(savedParams["dimensions"])
                print("loaded")
                isSavedParams = True

        if not isSavedParams:
            self.dimensions = ModelDimensions(ed, qk_d, v_d,heads*v_d, ffn_wd, nToken,layers,heads,len(itos))
            self.params = ModelTrainableParams(self.dimensions)
         
        # to initialize gradients containers for batch gradient updates
        self.gradients = ModelTrainableParams(self.dimensions,allzero=True)
        self.gradients.__init__(self.dimensions,allzero=True)
        #! Temporary storage for runtime params in forward Pass
        t = self.forward_runtime = {}
        t["ffn"] = {"A0": [], "A1": [], "Z0": [],
                    "Xfn": [], "Xhat": [], "Xlm": [], "Xlv": [], "input": []}
        t["attention"] = {"V": [], "Q": [],
                          "K": [], "Qk": [], "att": [], "Xhat": [], "Xlm": [], "Xlv": [], "combined_att": [], "Xfn": [], "input": []}
        t["init"] = {"Xp": [], "input": []}
# refactor every variable access way to match refactoring
    def init_step(self, x):
        # ? dim(input) = (number of tokens in input (nToken) , embedding_dimension)
        input = np.stack([self.params.w_emb[xi]
                         for xi in x])
        # ? add absolute fixed positional encoding
        Xp = input + self.positional_encoding(input)
        self.forward_runtime["init"]["Xp"] = Xp
        self.forward_runtime["init"]["x"] = x
        return Xp

    def attention(self, x, layer):  # dim(x) : (nToken,ed)
        # Apply layernorm to input(x)
        Xhat, mean, variance = layer_norm_cal(x, epsilon)
        Xn1 = Xhat * self.params.attention.gama[layer] + \
            self.params.attention.beta[layer]

        # run attention layer heads on norm x
        mask = np.tril(np.ones((x.shape[0])))
        mask[mask == 0] = -np.inf
        mask[mask == 1] = 0
        # storing runtime variables/calculated values
        t = self.forward_runtime["attention"]
        t["Xhat"].append(Xhat)
        t["Xlm"].append(mean)
        t["Xlv"].append(variance)
        t["Xfn"].append(Xn1)

        # dim(V) : (head,nToken,qk_d)
        V = Xn1 @ self.params.attention.Wv[layer]
        Q = Xn1 @ self.params.attention.Wq[layer]
        K = Xn1 @ self.params.attention.Wk[layer]

        # dim(QK) : (head,nToken,v_d)
        Qk = Q @ np.transpose(K, axes=(0, 2, 1))
        Qk /= (self.dimensions.qk_d)**0.5
        Qk += mask
        att = softmax(Qk)
        attv = att @ V
        attv = np.reshape(np.transpose(attv, axes=(1, 0, 2)),
                          (attv.shape[1], -1))

        # dim(combined_att) : (nToken,heads*v_d)

        t["combined_att"].append(attv)
        # ? storing runtime variables
        t["V"].append(V)
        t["Q"].append(Q)
        t["K"].append(K)
        t["Qk"].append(Qk)
        t["att"].append(att)
        t["input"].append(x)

        # dim(final) : (nToken,ed)
        projected_att = attv @ self.params.attention.Wp[layer]

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

        # ? applying sine to even positions and cos to odd ones
        encoded[:, 0::2] = np.sin(positions/divisor)
        encoded[:, 1::2] = np.cos(positions/divisor)

        return encoded

    def feedforwardlayer(self, x, layer):  # dim(x) : (nToken,ed)
        # apply layernorm to x
        # dim(Xn2) = dim(x)
        Xhat, mean, variance = layer_norm_cal(x, epsilon)
        Xn2 = Xhat * self.params.ffn.gama[layer] + \
            self.params.ffn.beta[layer]
        # dim(Z0)=dim(A0) : (nToken,ffn_wd)
        Z0 = Xn2  @ self.params.ffn.W0[layer] + \
            self.params.ffn.B0[layer]
        A0 = gelu(Z0)
        # dim(A1) : (nToken,ed)
        A1 = A0 @ self.params.ffn.W1[layer] + \
            self.params.ffn.B1[layer]

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
        t["ffn"]["input"].append(x)
        # residual connection to original input(x)
        Xfn = A1 + x
        return Xfn

    def final_step(self, Xfn):  # dim(Xfn) : (nToken,ed)
        # apply layer norm
        Xhat, mean, variance = layer_norm_cal(Xfn, epsilon)
        Xlm, Xlv = mean, variance
        # dim(Xlf) :(nToken , ed)
        Xlf = Xhat * self.params.final.gama + \
            self.params.final.beta
        # dim(Xv) : (nToken,no. of tokens in vocabulary)
        Xv = Xlf @ self.params.final.Wu
        Xf = softmax(Xv)

        # setting runtime variables
        t = self.forward_runtime["final"]
        t["Xhat"], t["Xlf"], t["Xlm"], t["Xlv"], t["Xv"], t["Xf"], t["input"] = [
            Xhat], [Xlf], [Xlm], [Xlv], [Xv], [Xf], [Xfn]

        return Xf

    def loss_calculation(self, probs, y):
        return -np.mean([np.log(probs[i, y[i]]+1e-9) for i in range(probs.shape[0])])

    def forward(self, x):
        if len(x) != self.dimensions.nToken:
            print("Unacceptable number of input tokens")
            exit(501)
            return
        oFFn = self.init_step(x)
        for i in range(self.dimensions.layers):
            oAttention = self.attention(oFFn, i)
            oFFn = self.feedforwardlayer(oAttention, i)
        self.forward_runtime["final"] = {}
        t = self.forward_runtime["final"]
        t["Xfn"] = oFFn
        Xf = self.final_step(oFFn)
        return Xf

    def layernormGrad(self, dXlf, runtimeParams, modelParams, layerindex, batch_size, Gradients,alpha):

        dgamaf = np.sum(
            dXlf * runtimeParams["Xhat"][layerindex], axis=0)  # (1,ed)

        dbetaf = np.sum(dXlf, axis=0)

        dX_hat = dXlf * modelParams.gama[layerindex]
        # d= derivative , m = mean , f= final layer norm
        dXhat_dmf = -1 / \
            ((runtimeParams["Xlv"][layerindex]+epsilon)**0.5)  # m = mean
        dmf = np.sum(dX_hat * dXhat_dmf, axis=1, keepdims=True)
        dXhat_dvf = -1/2 * (runtimeParams["input"][layerindex]-runtimeParams["Xlm"][layerindex]) / (
            (runtimeParams["Xlv"][layerindex]+epsilon)**(3/2))  # v = variance
        dvf = np.sum(dX_hat * dXhat_dvf, axis=1, keepdims=True)
        dXhat_dXfn = -dXhat_dmf
        dm_dXfn = 1/(self.dimensions.ed)
        dv_dXfn = (2/(self.dimensions.ed)) * (runtimeParams["input"]
                                                 [layerindex]-runtimeParams["Xlm"][layerindex])
        dXfn = dX_hat * dXhat_dXfn + dmf * dm_dXfn + dvf * dv_dXfn

        Gradients.beta[layerindex] += alpha/batch_size * dbetaf
        Gradients.gama[layerindex] += alpha/batch_size * dgamaf
        return dXfn

    def softmaxGrad(self, input, dinput):
        return input * (dinput - np.sum(dinput*input, axis=1, keepdims=True))

    def backward(self, x, y, batch_size,alpha):
        Xf = self.forward(x)
        if Xf is None:
            return

        # All the runtime variables
        t = self.forward_runtime

        #! final step gradients
        T = Xf.shape[0]
        Xf[np.arange(T), y] -= 1
        dXv = Xf * (1/T)  # (T,vocab_size)

        dWu = t["final"]["Xlf"][0].T @ dXv  # (ed,vocab_size)
        dXlf = dXv @ self.params.final.Wu.T  # (T,ed)

        self.gradients.final.Wu += alpha/batch_size * dWu

        # ? final step layer norm gradient
        dXfn = self.layernormGrad(dXlf, t["final"], self.params.final,
                                  0, batch_size=batch_size, Gradients=self.gradients.final,alpha=alpha)

        # default initialization
        dXp = np.zeros_like(t["init"]["Xp"])
        for layer in range(self.dimensions.layers-1, -1, -1):
            # ffn gradients
            dA1 = dXfn

            dW1 = t["ffn"]["A0"][layer].T @ dA1
            dB1 = np.sum(dA1, axis=0, keepdims=True)

            dA0 = dA1 @ self.params.ffn.W1[layer].T
            dZ0 = dA0 * dgelu(t["ffn"]["Z0"][layer])

            dW0 = t["ffn"]["Xfn"][layer].T @ dZ0
            dB0 = np.sum(dZ0, axis=0, keepdims=True)
            dXn2 = dZ0 @ self.params.ffn.W0[layer].T
            dXat = 1*dXfn + self.layernormGrad(dXn2, t["ffn"], self.params.ffn, layer, batch_size, Gradients=self.gradients.ffn,alpha=alpha)

            self.gradients.ffn.W1[layer] += alpha/batch_size * dW1
            self.gradients.ffn.B1[layer] += alpha/batch_size * dB1
            self.gradients.ffn.W0[layer] += alpha/batch_size * dW0
            self.gradients.ffn.B0[layer] += alpha/batch_size * dB0

            #! Attention block grads
            dprojected_att = dXat

            dWp = t["attention"]["combined_att"][layer].T @ dprojected_att

            dcombined_att = dprojected_att @ self.params.attention.Wp[layer].T

            self.gradients.attention.Wp[layer] += alpha/batch_size * dWp
            dXn1 = np.zeros_like(t["attention"]["Xfn"][layer])  # to be initialized
            for head in range(self.dimensions.heads):
                sti = head * self.dimensions.v_d
                dattv = dcombined_att[:, sti: sti + self.dimensions.v_d]

                dV = t["attention"]["att"][layer][head].T @ dattv

                datt = dattv @ t["attention"]["V"][layer][head].T

                att = t["attention"]["att"][layer][head]
                dQK = self.softmaxGrad(att, datt)
                dQK_scaled = dQK * (1/np.sqrt(self.dimensions.qk_d))
                dQ = dQK_scaled @ t["attention"]["K"][layer][head]
                dK = dQK_scaled.T @ t["attention"]["Q"][layer][head]
                dWq = t["attention"]["Xfn"][layer].T @ dQ
                dWk = t["attention"]["Xfn"][layer].T @ dK
                dWv = t["attention"]["Xfn"][layer].T @ dV

                dXn1_ = dV @ self.params.attention.Wv[layer][head].T + \
                    dQ @ self.params.attention.Wq[layer][head].T + \
                    dK @ self.params.attention.Wk[layer][head].T
                dXn1 += dXn1_

                self.gradients.attention.Wv[layer][head] += alpha / \
                    batch_size * dWv
                self.gradients.attention.Wq[layer][head] += alpha / \
                    batch_size * dWq
                self.gradients.attention.Wk[layer][head] += alpha / \
                    batch_size * dWk

            dXfn = dXp = 1*dXat + self.layernormGrad(dXn1, t["attention"], self.params.attention, layer,batch_size, self.gradients.attention,alpha=alpha)

        #! init steps Grad
        dEmbd = np.zeros_like(self.params.w_emb)
        np.add.at(dEmbd, self.forward_runtime["init"]["x"], dXp)
        self.gradients.w_emb += alpha/batch_size * dEmbd

    def updateWeights(self):
        self.params.w_emb -= self.gradients.w_emb

        self.params.attention.gama-= self.gradients.attention.gama
        self.params.attention.beta-= self.gradients.attention.beta

        self.params.attention.Wp-= self.gradients.attention.Wp
        self.params.attention.Wq-= self.gradients.attention.Wq
        self.params.attention.Wk-= self.gradients.attention.Wk
        self.params.attention.Wv-= self.gradients.attention.Wv

        self.params.ffn.gama-= self.gradients.ffn.gama
        self.params.ffn.beta-= self.gradients.ffn.beta

        self.params.ffn.W0-= self.gradients.ffn.W0
        self.params.ffn.B0-= self.gradients.ffn.B0
        self.params.ffn.W1-= self.gradients.ffn.W1
        self.params.ffn.B1-= self.gradients.ffn.B1

        self.params.final.Wu-= self.gradients.final.Wu

        self.params.final.gama-= self.gradients.final.gama
        self.params.final.beta-= self.gradients.final.beta
#      set all gradients back to 0
        self.gradients.__init__(self.dimensions,allzero=True)

    def calculateBatchLoss(self, x, y):
        loss = 0
        for input, output in zip(x, y):
            loss += self.loss_calculation(self.forward(input), output)
        return loss/batch_size

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({"params": self.params, "dimensions": self.dimensions},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

filename = "shakespare.pkl"
tinygpt = Transformer(ed=64, heads=4, layers=8 , qk_d=32, v_d=32,
                      nToken=block_size, ffn_wd=4*64, savedModelFileName=filename)

def train_network(iter,decay_rate,checkpoint_rate,tracking_rate,base_alpha):
    tracking = int(iter*tracking_rate)
    checkpoint = int(checkpoint_rate*iter)
    for i in range(iter):
        alpha = float(base_alpha/(1+float(decay_rate*i)))
        x, y = get_target_labels(batch_size=batch_size,data=train_set, block_size=block_size)
        for input, output in zip(x, y):
            tinygpt.backward(input, output, batch_size,alpha)
        tinygpt.updateWeights()
        if i% checkpoint == 0 and i!=0:
            tinygpt.save(filename)
        if i % tracking == 0:
            print(alpha)
            print(i, "/", iter, " Loss : ", tinygpt.calculateBatchLoss(x, y))

train_network(20,0,float(1),float(1/5),0.01)
# predictoin time
init_input = tinygpt.encode("""First Citizen:
Before we proceed any further, hear me speak.

Al""")
predictions = list(init_input)
for i in range(200):
    output = tinygpt.forward(init_input)
    if output is None:
        break
    decodedindices = init_input[1:]

    maxi = np.argmax(output[output.shape[0]-1]).item()

    decodedindices.append(maxi)
    init_input = decodedindices
    predictions.append(maxi)

tinygpt.save(filename)
text = tinygpt.decode(predictions)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)
