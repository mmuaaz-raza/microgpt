"""
model.py
--------
NumPy-only implementation of a decoder-only Transformer language model with
manual backpropagation and Adam/SGD optimization.
"""

import copy
import os
import pickle
import numpy as np

from process_data import load_essentials, softmax, layer_norm_cal, gelu, dgelu
from Tmodel import ModelTrainableParams, ModelDimensions, RuntimeParams

block_size    = 64
batch_size    = 32
epsilon       = 1e-7
beta1         = 0.9
beta2         = 0.999
modelFilename = "shakespare.pkl"

ed     = 128
heads  = 4
qk_d   = 32
v_d    = 32
layers = 4
ffn_wd = 4 * ed


class Transformer:
    params:          ModelTrainableParams
    gradients:       ModelTrainableParams
    dimensions:      ModelDimensions

    def load_model(self, filename: str) -> bool:
        if filename and os.path.exists(filename):
            with open(filename, "rb") as f:
                saved = pickle.load(f)
                self.params      = copy.deepcopy(saved["params"])
                self.adamParamsM = copy.deepcopy(saved["adamParamsM"])
                self.adamParamsV = copy.deepcopy(saved["adamParamsV"])
                self.dimensions  = copy.deepcopy(saved["dimensions"])
            print(f"Model loaded from '{filename}'.")
            return True
        return False

    def __init__(
        self,
        ed: int,
        heads: int,
        qk_d: int,
        v_d: int,
        layers: int,
        ffn_wd: int,
        seq_len: int,
        ModelFileName: str = None,  # type: ignore
    ) -> None:
        (
            self.train_set,
            self.test_set,
            self.itos,
            self.stoi,
            self.encode,
            self.decode,
        ) = load_essentials()

        self.modelFileName = ModelFileName
        is_saved = self.load_model(ModelFileName)

        if not is_saved:
            vocab_size = len(self.itos)
            self.dimensions = ModelDimensions(
                ed, qk_d, v_d,
                heads * v_d,
                ffn_wd, seq_len, layers, heads, vocab_size,
            )
            self.params      = ModelTrainableParams(self.dimensions)
            self.adamParamsM = ModelTrainableParams(self.dimensions, allzero=True)
            self.adamParamsV = ModelTrainableParams(self.dimensions, allzero=True)

        self.gradients       = ModelTrainableParams(self.dimensions, allzero=True)
        self.forward_runtime = RuntimeParams()

    def save(self, epoch: int) -> None:
        if self.modelFileName:
            payload = {
                "params":      self.params,
                "dimensions":  self.dimensions,
                "adamParamsM": self.adamParamsM,
                "adamParamsV": self.adamParamsV,
                "epoch":       epoch,
            }
            with open(self.modelFileName, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def positional_encoding(self, x: np.ndarray) -> np.ndarray:
        seq_len, d_model = x.shape
        encoded   = np.zeros((seq_len, d_model))
        divisor   = 10000 ** (np.arange(0, d_model, 2) / d_model)
        positions = np.arange(seq_len)[:, np.newaxis]

        encoded[:, 0::2] = np.sin(positions / divisor)
        encoded[:, 1::2] = np.cos(positions / divisor)
        return encoded

    def init_step(self, x: np.ndarray) -> np.ndarray:
        embeddings = np.stack([self.params.w_emb[xi] for xi in x])
        Xp = embeddings + self.positional_encoding(embeddings)

        self.forward_runtime.init.Xp    = Xp
        self.forward_runtime.init.input = x
        return Xp

    def attention(self, Xin: np.ndarray, layer: int) -> np.ndarray:
        t = self.forward_runtime.attention

        Xhat, mean, variance = layer_norm_cal(Xin, epsilon)
        Xn1 = Xhat * self.params.attention.gama[layer] + self.params.attention.beta[layer]

        # Causal mask: prevent attending to future tokens
        seq_len = Xin.shape[0]
        mask = np.tril(np.ones((seq_len, seq_len)))
        mask[mask == 0] = -np.inf
        mask[mask == 1] = 0.0

        Q = Xn1 @ self.params.attention.Wq[layer]
        K = Xn1 @ self.params.attention.Wk[layer]
        V = Xn1 @ self.params.attention.Wv[layer]

        QK  = Q @ np.transpose(K, axes=(0, 2, 1))
        QK /= self.dimensions.qk_d ** 0.5
        QK += mask
        att  = softmax(QK)
        attv = att @ V

        combined_att = np.reshape(
            np.transpose(attv, axes=(1, 0, 2)),
            (attv.shape[1], -1),
        )

        projected_att = combined_att @ self.params.attention.Wp[layer]
        Xat = Xin + projected_att

        t.Xhat.append(Xhat)
        t.Xlm.append(mean)
        t.Xlv.append(variance)
        t.Xfn.append(Xn1)
        t.V.append(V)
        t.Q.append(Q)
        t.K.append(K)
        t.QK.append(QK)
        t.att.append(att)
        t.combined_att.append(combined_att)
        t.input.append(Xin)

        return Xat

    def feedforwardlayer(self, Xat: np.ndarray, layer: int) -> np.ndarray:
        t = self.forward_runtime.ffn

        Xhat, mean, variance = layer_norm_cal(Xat, epsilon)
        Xn2 = Xhat * self.params.ffn.gama[layer] + self.params.ffn.beta[layer]

        Z0  = Xn2 @ self.params.ffn.W0[layer] + self.params.ffn.B0[layer]
        A0  = gelu(Z0)
        A1  = A0 @ self.params.ffn.W1[layer] + self.params.ffn.B1[layer]
        Xfn = Xat + A1

        t.A0.append(A0)
        t.A1.append(A1)
        t.Z0.append(Z0)
        t.Xfn.append(Xn2)
        t.Xhat.append(Xhat)
        t.Xlm.append(mean)
        t.Xlv.append(variance)
        t.input.append(Xat)

        return Xfn

    def final_step(self, Xfn: np.ndarray) -> np.ndarray:
        Xhat, mean, variance = layer_norm_cal(Xfn, epsilon)
        Xlf = Xhat * self.params.final.gama + self.params.final.beta

        Xv = Xlf @ self.params.final.Wu
        Xf = softmax(Xv)

        t = self.forward_runtime.final
        t.Xhat  = [Xhat]
        t.Xlf   = [Xlf]
        t.Xlm   = [mean]
        t.Xlv   = [variance]
        t.Xv    = [Xv]
        t.Xf    = [Xf]
        t.input = [Xfn]

        return Xf

    def forward(self, x: np.ndarray) -> np.ndarray:
        if len(x) > self.dimensions.seq_len:
            x = x[-self.dimensions.seq_len:]

        self.forward_runtime = RuntimeParams()

        out = self.init_step(x)
        for layer in range(self.dimensions.layers):
            out = self.attention(out, layer)
            out = self.feedforwardlayer(out, layer)

        return self.final_step(out)

    def loss_calculation(self, probs: np.ndarray, y: np.ndarray) -> float:
        y_trunc = y[-probs.shape[0]:]
        return -np.mean(
            [np.log(probs[i, y_trunc[i]] + 1e-9) for i in range(probs.shape[0])]
        )

    def calculateBatchLoss(self, x: np.ndarray, y: np.ndarray) -> float:
        total_loss = sum(
            self.loss_calculation(self.forward(inp), out)
            for inp, out in zip(x, y)
        )
        return total_loss / batch_size

    def layernormGrad(
        self,
        dY: np.ndarray,
        runtimeParams,
        modelParams,
        layerindex: int,
        Gradients,
    ) -> np.ndarray:
        X    = runtimeParams.input[layerindex]
        Xhat = runtimeParams.Xhat[layerindex]

        dgamma = np.sum(dY * Xhat, axis=0)
        dbeta  = np.sum(dY,        axis=0)

        dXhat = dY * modelParams.gama[layerindex]

        std_inv         = 1.0 / np.sqrt(runtimeParams.Xlv[layerindex] + epsilon)
        mean_dXhat      = np.mean(dXhat,        axis=1, keepdims=True)
        mean_dXhat_Xhat = np.mean(dXhat * Xhat, axis=1, keepdims=True)

        dX = std_inv * (dXhat - mean_dXhat - Xhat * mean_dXhat_Xhat)

        Gradients.gama[layerindex] += dgamma
        Gradients.beta[layerindex] += dbeta

        return dX

    def softmaxGrad(self, probs: np.ndarray, dprobs: np.ndarray) -> np.ndarray:
        return probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        Xf = self.forward(x)
        if Xf is None:
            return

        t = self.forward_runtime

        seq_len = Xf.shape[0]
        y_trunc = y[-seq_len:]

        # Combined softmax + cross-entropy gradient: dL/d(logit) = (prob - 1) / N
        dXf = Xf.copy()
        dXf[np.arange(seq_len), y_trunc] -= 1
        dXv = dXf / seq_len

        dWu  = t.final.Xlf[0].T @ dXv
        dXlf = dXv @ self.params.final.Wu.T
        self.gradients.final.Wu += dWu

        dXfn = self.layernormGrad(
            dXlf, t.final, self.params.final, 0, self.gradients.final
        )

        dXp = np.zeros_like(t.init.Xp)

        # Backpropagation through Transformer blocks in reverse
        for layer in range(self.dimensions.layers - 1, -1, -1):
            dA1 = dXfn

            dW1 = t.ffn.A0[layer].T @ dA1
            dB1 = np.sum(dA1, axis=0, keepdims=True)

            dA0 = dA1 @ self.params.ffn.W1[layer].T
            dZ0 = dA0 * dgelu(t.ffn.Z0[layer])

            dW0  = t.ffn.Xfn[layer].T @ dZ0
            dB0  = np.sum(dZ0, axis=0, keepdims=True)
            dXn2 = dZ0 @ self.params.ffn.W0[layer].T

            dXat = dXfn + self.layernormGrad(
                dXn2, t.ffn, self.params.ffn, layer, self.gradients.ffn
            )

            self.gradients.ffn.W1[layer] += dW1
            self.gradients.ffn.B1[layer] += dB1
            self.gradients.ffn.W0[layer] += dW0
            self.gradients.ffn.B0[layer] += dB0

            dprojected_att = dXat

            dWp           = t.attention.combined_att[layer].T @ dprojected_att
            dcombined_att = dprojected_att @ self.params.attention.Wp[layer].T
            self.gradients.attention.Wp[layer] += dWp

            dattv = np.transpose(
                dcombined_att.reshape(
                    (dcombined_att.shape[0], self.dimensions.heads, self.dimensions.v_d)
                ),
                axes=(1, 0, 2),
            )

            dV   = np.transpose(t.attention.att[layer], axes=(0, 2, 1)) @ dattv
            datt = dattv @ np.transpose(t.attention.V[layer], axes=(0, 2, 1))

            dQK        = self.softmaxGrad(t.attention.att[layer], datt)
            dQK_scaled = dQK / self.dimensions.qk_d ** 0.5

            dQ = dQK_scaled @ t.attention.K[layer]
            dK = np.transpose(dQK_scaled, axes=(0, 2, 1)) @ t.attention.Q[layer]

            dWq = t.attention.Xfn[layer].T @ dQ
            dWk = t.attention.Xfn[layer].T @ dK
            dWv = t.attention.Xfn[layer].T @ dV
            self.gradients.attention.Wq[layer] += dWq
            self.gradients.attention.Wk[layer] += dWk
            self.gradients.attention.Wv[layer] += dWv

            dXn1 = np.sum(
                dV @ np.transpose(self.params.attention.Wv[layer], axes=(0, 2, 1))
                + dQ @ np.transpose(self.params.attention.Wq[layer], axes=(0, 2, 1))
                + dK @ np.transpose(self.params.attention.Wk[layer], axes=(0, 2, 1)),
                axis=0,
            )

            dXfn = dXp = dXat + self.layernormGrad(
                dXn1, t.attention, self.params.attention, layer, self.gradients.attention
            )

        dEmbd = np.zeros_like(self.params.w_emb)
        np.add.at(dEmbd, self.forward_runtime.init.input, dXp)
        self.gradients.w_emb += dEmbd

    def _adam_update(
        self,
        m: np.ndarray,
        v: np.ndarray,
        gradient: np.ndarray,
        param: np.ndarray,
        beta1: float,
        beta2: float,
        alpha: float,
        batch_size: int,
        t: int,
        epsilon: float = 1e-8,
    ) -> None:
        grad_avg = gradient / batch_size

        m[:] = beta1 * m + (1 - beta1) * grad_avg
        v[:] = beta2 * v + (1 - beta2) * grad_avg ** 2

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        param -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    def _sgd_update(
        self,
        gradient: np.ndarray,
        param: np.ndarray,
        alpha: float,
        batch_size: int,
    ) -> None:
        param -= alpha * (gradient / batch_size)

    def updateWeightsAdam(
        self,
        alpha: float,
        batch_size: int,
        t: int,
        beta1: float,
        beta2: float,
    ) -> None:
        p, g = self.params, self.gradients
        m, v = self.adamParamsM, self.adamParamsV

        param_groups = [
            (m.w_emb,               v.w_emb,               g.w_emb,               p.w_emb),
            (m.attention.gama,      v.attention.gama,      g.attention.gama,      p.attention.gama),
            (m.attention.beta,      v.attention.beta,      g.attention.beta,      p.attention.beta),
            (m.attention.Wp,        v.attention.Wp,        g.attention.Wp,        p.attention.Wp),
            (m.attention.Wq,        v.attention.Wq,        g.attention.Wq,        p.attention.Wq),
            (m.attention.Wk,        v.attention.Wk,        g.attention.Wk,        p.attention.Wk),
            (m.attention.Wv,        v.attention.Wv,        g.attention.Wv,        p.attention.Wv),
            (m.ffn.gama,            v.ffn.gama,            g.ffn.gama,            p.ffn.gama),
            (m.ffn.beta,            v.ffn.beta,            g.ffn.beta,            p.ffn.beta),
            (m.ffn.W0,              v.ffn.W0,              g.ffn.W0,              p.ffn.W0),
            (m.ffn.B0,              v.ffn.B0,              g.ffn.B0,              p.ffn.B0),
            (m.ffn.W1,              v.ffn.W1,              g.ffn.W1,              p.ffn.W1),
            (m.ffn.B1,              v.ffn.B1,              g.ffn.B1,              p.ffn.B1),
            (m.final.Wu,            v.final.Wu,            g.final.Wu,            p.final.Wu),
            (m.final.gama,          v.final.gama,          g.final.gama,          p.final.gama),
            (m.final.beta,          v.final.beta,          g.final.beta,          p.final.beta),
        ]

        for mi, vi, gi, pi in param_groups:
            self._adam_update(mi, vi, gi, pi, beta1, beta2, alpha, batch_size, t)

        self.gradients = ModelTrainableParams(self.dimensions, allzero=True)

    def updateWeightsSDG(self, alpha: float, batch_size: int) -> None:
        p, g = self.params, self.gradients

        param_groups = [
            (g.w_emb,          p.w_emb),
            (g.attention.gama, p.attention.gama),
            (g.attention.beta, p.attention.beta),
            (g.attention.Wp,   p.attention.Wp),
            (g.attention.Wq,   p.attention.Wq),
            (g.attention.Wk,   p.attention.Wk),
            (g.attention.Wv,   p.attention.Wv),
            (g.ffn.gama,       p.ffn.gama),
            (g.ffn.beta,       p.ffn.beta),
            (g.ffn.W0,         p.ffn.W0),
            (g.ffn.B0,         p.ffn.B0),
            (g.ffn.W1,         p.ffn.W1),
            (g.ffn.B1,         p.ffn.B1),
            (g.final.Wu,       p.final.Wu),
            (g.final.gama,     p.final.gama),
            (g.final.beta,     p.final.beta),
        ]

        for gi, pi in param_groups:
            self._sgd_update(gi, pi, alpha, batch_size)

        self.gradients = ModelTrainableParams(self.dimensions, allzero=True)
