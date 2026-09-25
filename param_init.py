from dataclasses import dataclass, asdict, field , InitVar
from typing import Any
import numpy as np


@dataclass
class ModelDimensions():
    ed: int
    qk_d: int
    v_d: int
    p_d: int
    ffn_wd: int
    nToken: int
    layers: int
    heads: int
    vocab_size:int
  


@dataclass
class FFNParams():
    dimensions : InitVar[ModelDimensions]
    allzero:InitVar[bool] = False
    W0: np.ndarray = field(init=False)
    B0: np.ndarray = field(init=False)
    W1: np.ndarray = field(init=False)
    B1:  np.ndarray = field(init=False)
    gama: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)

    def __post_init__(self,dimensions: ModelDimensions,allzero:bool = False):
        rng = np.random.default_rng()

        layers = dimensions.layers
        ffn_wd = dimensions.ffn_wd
        ed = dimensions.ed
        if allzero:
            self.gama = np.zeros((layers, ed))
            self.beta = np.zeros((layers, ed))

            self.W0 = np.zeros((layers, ed, ffn_wd)) 
            self.B0 = np.zeros((layers, 1, ffn_wd))
            self.W1 = np.zeros((layers, ffn_wd, ed)) 
            self.B1 = np.zeros((layers, 1, ed))
        else:
            self.gama = np.ones((layers, ed))
            self.beta = np.zeros((layers, ed))

            self.W0 = rng.standard_normal((layers, ed, ffn_wd)) * (2 / ed)**0.5
            self.B0 = np.zeros((layers, 1, ffn_wd))
            self.W1 = rng.standard_normal((layers, ffn_wd, ed)) * (2 / ffn_wd)**0.5
            self.B1 = np.zeros((layers, 1, ed))


@dataclass
class AttentionParams():
    dimensions: InitVar[ModelDimensions]
    allzero:InitVar[bool] = False
    Wq: np.ndarray = field(init=False,metadata={"detail":""})
    Wk: np.ndarray = field(init=False)
    Wv: np.ndarray = field(init=False)
    Wp:  np.ndarray = field(init=False)
    gama: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)

    def __post_init__(self,dimensions,allzero:bool=False):
        rng = np.random.default_rng()

        layers = dimensions.layers
        heads = dimensions.heads
        ed = dimensions.ed
        v_d = dimensions.v_d
        qk_d = dimensions.qk_d
        if allzero:
            self.gama = np.zeros((layers, ed))
            self.beta = np.zeros((layers, ed))
            
            self.Wq = np.zeros((layers, heads, ed, qk_d)) 
            self.Wk = np.zeros((layers, heads, ed, qk_d)) 
            self.Wv = np.zeros((layers, heads, ed, v_d)) 
            self.Wp = np.zeros((layers, v_d*heads, ed)) 
            return
        else :
            self.gama = np.ones((layers, ed))
            self.beta = np.zeros((layers, ed))

            self.Wq = rng.standard_normal((layers, heads, ed, qk_d)) * (1 / ed**0.5)
            self.Wk = rng.standard_normal((layers, heads, ed, qk_d)) * (1 / ed**0.5)
            self.Wv = rng.standard_normal((layers, heads, ed, v_d)) * (1 / ed**0.5)
            self.Wp = rng.standard_normal((layers, v_d*heads, ed)) * (1 / v_d**0.5)

@dataclass
class FinalBlockParams():
    dimensions: InitVar[ModelDimensions]
    allzero:InitVar[bool] = False
    Wu: np.ndarray = field(init=False)
    gama: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)
    
    def __post_init__(self,dimensions,allzero:bool=False):
        ed = dimensions.ed 
        vocab_size = dimensions.vocab_size
        if allzero:
            self.Wu = np.zeros((ed, vocab_size)) 
                    
            self.gama = np.zeros((1, ed))
            self.beta = np.zeros((1, ed))
            return
        else :
            rng = np.random.default_rng()
            self.Wu = rng.standard_normal((ed, vocab_size)) / ed**0.5
            
            self.gama = np.ones((1, ed))
            self.beta = np.zeros((1, ed))
            


@dataclass
class ModelTrainableParams():
    attention: AttentionParams = field(init=False)
    ffn: FFNParams = field(init=False)
    final:FinalBlockParams = field(init=False)
    w_emb: np.ndarray = field(init=False)

    def __init__(self,dimensions:ModelDimensions, allzero:bool=False) -> None:
        self.attention = AttentionParams(dimensions,allzero)
        self.ffn = FFNParams(dimensions,allzero)
        self.final = FinalBlockParams(dimensions,allzero)
        rng = np.random.default_rng()
        if allzero:
            self.w_emb= np.zeros((dimensions.vocab_size, dimensions.ed))
        else:
            self.w_emb= rng.standard_normal((dimensions.vocab_size, dimensions.ed)) * (1 / (dimensions.ed)**0.5)
        



@dataclass
class InitEmbeddingBlockRP():
    Xp: np.ndarray   = field(default_factory=lambda : np.array([]))
    input: np.ndarray = field(default_factory=lambda : np.array([]))
    


@dataclass
class FFNBlockRP():
    A0: list = field(default_factory=lambda :[])
    A1: list = field(default_factory=lambda :[])
    Z0: list = field(default_factory=lambda :[])
    Xfn: list = field(default_factory=lambda :[])
    Xhat: list = field(default_factory=lambda :[])
    Xlm: list = field(default_factory=lambda :[])
    Xlv: list = field(default_factory=lambda :[])
    input: list = field(default_factory=lambda :[])

@dataclass
class AttentionBlockRP:
    K: list = field(default_factory=lambda :[])
    Q: list = field(default_factory=lambda :[])
    V: list = field(default_factory=lambda :[])
    QK: list = field(default_factory=lambda :[])
    att: list = field(default_factory=lambda :[])
    Xhat: list = field(default_factory=lambda :[])
    Xlm: list = field(default_factory=lambda :[])
    Xlv: list = field(default_factory=lambda :[])
    combined_att: list = field(default_factory=lambda :[])
    Xfn: list = field(default_factory=lambda :[])
    input: list = field(default_factory=lambda :[])

@dataclass
class FinalBlockRP:
       Xhat: list = field(default_factory=lambda :[])
       Xlf: list = field(default_factory=lambda :[])
       Xlm: list = field(default_factory=lambda :[])
       Xlv: list = field(default_factory=lambda :[])
       Xv: list = field(default_factory=lambda :[])
       Xf: list = field(default_factory=lambda :[])
       input: list = field(default_factory=lambda :[])
    


@dataclass
class RuntimeParams():
    init : InitEmbeddingBlockRP
    attention : AttentionBlockRP
    ffn: FFNBlockRP
    final = FinalBlockRP

    def __init__(self) -> None:
        self.init = InitEmbeddingBlockRP()
        self.attention = AttentionBlockRP()
        self.ffn = FFNBlockRP()
        self.final = FinalBlockRP()


