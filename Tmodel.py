"""
Tmodel.py
---------
Dataclasses holding model dimensions, trainable parameters, and runtime
activation caches used during forward and backward passes.
"""

from dataclasses import dataclass, field, InitVar
import numpy as np


@dataclass
class ModelDimensions:
    ed: int
    qk_d: int
    v_d: int
    p_d: int          # heads * v_d
    ffn_wd: int
    seq_len: int
    layers: int
    heads: int
    vocab_size: int


@dataclass
class FFNParams:
    """Weights and biases for feed-forward blocks across all layers."""

    dimensions: InitVar[ModelDimensions]
    allzero: InitVar[bool] = False

    W0:   np.ndarray = field(init=False)
    B0:   np.ndarray = field(init=False)
    W1:   np.ndarray = field(init=False)
    B1:   np.ndarray = field(init=False)
    gama: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)

    def __post_init__(self, dimensions: ModelDimensions, allzero: bool = False):
        rng = np.random.default_rng(56)
        L, E, F = dimensions.layers, dimensions.ed, dimensions.ffn_wd

        if allzero:
            self.gama = np.zeros((L, E),dtype=np.float32)
            self.beta = np.zeros((L, E),dtype=np.float32)
            self.W0   = np.zeros((L, E, F),dtype=np.float32)
            self.B0   = np.zeros((L, 1, F),dtype=np.float32)
            self.W1   = np.zeros((L, F, E),dtype=np.float32)
            self.B1   = np.zeros((L, 1, E),dtype=np.float32)
        else:
            self.gama = np.ones((L, E) ,dtype=np.float32)
            self.beta = np.zeros((L, E),dtype=np.float32)
            self.W0   = rng.standard_normal((L, E, F),dtype=np.float32) * (2 / E) ** 0.5
            self.B0 = np.zeros((L, 1, F), dtype=np.float32)
            self.W1   = rng.standard_normal((L, F, E),dtype=np.float32) * (2 / F) ** 0.5
            self.B1 = np.zeros((L, 1, E), dtype=np.float32)


@dataclass
class AttentionParams:
    """Weights for multi-head attention blocks across all layers."""

    dimensions: InitVar[ModelDimensions]
    allzero: InitVar[bool] = False

    Wq:   np.ndarray = field(init=False)
    Wk:   np.ndarray = field(init=False)
    Wv:   np.ndarray = field(init=False)
    Wp:   np.ndarray = field(init=False)
    gama: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)

    def __post_init__(self, dimensions: ModelDimensions, allzero: bool = False):
        rng = np.random.default_rng(57)
        L, H = dimensions.layers, dimensions.heads
        E, QK, V = dimensions.ed, dimensions.qk_d, dimensions.v_d

        if allzero:
            self.gama = np.zeros((L, E),dtype=np.float32)
            self.beta = np.zeros((L, E),dtype=np.float32)
            self.Wq   = np.zeros((L, H, E, QK),dtype=np.float32)
            self.Wk   = np.zeros((L, H, E, QK),dtype=np.float32)
            self.Wv = np.zeros((L, H, E, V), dtype=np.float32)
            self.Wp = np.zeros((L, V * H, E), dtype=np.float32)
        else:
            self.gama = np.ones((L, E),dtype=np.float32)
            self.beta = np.zeros((L, E),dtype=np.float32)
            self.Wq   = rng.standard_normal((L, H, E, QK),dtype=np.float32) * (1 / (E/H) ** 0.5)
            self.Wk   = rng.standard_normal((L, H, E, QK),dtype=np.float32) * (1 / (E/H) ** 0.5)
            self.Wv   = rng.standard_normal((L, H, E, V),dtype=np.float32)  * (1 / (E/H) ** 0.5)
            self.Wp   = rng.standard_normal((L, V * H, E),dtype=np.float32) * (1 / (V*H) ** 0.5)


@dataclass
class FinalBlockParams:
    """Weights for final layer-norm and unembedding projection."""

    dimensions: InitVar[ModelDimensions]
    allzero: InitVar[bool] = False

    Wu:   np.ndarray = field(init=False)
    gama: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)

    def __post_init__(self, dimensions: ModelDimensions, allzero: bool = False):
        E, V = dimensions.ed, dimensions.vocab_size

        if allzero:
            self.Wu   = np.zeros((E, V),dtype=np.float32)
            self.gama = np.zeros((1, E),dtype=np.float32)
            self.beta = np.zeros((1, E),dtype=np.float32)
        else:
            rng = np.random.default_rng(56)
            self.Wu   = rng.standard_normal((E, V),dtype=np.float32) / E ** 0.5
            self.gama = np.ones((1, E),dtype=np.float32)
            self.beta = np.zeros((1, E), dtype=np.float32)


@dataclass
class ModelTrainableParams:
    """Container for all trainable model parameters."""

    attention: AttentionParams  = field(init=False)
    ffn:       FFNParams        = field(init=False)
    final:     FinalBlockParams = field(init=False)
    w_emb:     np.ndarray       = field(init=False)

    def __init__(self, dimensions: ModelDimensions, allzero: bool = False) -> None:
        self.attention = AttentionParams(dimensions, allzero)
        self.ffn       = FFNParams(dimensions, allzero)
        self.final     = FinalBlockParams(dimensions, allzero)

        rng = np.random.default_rng(56)
        if allzero:
            self.w_emb = np.zeros((dimensions.vocab_size, dimensions.ed),dtype=np.float32)
        else:
            self.w_emb = (
                rng.standard_normal(
                    (dimensions.vocab_size, dimensions.ed), dtype=np.float32)
                * (1 / dimensions.ed ** 0.5)
            )


@dataclass
class InitEmbeddingBlockRP:
    Xp:    np.ndarray = field(default_factory=lambda: np.array([]))
    input: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class FFNBlockRP:
    A0:    list = field(default_factory=list)
    A1:    list = field(default_factory=list)
    Z0:    list = field(default_factory=list)
    Xfn:   list = field(default_factory=list)
    Xhat:  list = field(default_factory=list)
    Xlm:   list = field(default_factory=list)
    Xlv:   list = field(default_factory=list)
    input: list = field(default_factory=list)


@dataclass
class AttentionBlockRP:
    K:            list = field(default_factory=list)
    Q:            list = field(default_factory=list)
    V:            list = field(default_factory=list)
    QK:           list = field(default_factory=list)
    att:          list = field(default_factory=list)
    Xhat:         list = field(default_factory=list)
    Xlm:          list = field(default_factory=list)
    Xlv:          list = field(default_factory=list)
    combined_att: list = field(default_factory=list)
    Xfn:          list = field(default_factory=list)
    input:        list = field(default_factory=list)


@dataclass
class FinalBlockRP:
    Xhat:  list = field(default_factory=list)
    Xlf:   list = field(default_factory=list)
    Xlm:   list = field(default_factory=list)
    Xlv:   list = field(default_factory=list)
    Xv:    list = field(default_factory=list)
    Xf:    list = field(default_factory=list)
    input: list = field(default_factory=list)


@dataclass
class RuntimeParams:
    """Container for intermediate activations saved during a forward pass."""

    init:      InitEmbeddingBlockRP
    attention: AttentionBlockRP
    ffn:       FFNBlockRP
    final:     FinalBlockRP

    def __init__(self) -> None:
        self.init      = InitEmbeddingBlockRP()
        self.attention = AttentionBlockRP()
        self.ffn       = FFNBlockRP()
        self.final     = FinalBlockRP()

@dataclass
class DropOutMasks:
    """Container for dropout masks"""   
    Xp: np.ndarray = field(default_factory=lambda: np.array([]))
    attention: list = field(default_factory=list)
    ffn: list = field(default_factory=list)
