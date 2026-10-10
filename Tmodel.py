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
        rng = np.random.default_rng()
        L, E, F = dimensions.layers, dimensions.ed, dimensions.ffn_wd

        if allzero:
            self.gama = np.zeros((L, E))
            self.beta = np.zeros((L, E))
            self.W0   = np.zeros((L, E, F))
            self.B0   = np.zeros((L, 1, F))
            self.W1   = np.zeros((L, F, E))
            self.B1   = np.zeros((L, 1, E))
        else:
            self.gama = np.ones((L, E))
            self.beta = np.zeros((L, E))
            self.W0   = rng.standard_normal((L, E, F)) * (2 / E) ** 0.5
            self.B0   = np.zeros((L, 1, F))
            self.W1   = rng.standard_normal((L, F, E)) * (2 / F) ** 0.5
            self.B1   = np.zeros((L, 1, E))


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
        rng = np.random.default_rng()
        L, H = dimensions.layers, dimensions.heads
        E, QK, V = dimensions.ed, dimensions.qk_d, dimensions.v_d

        if allzero:
            self.gama = np.zeros((L, E))
            self.beta = np.zeros((L, E))
            self.Wq   = np.zeros((L, H, E, QK))
            self.Wk   = np.zeros((L, H, E, QK))
            self.Wv   = np.zeros((L, H, E, V))
            self.Wp   = np.zeros((L, V * H, E))
        else:
            self.gama = np.ones((L, E))
            self.beta = np.zeros((L, E))
            self.Wq   = rng.standard_normal((L, H, E, QK)) * (1 / E ** 0.5)
            self.Wk   = rng.standard_normal((L, H, E, QK)) * (1 / E ** 0.5)
            self.Wv   = rng.standard_normal((L, H, E, V))  * (1 / E ** 0.5)
            self.Wp   = rng.standard_normal((L, V * H, E)) * (1 / V ** 0.5)


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
            self.Wu   = np.zeros((E, V))
            self.gama = np.zeros((1, E))
            self.beta = np.zeros((1, E))
        else:
            rng = np.random.default_rng()
            self.Wu   = rng.standard_normal((E, V)) / E ** 0.5
            self.gama = np.ones((1, E))
            self.beta = np.zeros((1, E))


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

        rng = np.random.default_rng()
        if allzero:
            self.w_emb = np.zeros((dimensions.vocab_size, dimensions.ed))
        else:
            self.w_emb = (
                rng.standard_normal((dimensions.vocab_size, dimensions.ed))
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