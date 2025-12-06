from typing import Any, List, Optional

from zencfg import ConfigBase


class CardiacModel(ConfigBase):
    model_arch: str = "fnocardiac"
    n_modes: List[int] = [48, 48]
    hidden_channels: int = 128
    n_layers: int = 6
    domain_padding: float = 0.0625
    norm: Optional[Any] = None
    fno_skip: str = "linear"
    use_channel_mlp: int = 0
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: int = 0
    separable: bool = False
    factorization: Optional[Any] = None
    rank: float = 1.0
    freq_bands: int = 3
    classifier: bool = True
    n_classes: int = 2


class Opt(ConfigBase):
    n_epochs: int = 200
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-5
    mixed_precision: bool = False
    scheduler_T_max: int = 200
    scheduler_patience: int = 10
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.5


class Data(ConfigBase):
    folder: str = "~/data/cardiac/"
    batch_size: int = 16
    n_train: int = 2000
    train_resolution: int = 128
    n_tests: List[int] = [200, 100]
    test_resolutions: List[int] = [128, 256]
    test_batch_sizes: List[int] = [16, 8]
    encode_input: bool = True
    encode_output: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    model: CardiacModel = CardiacModel()
    opt: Opt = Opt()
    data: Data = Data()
