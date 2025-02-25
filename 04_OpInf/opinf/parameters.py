from dataclasses import dataclass, field
import numpy as np


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class Params(metaclass=Singleton):
    # input data
    file_suffix: str = '_flow_rate_down'
    case: str='case_1'

    # sample data size
    step_t_sampling: int = 1
    step_z_sampling: int = 1

    # model
    model_structure: str = 'AH'   # 'A', 'H', 'C', 'B'
    stability: str = 'global'  # local, global, none
    basis: str = 'POD'  # POD, NL-POD, AM
    regularization_H: float = 1e-4 # regularization
    scaling: bool = True
    smoothing: bool = False
    true_derivatives: bool = False

    # ROM size
    tolerance: float = 1e-3
    thresholds: np.ndarray = field(
        default_factory=lambda: np.arange(0.99, 0.9999, 0.001))
    r_X: int = 0  # basis size for conversion
    r_T: int = 0  # basis size for temperature
    ROM_order: int = 0  # summed up order of the reduced system
    input_dim: int = 0  # dimension of the input (how many trajectories are we having)

    # saving and plotting
    save_results: bool = True
    plot_results: bool = True

    # fitting
    batch_size: int = 1000000  # batch size
    num_epochs: int = 100000  # number of epochs

    # adam
    adam_lr: float = 0.001
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0

    # lr_schedule
    lr_schedule_step_factor: float = 10000
    lr_schedule_mode: str = "triangular2"
    lr_schedule_cycle_momentum: bool = False
    lr_schedule_base_lr: float = 1e-5
    lr_schedule_max_lr: float = 5E-1
