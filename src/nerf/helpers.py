from positional_encoder import PositionalEncoder
from mlp import MLP

from typing import Optional

from imports import *


def plot_samples(
    z_vals: torch.Tensor,
    z_hierarch: Optional[torch.Tensor] = None,
    ax: Optional[np.ndarray] = None,
):
    r"""
    Plot stratified and (optional) hierarchical samples.
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, "b-o")
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, "r-o")
    ax.set_ylim([-1, 2])
    ax.set_title("Stratified  Samples (blue) and Hierarchical Samples (red)")
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
    return ax


def crop_center(img: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    r"""
    Crop center square from image.
    """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]


class EarlyStopping:
    r"""
    Early stopping helper based on fitness criterion.
    """

    def __init__(self, patience: int = 30, margin: float = 1e-4):
        self.best_fitness = 0.0  # In our case PSNR
        self.best_iter = 0
        self.margin = margin
        self.patience = patience or float(
            "inf"
        )  # epochs to wait after fitness stops improving to stop

    def __call__(self, iter: int, fitness: float):
        r"""
        Check if criterion for stopping is met.
        """
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop


def init_models():
    r"""
    Initialize models, encoders, and optimizer for NeRF training.
    """
    # Encoders
    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)

    # View direction encoders
    encode_viewdirs = None
    d_viewdirs = None

    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(
            d_input, n_freqs_views, log_space=log_space
        )
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output

    # Models
    model = MLP(
        encoder.d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip,
        d_viewdirs=d_viewdirs,
    )
    model.to(device)
    model_params = list(model.parameters())  # all weights and bias

    fine_model = None
    if use_fine_model:
        fine_model = MLP(
            encoder.d_output,
            n_layers=n_layers,
            d_filter=d_filter,
            skip=skip,
            d_viewdirs=d_viewdirs,
        )
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)  # lr: learning rate

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper
