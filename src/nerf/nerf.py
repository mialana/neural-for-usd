asset_name = "japanesePlaneToy"

VISUALS_DIR = f"assets/{asset_name}/data/visuals"
CHECKPOINTS_DIR = f"assets/{asset_name}/data/checkpoints"
DATA_PATH = f"assets/{asset_name}/data/{asset_name}.npz"

import os
from typing import Optional, Tuple, List, Callable
import signal
import sys
from datetime import datetime

import numpy as np

import torch
from torch import nn

import matplotlib.pyplot as plt

from tqdm import trange

import logging

logging.basicConfig(format="%(message)s", level=logging.DEBUG)

debug = False
if not debug:
    logging.disable(logging.DEBUG)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

logging.info(device)

# For repeatability
seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)

### GLOBAL PARAMETERS

# Encoders
d_input = 3  # Number of input dimensions
n_freqs = 10  # Number of encoding functions for samples
log_space = True  # If set, frequencies scale in log space
use_viewdirs = True  # If set, use view direction as input
n_freqs_views = 4  # Number of encoding functions for views

# Stratified sampling
n_samples = 64  # Number of spatial samples per ray
perturb = True  # If set, applies noise to sample positions
inverse_depth = False  # If set, samples points linearly in inverse depth

# Model
d_filter = 128  # Dimensions of linear layer filters
n_layers = 2  # Number of layers in network bottleneck
skip = []  # Layers at which to apply input residual
use_fine_model = True  # If set, creates a fine model
d_filter_fine = 128  # Dimensions of linear layer filters of fine network
n_layers_fine = 6  # Number of layers in fine network bottleneck

# Hierarchical sampling
n_samples_hierarchical = 64  # Number of samples per ray
perturb_hierarchical = False  # If set, applies noise to sample positions

# Optimizer
lr = 5e-4  # Learning rate

# Training
n_iters = 10000
batch_size = 2**14  # Number of rays per gradient step (power of 2)
one_image_per_step = True # One image per gradient step (disables batching)
chunksize = 2**14  # Modify as needed to fit in GPU memory
center_crop = True  # Crop the center of image (one_image_per_)
center_crop_iters = 50  # Stop cropping center after this many epochs
eval_rate = 25  # Display test output every X epochs
display_rate = 50
use_warmup_stopper = False

# Early Stopping
warmup_iters = 100  # Number of iterations during warmup phase
warmup_min_fitness = 5.0  # Min val PSNR to continue training at warmup_iters
n_restarts = 10  # Number of times to restart if training stalls

# We bundle the kwargs for various functions to pass all at once.
kwargs_sample_stratified = {
    "n_samples": n_samples,
    "perturb": perturb,
    "inverse_depth": inverse_depth,
}
kwargs_sample_hierarchical = {"perturb": perturb}


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs
            )

        # Alternate sin and cos
        # Create functions that are applied to x for each freq band
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding functions to input position x.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class MLP(nn.Module):
    r"""
    Multilayer Perceptron module.
    """

    def __init__(
        self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        skip: Tuple[int] = (4,),
        d_viewdirs: Optional[int] = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)]
            + [
                (
                    nn.Linear(d_filter + self.d_input, d_filter)
                    if i in skip
                    else nn.Linear(d_filter, d_filter)
                )
                for i in range(n_layers - 1)
            ]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(
        self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if viewdirs is not None:
            if self.d_viewdirs is None:
                # Cannot use viewdirs if instantiated with d_viewdirs = None
                raise ValueError(
                    "Cannot input x_direction if d_viewdirs was not given."
                )
            else:  # have both viewdirs and dimension of viewdirs
                # Split alpha from network output
                alpha = self.alpha_out(x)

                # Pass through bottleneck to get RGB
                x = self.rgb_filters(x)
                x = torch.concat([x, viewdirs], dim=-1)
                x = self.act(self.branch(x))
                x = self.output(x)

                # Concatenate alphas to output
                x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)

        return x


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


def get_rays(
    height: int, width: int, focal_length: float, c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Find origin and direction of rays through every pixel and camera origin.
    """

    # Apply pinhole camera model to gather directions at each pixel
    # i is a [width, height] grid with x-values of pixel at coordinate
    # j is a [width, height] grid with x-values of pixel at coordinate
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(c2w),
        torch.arange(height, dtype=torch.float32).to(c2w),
        indexing="ij",
    )
    logging.debug("i Shape: ")
    logging.debug(i.shape)
    logging.debug("j Shape: ")
    logging.debug(j.shape)
    logging.debug("")

    # swaps the last two dimensions (should be just 2, to get i and j shaped [height, width].
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    logging.debug("i Shape Transposed: ")
    logging.debug(i.shape)
    logging.debug("j Shape Transposed: ")
    logging.debug(j.shape)
    logging.debug("")

    # Map to [(-1, -1), (1, 1)] and then NDC (scaled by focal length):
    #   x: (i - width/2) / focal
    #   y: -(j - height/2) / focal
    #   z: -1 (-1 is camera's forward)
    directions = torch.stack(
        [
            (i - width * 0.5) / focal_length,
            -(j - height * 0.5) / focal_length,
            -torch.ones_like(i),
        ],
        dim=-1,
    )

    # Convert to world coords
    # Apply camera pose to directions
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
    logging.debug("Ray Directions:")
    logging.debug("Shape: ")
    logging.debug(rays_d.shape)
    logging.debug("Sample: ")
    logging.debug(rays_d[height // 2, width // 2, :])
    logging.debug("")

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    logging.debug("Ray Origins:")
    logging.debug("Shape: ")
    logging.debug(rays_o.shape)
    logging.debug("Sample: ")
    logging.debug(rays_o[height // 2, width // 2, :])
    logging.debug("")

    return rays_o, rays_d


def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: Optional[bool] = True,
    inverse_depth: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Sample along ray from regularly-spaced bins.
    """

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def _sample_pdf(
    bins: torch.Tensor, weights: torch.Tensor, n_samples: int, perturb: bool = False
) -> torch.Tensor:
    r"""
    Apply inverse transform sampling to a weighted set of points.
    """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(
        weights + 1e-5, -1, keepdims=True
    )  # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
    cdf = torch.concat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [n_samples], device=cdf.device
        )  # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous()  # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(
        bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
    )

    # Convert samples to ray length.
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples  # [n_rays, n_samples]


def sample_hierarchical(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Apply hierarchical sampling to the rays.
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = _sample_pdf(
        z_vals_mid, weights[..., 1:-1], n_samples, perturb=perturb
    )
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    )  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    # Inclusive cumprod of [a, b, c] is [a, a*b, a*b*c]
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" (shift) the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    # Exclusive cumprod of [a, b, c] should be [1, a, a*b]
    cumprod[..., 0] = 1.0

    return cumprod


def raw2outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """

    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # concatenates a large “dummy” distance (1e10) for the last sample so that dists matches
    # [n_rays, n_samples] in size.
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    # standard “transmittance” formula from volume rendering says that the fraction of light
    # absorbed over a small distance d is 1 - e^(-density * distance)
    # alpha is effectively the opacity at each sample.
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    # "weights" indicates how much each sample contributes to the final color
    # cumprod_exclusive(1. - alpha) gives the transmittance from the start of the ray
    # up to (but not including) sample i.
    # Multiplying by alpha[i] yields the fraction of light that gets stopped at sample i.
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    # Compute weighted RGB map.
    # sigmoid squeezes into [0, 1] range
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    # Multiply each color by its weight, then sum over samples dim =-2
    # Gives you one final RGB color per ray
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    # Weighted sum of the z_vals (the sample positions) by weights.
    # Since weights is how likely the light ray ends at that position, this weighted sum
    # maps depth
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth (i.e. 1 / depth)
    # In many 3D tasks, we often use disparity (inverse depth) because it helps highlight
    # near vs. far geometry.
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )

    # Sum of weights along each ray. Tells how much of the ray is “opaque” vs. “transparent.”
    # In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    # If the background is white instead of black, any un-accumulated opacity
    # (which would otherwise be rendered black) is replaced with white.
    if white_bkgd:
        # if acc_map < 1, that leftover fraction is “background color.”
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


def get_chunks(inputs: torch.Tensor, chunksize: int = 2**15) -> List[torch.Tensor]:
    r"""
    Divide an input into chunks.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks(
    points: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15,
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify points to prepare for NeRF model.
    """
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points


def prepare_viewdirs_chunks(
    points: torch.Tensor,
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15,
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify viewdirs to prepare for NeRF model.
    """
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    coarse_model: nn.Module,
    kwargs_sample_stratified: dict = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    fine_model=None,
    viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    chunksize: int = 2**15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""
    Compute forward pass through model(s).
    """

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified
    )

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(
            query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize
        )
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {"z_vals_stratified": z_vals}

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(
            rays_o,
            rays_d,
            z_vals,
            weights,
            n_samples_hierarchical,
            **kwargs_sample_hierarchical,
        )

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(
                query_points, rays_d, viewdirs_encoding_fn, chunksize=chunksize
            )
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs["z_vals_hierarchical"] = z_hierarch
        outputs["rgb_map_0"] = rgb_map_0
        outputs["depth_map_0"] = depth_map_0
        outputs["acc_map_0"] = acc_map_0

    # Store outputs.
    outputs["rgb_map"] = rgb_map
    outputs["depth_map"] = depth_map
    outputs["acc_map"] = acc_map
    outputs["weights"] = weights
    return outputs


def find_checkpoint_indices(pattern: str):
    """
    Finds the last index in a sequence of filepaths where the file does not exist.

    Args:
        pattern (str): A string pattern for the filepaths,
                                 e.g., "data/file_{}.txt"
    """

    i = 0
    while True:
        next_filepath = pattern.format(i)
        if not os.path.exists(next_filepath):
            logging.info(f"Next checkpoint is {i}")
            logging.info(f"Latest checkpoint is {i-1}")
            if i == 0:
                logging.info(f"No previous checkpoint available")
                return next_filepath, None

            latest_filepath = pattern.format(i - 1)
            return next_filepath, latest_filepath
        else:
            i += 1


class NeRF:
    def __init__(self, checkpoint):
        self.reinit()

    def reinit(self):
        self.model = None
        self.fine_model = None

    def save(self):
        pass

    def load(self):
        pass

    def train(self):
        pass

    def visualize(self):
        pass

    def render_view(self):
        pass

    def render_video(self):
        pass


def train():
    r"""
    Launch training session for NeRF.
    """

    # Shuffle rays across all images.
    if not one_image_per_step:
        height, width = images.shape[1:3]
        logging.debug(height, " ", width)
        all_rays = torch.stack(
            [
                torch.stack(get_rays(height, width, focal, p), 0)
                for p in poses[:n_training]
            ],
            0,
        )
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in trange(n_iters):
        model.train()  # tells that you are training model, doesn't perform forward()
        fine_model.train()

        if one_image_per_step:
            # Randomly pick an image as the target.
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)
            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        else:
            # Random over all images.
            batch = rays_rgb[i_batch : i_batch + batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size
            # Shuffle after one epoch
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape([-1, 3])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = nerf_forward(
            rays_o,
            rays_d,
            near,
            far,
            encode,
            model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            n_samples_hierarchical=n_samples_hierarchical,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
            fine_model=fine_model,
            viewdirs_encoding_fn=encode_viewdirs,
            chunksize=chunksize,
        )

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                logging.info(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                logging.info(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
        rgb_predicted = outputs["rgb_map"]
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()  # rgb_predicted and target_img now have grad()
        optimizer.step()
        optimizer.zero_grad()
        psnr = -10.0 * torch.log10(loss)  # higher psnr is better quality
        train_psnrs.append(psnr.item())
        logging.info(f"Loss: {loss.item()}")
        logging.info(f"PSNR: {psnr.item()}")

        # Evaluate testimg at given display rate.
        if i % eval_rate == 0:
            model.eval()
            fine_model.eval()

            height, width = testimg.shape[:2]
            rays_o, rays_d = get_rays(height, width, focal, testpose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(
                rays_o,
                rays_d,
                near,
                far,
                encode,
                model,
                kwargs_sample_stratified=kwargs_sample_stratified,
                n_samples_hierarchical=n_samples_hierarchical,
                kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                fine_model=fine_model,
                viewdirs_encoding_fn=encode_viewdirs,
                chunksize=chunksize,
            )

            rgb_predicted = outputs["rgb_map"]
            loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
            val_psnr = -10.0 * torch.log10(loss)
            val_psnrs.append(val_psnr.item())
            iternums.append(i)
            logging.info(f"Val Loss: {loss.item()}")
            logging.info(f"Val PSNR: {val_psnr}")

            # Plot example outputs
            fig, ax = plt.subplots(
                1, 4, figsize=(24, 4), gridspec_kw={"width_ratios": [1, 1, 1, 3]}
            )
            ax[0].imshow(
                rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()
            )
            ax[0].set_title(f"Iteration: {i}")
            ax[1].imshow(testimg.detach().cpu().numpy())
            ax[1].set_title(f"Target")
            ax[2].plot(range(0, i + 1), train_psnrs, "r")
            ax[2].plot(iternums, val_psnrs, "b")
            ax[2].set_title("PSNR (train=red, val=blue")
            z_vals_strat = outputs["z_vals_stratified"].view((-1, n_samples))
            z_sample_strat = (
                z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
            )
            if "z_vals_hierarchical" in outputs:
                z_vals_hierarch = outputs["z_vals_hierarchical"].view(
                    (-1, n_samples_hierarchical)
                )
                z_sample_hierarch = (
                    z_vals_hierarch[z_vals_hierarch.shape[0] // 2]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                z_sample_hierarch = None
            _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
            ax[3].margins(0)
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if i % display_rate == 0:
                os.makedirs(VISUALS_DIR, exist_ok=True)
                plt.savefig(os.path.join(VISUALS_DIR, f"fig_{timestamp}.png"))
                plt.close()

            # plt.show()

        # Check PSNR for issues and stop if any are found.
        if i == warmup_iters - 1:
            if val_psnr < warmup_min_fitness:
                logging.info(
                    f"Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping..."
                )
                return False, train_psnrs, val_psnrs
        elif i < warmup_iters and i % 5 == 0:
            if use_warmup_stopper and warmup_stopper is not None and warmup_stopper(i, psnr):
                logging.info(
                    f"Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping..."
                )
                return False, train_psnrs, val_psnrs

    return True, train_psnrs, val_psnrs


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
            d_filter=d_filter_fine,
            skip=skip,
            d_viewdirs=d_viewdirs,
        )
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)  # lr: learning rate

    if latest_checkpoint_path is not None:
        model.load_state_dict(
            torch.load(f"{latest_checkpoint_path}/nerf.pt", weights_only=True)
        )
        fine_model.load_state_dict(
            torch.load(f"{latest_checkpoint_path}/nerf-fine.pt", weights_only=True)
        )
        optimizer.load_state_dict(
            torch.load(f"{latest_checkpoint_path}/optimizer.pt", weights_only=True)
        )

        logging.info(f"Loaded data from {latest_checkpoint_path}")

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper


# if not os.path.exists("tiny_nerf_data.npz"):
#     wget.download(
#         "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
#     )

data = np.load(DATA_PATH)
n_training = 100
testimg_idx = 101

near, far = 2.0, 6.0

# Gather as torch tensors
images = torch.from_numpy(data["images"][:n_training]).to(device)
poses = torch.from_numpy(data["poses"]).to(device)
focal = torch.from_numpy(data["focal"]).to(dtype=torch.float32).to(device)
testimg = torch.from_numpy(data["images"][testimg_idx]).to(device)
testpose = torch.from_numpy(data["poses"][testimg_idx]).to(device)

# Grab rays from sample image
height, width = images.shape[1:3]
with torch.no_grad():
    ray_origin, ray_direction = get_rays(height, width, focal, testpose)

logging.debug("Ray Origin")
logging.debug(ray_origin.shape)
logging.debug(ray_origin[height // 2, width // 2, :])
logging.debug("")

logging.debug("Ray Direction")
logging.debug(ray_direction.shape)
logging.debug(ray_direction[height // 2, width // 2, :])
logging.debug("")

# Draw stratified samples from example
rays_o = ray_origin.view([-1, 3])
rays_d = ray_direction.view([-1, 3])
perturb = True
inverse_depth = False
with torch.no_grad():
    pts, z_vals = sample_stratified(
        rays_o,
        rays_d,
        near,
        far,
        n_samples,
        perturb=perturb,
        inverse_depth=inverse_depth,
    )

logging.debug("Input Points")
logging.debug(pts.shape)


logging.debug("")
logging.debug("Distances Along Ray")
logging.debug(z_vals.shape)

y_vals = torch.zeros_like(z_vals)

_, z_vals_unperturbed = sample_stratified(
    rays_o, rays_d, near, far, n_samples, perturb=False, inverse_depth=inverse_depth
)

# Create encoders for points and view directions
encoder = PositionalEncoder(3, 10)
viewdirs_encoder = PositionalEncoder(3, 4)

# Grab flattened points and view directions
pts_flattened = pts.reshape(-1, 3)
viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
flattened_viewdirs = viewdirs[:, None, ...].expand(pts.shape).reshape((-1, 3))

# Encode inputs
encoded_points = encoder(pts_flattened)
encoded_viewdirs = viewdirs_encoder(flattened_viewdirs)

logging.info("Encoded Points")
logging.info(encoded_points.shape)
logging.info(
    (torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points))
)
logging.info("")

logging.info(encoded_viewdirs.shape)
logging.info("Encoded Viewdirs")
logging.info(
    (
        torch.min(encoded_viewdirs),
        torch.max(encoded_viewdirs),
        torch.mean(encoded_viewdirs),
    )
)
logging.info("")


def signal_handler(sig, frame):
    logging.info("You pressed Ctrl+C!")
    os.makedirs(next_checkpoint_path, exist_ok=True)

    torch.save(model.state_dict(), f"{next_checkpoint_path}/nerf.pt")
    torch.save(fine_model.state_dict(), f"{next_checkpoint_path}/nerf-fine.pt")
    torch.save(optimizer.state_dict(), f"{next_checkpoint_path}/optimizer.pt")

    logging.info(f"Saved to {next_checkpoint_path}")
    sys.exit(0)

next_checkpoint_path, latest_checkpoint_path = find_checkpoint_indices(
    os.path.join(CHECKPOINTS_DIR, "checkpoint_{}")
)

if __name__ == '__main__':
    # Run training session(s)
    for i in range(n_restarts):
        model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = (
            init_models()
        )

        if i == 0:
            signal.signal(signal.SIGINT, signal_handler)

        success, train_psnrs, val_psnrs = train()
        if success and val_psnrs[-1] >= warmup_min_fitness:
            logging.info("Training successful!")
            break
        else:
            logging.info(f"Restart no.{i}")

    logging.info("")
    logging.info(f"Done!")

    signal.raise_signal(signal.SIGINT)  # write to file
