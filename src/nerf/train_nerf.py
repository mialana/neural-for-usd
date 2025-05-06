"""Contains functions and classes for the main `nerf.py` module"""

from typing import Optional, Tuple, List, Callable

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


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
            [nn.Linear(self.d_input, d_filter)] + [
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


def _sample_hierarchical(
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


def _cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimicking the functionality of
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


def _raw2outputs(
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
    weights = alpha * _cumprod_exclusive(1.0 - alpha + 1e-10)

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
    _ = 1.0 / torch.max(
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


def _get_chunks(inputs: torch.Tensor, curr_chunksize: int = 2**15) -> List[torch.Tensor]:
    r"""
    Divide an input into chunks.
    """
    return [inputs[i: i + curr_chunksize] for i in range(0, inputs.shape[0], curr_chunksize)]


def _prepare_chunks(
    points: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    curr_chunksize: int = 2**15,
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify points to prepare for NeRF model.
    """
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = _get_chunks(points, curr_chunksize=curr_chunksize)
    return points


def _prepare_viewdirs_chunks(
    points: torch.Tensor,
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    curr_chunksize: int = 2**15,
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify viewdirs to prepare for NeRF model.
    """
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = _get_chunks(viewdirs, curr_chunksize=curr_chunksize)
    return viewdirs


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

    # swaps the last two dimensions (should be just 2, to get i and j shaped [height, width].
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)

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

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    curr_near: float,
    curr_far: float,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    coarse_model: nn.Module,
    curr_kwargs_sample_stratified: dict = None,
    curr_n_samples_hierarchical: int = 0,
    curr_kwargs_sample_hierarchical: dict = None,
    curr_fine_model=None,
    viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    curr_chunksize: int = 2**15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""
    Compute forward pass through model(s).
    """

    # Set no kwargs if none are given.
    if curr_kwargs_sample_stratified is None:
        curr_kwargs_sample_stratified = {}
    if curr_kwargs_sample_hierarchical is None:
        curr_kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, curr_near, curr_far, **curr_kwargs_sample_stratified
    )

    # Prepare batches.
    batches = _prepare_chunks(query_points, encoding_fn, curr_chunksize=curr_chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = _prepare_viewdirs_chunks(
            query_points, rays_d, viewdirs_encoding_fn, curr_chunksize=curr_chunksize
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
    rgb_map, depth_map, acc_map, weights = _raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {"z_vals_stratified": z_vals}

    # Fine model pass.
    if curr_n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = _sample_hierarchical(
            rays_o,
            rays_d,
            z_vals,
            weights,
            curr_n_samples_hierarchical,
            **curr_kwargs_sample_hierarchical,
        )

        # Prepare inputs as before.
        batches = _prepare_chunks(query_points, encoding_fn, curr_chunksize=curr_chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = _prepare_viewdirs_chunks(
                query_points, rays_d, viewdirs_encoding_fn, curr_chunksize=curr_chunksize
            )
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        curr_fine_model = (
            curr_fine_model if curr_fine_model is not None else coarse_model
        )
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(curr_fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = _raw2outputs(
            raw, z_vals_combined, rays_d
        )

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


if __name__ == "__main__":
    pass
