import os
from dataclasses import dataclass
import signal
from datetime import datetime

import numpy as np

import torch

import matplotlib.pyplot as plt

from tqdm import trange

import click

import argparse

from nerf import (
    plot_samples,
    crop_center,
    sample_stratified,
    get_rays,
    nerf_forward
)
from nerf import PositionalEncoder, MLP, EarlyStopping

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

click.secho(device, fg="green")

# if not os.path.exists("tiny_nerf_data.npz"):
#     wget.download(
#         "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
#     )

interrupted = False  # state variable that tracks if interruption was recieved. stops train loop after curr iter

# GLOBAL PARAMETERS

# Render
near, far = 1.0, 10.0

n_training = 100
testimg_idx = 101

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
one_image_per_step = True  # One image per gradient step (disables batching)
chunksize = 2**14  # Modify as needed to fit in GPU memory
center_crop = True  # Crop the center of image (one_image_per_)
center_crop_iters = 50  # Stop cropping center after this many epochs
eval_rate = 25  # Test results every X epochs
create_figure_rate = 50
use_warmup_stopper = False
show_figures = False
show_figures_duration = 10  # show figures for 10 seconds, then close

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


@dataclass
class NeRFState:
    VISUALS_DIR = ""
    CHECKPOINTS_DIR = ""
    DATA_PATH = ""

    images = None
    poses = None
    focal = None
    testimg = None
    testpose = None
    model = None
    fine_model = None
    optimizer = None
    encode = None
    encode_viewdirs = None
    warmup_stopper = None

    show_figures = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-name", type=str, default="simpleCube")
    parser.add_argument("--show-figures", action="store_true", default=False)
    return parser.parse_args()


def signal_handler(sig, frame):
    global interrupted

    click.secho(
        "Interrupt received. Finishing current step and exiting gracefully...", fg="red"
    )
    interrupted = True


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
            click.secho(f"Next checkpoint is {i}", fg="green")
            click.secho(f"Latest checkpoint is {i-1}", fg="green")
            if i == 0:
                click.secho("No previous checkpoint available", fg="green")
                return next_filepath, None

            latest_filepath = pattern.format(i - 1)
            return next_filepath, latest_filepath
        else:
            i += 1


def _debug_rays(state: NeRFState):
    # Grab rays from sample image
    height, width = state.images.shape[1:3]
    with torch.no_grad():
        ray_origin, ray_direction = get_rays(height, width, state.focal, state.testpose)

    click.secho("Ray Origin", fg="magenta")
    click.secho(ray_origin.shape, fg="magenta")
    click.secho(f"{ray_origin[height // 2, width // 2, :]}\n", fg="magenta")

    click.secho("Ray Direction", fg="magenta")
    click.secho(ray_direction.shape, fg="magenta")
    click.secho(f"{ray_direction[height // 2, width // 2, :]}\n", fg="magenta")

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

    click.secho("Input Points", fg="magenta")
    click.secho(f"{pts.shape}\n", fg="magenta")

    click.secho("Distances Along Ray", fg="magenta")
    click.secho(f"{z_vals.shape}\n", fg="magenta")

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

    click.secho("Encoded Points", fg="magenta")
    click.secho(encoded_points.shape, fg="magenta")
    click.sechoclick.secho(
        f"{torch.min(encoded_points)}, {torch.max(encoded_points)}, {torch.mean(encoded_points)}\n",
        fg="magenta",
    )

    click.secho(encoded_viewdirs.shape, fg="magenta")
    click.secho("Encoded Viewdirs")
    click.secho(
        f"{torch.min(encoded_viewdirs)}, {torch.max(encoded_viewdirs)}, {torch.mean(encoded_viewdirs)}\n",
        fg="magenta",
    )


def init_models(state: NeRFState):
    r"""
    Initialize models, encoders, and optimizer for NeRF training.
    """

    # Encoders
    state.encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    state.encode = lambda x: state.encoder(x)

    # View direction encoders
    d_viewdirs = None

    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(
            d_input, n_freqs_views, log_space=log_space
        )
        state.encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output

    # Models
    state.model = MLP(
        state.encoder.d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip,
        d_viewdirs=d_viewdirs,
    )
    state.model.to(device)
    model_params = list(state.model.parameters())  # all weights and bias

    state.fine_model = None
    if use_fine_model:
        state.fine_model = MLP(
            state.encoder.d_output,
            n_layers=n_layers,
            d_filter=d_filter_fine,
            skip=skip,
            d_viewdirs=d_viewdirs,
        )
        state.fine_model.to(device)
        model_params = model_params + list(state.fine_model.parameters())

    # Optimizer
    state.optimizer = torch.optim.Adam(model_params, lr=lr)  # lr: learning rate

    _, latest_checkpoint_path = find_checkpoint_indices(
        os.path.join(state.CHECKPOINTS_DIR, "checkpoint_{}")
    )

    if latest_checkpoint_path is not None:
        state.model.load_state_dict(
            torch.load(f"{latest_checkpoint_path}/nerf.pt", weights_only=True)
        )
        state.fine_model.load_state_dict(
            torch.load(f"{latest_checkpoint_path}/nerf-fine.pt", weights_only=True)
        )
        state.optimizer.load_state_dict(
            torch.load(f"{latest_checkpoint_path}/optimizer.pt", weights_only=True)
        )

        click.secho(f"Loaded data from {latest_checkpoint_path}", fg="green")

    # Early Stopping
    state.warmup_stopper = EarlyStopping(patience=50)


def train(state: NeRFState):
    r"""
    Launch training session for NeRF.
    """

    # Shuffle rays across all images.
    if not one_image_per_step:
        height, width = state.images.shape[1:3]
        click.secho(f"Image height: {height}, Images width: {width}", fg="magenta")
        all_rays = torch.stack(
            [
                torch.stack(get_rays(height, width, state.focal, p), 0)
                for p in state.poses[:state.n_training]
            ],
            0,
        )

        rays_rgb = torch.cat([all_rays, state.images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []

    global interrupted  # use interrupted global variable to check if NeRF needs to exit on this iteration

    for i in trange(n_iters):
        if interrupted:
            click.secho(f"Stopped at iteration {i}", fg='magenta')
            return False, train_psnrs, val_psnrs

        state.model.train()  # tells that you are training model, doesn't perform forward()
        if state.fine_model:
            state.fine_model.train()

        if one_image_per_step:
            # Randomly pick an image as the target.
            target_img_idx = np.random.randint(state.images.shape[0])
            target_img = state.images[target_img_idx].to(device)
            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = state.poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, state.focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        else:
            # Random over all images.
            batch = rays_rgb[i_batch: i_batch + batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size
            # Shuffle after one epoch
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape([-1, 3])

        # Run one iteration of NeRF and get the rendered RGB image.
        outputs = nerf_forward(
            rays_o,
            rays_d,
            near,
            far,
            state.encode,
            state.model,
            curr_kwargs_sample_stratified=kwargs_sample_stratified,
            curr_n_samples_hierarchical=n_samples_hierarchical,
            curr_kwargs_sample_hierarchical=kwargs_sample_hierarchical,
            curr_fine_model=state.fine_model,
            viewdirs_encoding_fn=state.encode_viewdirs,
            curr_chunksize=chunksize,
        )

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                click.secho(f"! [Numerical Alert] {k} contains NaN.", fg="green")
            if torch.isinf(v).any():
                click.secho(f"! [Numerical Alert] {k} contains Inf.", fg="green")

        # Backprop!
        rgb_predicted = outputs["rgb_map"]
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()  # rgb_predicted and target_img now have grad()
        state.optimizer.step()
        state.optimizer.zero_grad()
        psnr = -10.0 * torch.log10(loss)  # higher psnr is better quality
        train_psnrs.append(psnr.item())
        click.secho(f"Loss: {loss.item()}", fg="green")
        click.secho(f"PSNR: {psnr.item()}", fg="green")

        # Evaluate testimg at given display rate.
        if i % eval_rate == 0:
            state.model.eval()
            if state.fine_model:
                state.fine_model.eval()

            height, width = state.testimg.shape[:2]
            rays_o, rays_d = get_rays(height, width, state.focal, state.testpose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(
                rays_o,
                rays_d,
                near,
                far,
                state.encode,
                state.model,
                curr_kwargs_sample_stratified=kwargs_sample_stratified,
                curr_n_samples_hierarchical=n_samples_hierarchical,
                curr_kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                curr_fine_model=state.fine_model,
                viewdirs_encoding_fn=state.encode_viewdirs,
                curr_chunksize=chunksize,
            )

            rgb_predicted = outputs["rgb_map"]
            loss = torch.nn.functional.mse_loss(
                rgb_predicted, state.testimg.reshape(-1, 3)
            )
            val_psnr = -10.0 * torch.log10(loss)
            val_psnrs.append(val_psnr.item())
            iternums.append(i)
            click.secho(f"Evaluated Loss: {loss.item()}", fg="cyan")
            click.secho(f"Evaluated PSNR: {val_psnr}", fg="cyan")

            if i % create_figure_rate == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(state.VISUALS_DIR, exist_ok=True)
                figure_path = os.path.join(state.VISUALS_DIR, f"fig_{timestamp}.png")

                click.secho(f"Creating figure now. Saving to {figure_path}", fg="cyan")

                # Plot example outputs
                fig, ax = plt.subplots(
                    1, 4, figsize=(24, 4), gridspec_kw={"width_ratios": [1, 1, 1, 3]}
                )
                ax[0].imshow(
                    rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()
                )
                ax[0].set_title(f"Iteration: {i}")
                ax[1].imshow(state.testimg.detach().cpu().numpy())
                ax[1].set_title("Target")
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

                plt.savefig(figure_path)

                if state.show_figures:
                    plt.show()
                    plt.close()

        # Check PSNR for issues and stop if any are found.
        if i == warmup_iters - 1:
            if val_psnr < warmup_min_fitness:
                click.secho(
                    f"Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...",
                    fg="red",
                )
                return False, train_psnrs, val_psnrs
        elif i < warmup_iters and i % 5 == 0:
            if (
                use_warmup_stopper and state.warmup_stopper is not None and state.warmup_stopper(i, psnr)
            ):
                click.secho(
                    f"Train PSNR flatlined at {psnr} for {state.warmup_stopper.patience} iters. Stopping...",
                    fg="red",
                )
                return False, train_psnrs, val_psnrs

    return True, train_psnrs, val_psnrs


def shutdown_nerf(state: NeRFState):
    click.secho("Shutting down NeRF gracefully...", fg="blue")

    plt.close("all")

    click.secho("All matplotlib figures closed", fg="blue")

    next_checkpoint_path, _ = find_checkpoint_indices(
        os.path.join(state.CHECKPOINTS_DIR, "checkpoint_{}")
    )

    os.makedirs(next_checkpoint_path, exist_ok=True)

    try:
        torch.save(state.model.state_dict(), f"{next_checkpoint_path}/nerf.pt")
        torch.save(state.fine_model.state_dict(), f"{next_checkpoint_path}/nerf-fine.pt")
        torch.save(state.optimizer.state_dict(), f"{next_checkpoint_path}/optimizer.pt")

        click.secho(f"Models saved to {next_checkpoint_path}", fg="blue")
    except Exception as e:
        print(f"Failed to save one or more models: {e}")

    click.secho("NeRF shut down successfully.", fg="blue")


def main():
    state = NeRFState()

    args = parse_args()
    asset_name = args.asset_name

    state.show_figures = args.show_figures

    state.VISUALS_DIR = f"assets/{asset_name}/data/visuals"
    state.CHECKPOINTS_DIR = f"assets/{asset_name}/data/checkpoints"
    state.DATA_PATH = f"assets/{asset_name}/data/{asset_name}.npz"

    # For repeatability
    seed = 3407
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(state.DATA_PATH)

    # Gather as torch tensors
    state.images = torch.from_numpy(data["images"][:n_training]).to(device)
    state.poses = torch.from_numpy(data["poses"]).to(device)
    state.focal = torch.from_numpy(data["focal"]).to(dtype=torch.float32).to(device)
    state.testimg = torch.from_numpy(data["images"][testimg_idx]).to(device)
    state.testpose = torch.from_numpy(data["poses"][testimg_idx]).to(device)

    # Run training session(s)
    for i in range(n_restarts):
        init_models(state)

        if i == 0:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        success, train_psnrs, val_psnrs = train(state)
        if success and val_psnrs[-1] >= warmup_min_fitness:
            click.secho("Training successful!", fg="green")
            break
        if not success and interrupted:
            click.secho("Training interrupted successfully!", fg="yellow")
            break
        else:
            click.secho(f"Restart no.{i}", fg="red")

    shutdown_nerf(state)


if __name__ == "__main__":
    main()
