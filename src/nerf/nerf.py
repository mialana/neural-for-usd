import os
from typing import Optional, Tuple, Callable
import signal
import sys
from datetime import datetime
import time
import threading

import numpy as np

import torch
from torch import nn

import matplotlib.pyplot as plt

from tqdm import trange

import click

import argparse

from train_nerf import (sample_stratified, sample_hierarchical, prepare_chunks, prepare_viewdirs_chunks, raw2outputs, get_rays, crop_center, plot_samples)
from train_nerf import (PositionalEncoder, MLP, EarlyStopping)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

click.secho(device, fg='green')

### GLOBAL STATE VARIABLES
VISUALS_DIR, CHECKPOINTS_DIR, DATA_PATH
images, focal, poses, n_training, testimg, testpose
model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper
interrupted = False # state variable that tracks if a interruption was recieved and stops the model after the current step finishes

### GLOBAL PARAMETERS

# Render
near, far = 1.0, 10.0

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
show_figures = False
show_figures_duration = 10 # show figures for 10 seconds, then close

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-name", type=str, default="simpleCube")
    parser.add_argument("--show-figures", action='store_true', default=False)
    return parser.parse_args()

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
            click.secho(f"Next checkpoint is {i}", fg='green')
            click.secho(f"Latest checkpoint is {i-1}", fg='green')
            if i == 0:
                click.secho(f"No previous checkpoint available", fg='green')
                return next_filepath, None

            latest_filepath = pattern.format(i - 1)
            return next_filepath, latest_filepath
        else:
            i += 1

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

def train():
    r"""
    Launch training session for NeRF.
    """

    # Shuffle rays across all images.
    if not one_image_per_step:
        height, width = images.shape[1:3]
        click.secho(f"Image height: {height}, Images width: {width}", fg='magenta')
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

    global interrupted # use interrupted global variable to check if NeRF needs to exit on this iteration

    try:
        for i in trange(n_iters):
            if interrupted:
                return False, train_psnrs, val_psnrs

            model.train()  # tells that you are training model, doesn't perform forward()
            fine_model.train()
            optimizer.train()

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

            # Run one iteration of NeRF and get the rendered RGB image.
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
                    click.secho(f"! [Numerical Alert] {k} contains NaN.", fg='green')
                if torch.isinf(v).any():
                    click.secho(f"! [Numerical Alert] {k} contains Inf.", fg='green')

            # Backprop!
            rgb_predicted = outputs["rgb_map"]
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            loss.backward()  # rgb_predicted and target_img now have grad()
            optimizer.step()
            optimizer.zero_grad()
            psnr = -10.0 * torch.log10(loss)  # higher psnr is better quality
            train_psnrs.append(psnr.item())
            click.secho(f"Loss: {loss.item()}", fg='green')
            click.secho(f"PSNR: {psnr.item()}", fg='green')

            # Evaluate testimg at given display rate.
            if i % eval_rate == 0:
                model.eval()
                fine_model.eval()
                optimizer.eval()

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
                click.secho(f"Val Loss: {loss.item()}", fg='green')
                click.secho(f"Val PSNR: {val_psnr}", fg='green')

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

                    if show_figures:
                        def _delay_plt_close():
                            time.sleep(show_figures_duration)
                            plt.close()
                        threading.Thread(target=_delay_plt_close).start()
                        plt.show()
                        plt.pause(show_figures_duration)

            # Check PSNR for issues and stop if any are found.
            if i == warmup_iters - 1:
                if val_psnr < warmup_min_fitness:
                    click.secho(
                        f"Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...", fg='red'
                    )
                    return False, train_psnrs, val_psnrs
            elif i < warmup_iters and i % 5 == 0:
                if use_warmup_stopper and warmup_stopper is not None and warmup_stopper(i, psnr):
                    click.secho(
                        f"Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...", fg='red'
                    )
                    return False, train_psnrs, val_psnrs
                
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

    return True, train_psnrs, val_psnrs


def init_models(checkpoints_dir):
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

    _, latest_checkpoint_path = find_checkpoint_indices(
        os.path.join(checkpoints_dir, "checkpoint_{}")
    )

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

        click.secho(f"Loaded data from {latest_checkpoint_path}", fg='green')

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper

def shutdown_nerf():
    global model, fine_model, optimizer

    click.secho("Shutting down NeRF gracefully...", fg='blue')

    plt.close('all')

    click.secho("All matplotlib figures closed", fg='blue')

    next_checkpoint_path, _ = find_checkpoint_indices(
        os.path.join(CHECKPOINTS_DIR, "checkpoint_{}")
    )

    os.makedirs(next_checkpoint_path, exist_ok=True)

    try:
        torch.save(model.state_dict(), f"{next_checkpoint_path}/nerf.pt")
        torch.save(fine_model.state_dict(), f"{next_checkpoint_path}/nerf-fine.pt")
        torch.save(optimizer.state_dict(), f"{next_checkpoint_path}/optimizer.pt")

        click.secho(f"Models saved to {next_checkpoint_path}", fg='blue')
    except Exception as e:
        print(f"Failed to save one or more models: {e}")

    click.secho("NeRF shut down successfully. Thanks!", fg='blue')

def signal_handler(sig, frame):
    global interrupted

    click.secho("Interrupt received. Finishing current step and exiting gracefully...", fg='red')
    interrupted = True

def main():
    global show_figures, VISUALS_DIR, CHECKPOINTS_DIR, DATA_PATH
    global images, focal, poses, n_training, near, far, testimg, testpose
    global model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper

    args = parse_args()
    asset_name = args.asset_name

    show_figures = args.show_figures
    
    VISUALS_DIR = f"assets/{asset_name}/data/visuals"
    CHECKPOINTS_DIR = f"assets/{asset_name}/data/checkpoints"
    DATA_PATH = f"assets/{asset_name}/data/{asset_name}.npz"

    # For repeatability
    seed = 3407
    torch.manual_seed(seed)
    np.random.seed(seed)

    # if not os.path.exists("tiny_nerf_data.npz"):
    #     wget.download(
    #         "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    #     )

    data = np.load(DATA_PATH)
    n_training = 100
    testimg_idx = 101

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

    # click.secho("Ray Origin", fg='magenta')
    # click.secho(ray_origin.shape, fg='magenta')
    # click.secho(f"{ray_origin[height // 2, width // 2, :]}\n", fg='magenta')

    # click.secho("Ray Direction", fg='magenta')
    # click.secho(ray_direction.shape, fg='magenta')
    # click.secho(f"{ray_direction[height // 2, width // 2, :]}\n", fg='magenta')

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

    click.secho("Input Points", fg='magenta')
    click.secho(f"{pts.shape}\n", fg='magenta')

    click.secho("Distances Along Ray", fg='magenta')
    click.secho(f"{z_vals.shape}\n", fg='magenta')

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

    click.secho("Encoded Points", fg='magenta')
    click.secho(encoded_points.shape, fg='magenta')
    click.secho(
        f"{(torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points))}\n", fg='magenta'
    )

    click.secho(encoded_viewdirs.shape, fg='magenta')
    click.secho("Encoded Viewdirs")
    click.secho(f"{(
            torch.min(encoded_viewdirs),
            torch.max(encoded_viewdirs),
            torch.mean(encoded_viewdirs),
        )}\n", fg='magenta'
    )

    # Run training session(s)
    for i in range(n_restarts):
        model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper  = init_models(CHECKPOINTS_DIR)

        if i == 0:
            signal.signal(signal.SIGINT, signal_handler)

        success, train_psnrs, val_psnrs = train()
        if success and val_psnrs[-1] >= warmup_min_fitness:
            click.secho("Training successful!", fg='green')
            break
        if not success and interrupted:
            click.secho("Training interrupted successfully!", fg='yellow')
            break
        else:
            click.secho(f"Restart no.{i}", fg='red')

    shutdown_nerf()

if __name__ == "__main__":
    main()