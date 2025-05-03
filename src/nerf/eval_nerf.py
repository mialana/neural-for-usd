NOVEL_POSITION = [-0.5, 1.0, 1.5]
TEST_IDX = 25

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from nerf import (
    init_models, get_rays, nerf_forward, near, far, kwargs_sample_hierarchical, kwargs_sample_stratified, n_samples_hierarchical, chunksize, VISUALS_DIR, CHECKPOINTS_DIR, DATA_PATH
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def generate_random_pose(radius=5.0, theta_range=(0, 2 * math.pi), phi_range=(math.pi/6, math.pi/3)):
    """
    Generates a random camera-to-world (c2w) matrix looking at the origin.

    - radius: distance from the origin
    - theta: azimuth angle (in the xy-plane)
    - phi: elevation angle (from vertical axis)

    Returns: torch.FloatTensor of shape (4, 4)
    """
    theta = torch.rand(1).item() * (theta_range[1] - theta_range[0]) + theta_range[0]
    phi = torch.rand(1).item() * (phi_range[1] - phi_range[0]) + phi_range[0]
    
    print("Theta degrees:", math.degrees(theta))
    print("Phi degrees:", math.degrees(phi))

    # Spherical to Cartesian
    cam_pos = torch.tensor([
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
        radius * math.sin(phi) * math.cos(theta)
    ])

    # Camera looks at origin
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    forward = (target - cam_pos)
    forward = forward / torch.norm(forward)
    up = torch.tensor([0, 0, 1], dtype=torch.float32)
    up = up / torch.norm(up)

    right = torch.cross(up, forward, dim=0)
    right = right / torch.norm(right)

    up = torch.cross(forward, right, dim=0)

    # Build rotation matrix
    R = torch.stack([right, up, -forward], dim=1)  # [3, 3]
    c2w = torch.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos

    return c2w.to(device)

def replicate_view(test_idx: int):
    test_img = torch.from_numpy(images_np[test_idx]).to(device)
    test_pose = torch.from_numpy(poses_np[test_idx]).to(device)
    focal = torch.tensor(focal_np).to(dtype=torch.float32, device=device)

    # Generate rays
    height, width = test_img.shape[:2]
    rays_o, rays_d = get_rays(height, width, focal, test_pose)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])

    # Run NeRF forward pass
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

    # Get the RGB image
    rgb = outputs["rgb_map"]
    rendered_img = rgb.reshape(height, width, 3).detach().cpu().numpy()

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_img.cpu().numpy())
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(rendered_img)
    axes[1].set_title("Rendered NeRF")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

def generate_novel_view(c2w: torch.tensor):
    focal = torch.tensor(focal_np).to(dtype=torch.float32, device=device)

    height, width = 100, 100
    rays_o, rays_d = get_rays(height, width, focal, c2w)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])

    # Run NeRF forward pass
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

    rgb = outputs["rgb_map"]
    image = rgb.reshape(height, width, 3).detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2])

    # === Left: Rendered Image ===
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image)
    ax_img.set_title("Rendered View")
    ax_img.axis("off")

    c2w = c2w.cpu().numpy()

    cam_pos = c2w[:3, 3]
    right = c2w[:3, 0]
    up = c2w[:3, 1]
    forward = -c2w[:3, 2]

    ax: Axes3D = fig.add_subplot(gs[1], projection='3d')

    ax.quiver(*cam_pos, *right, length=1.5, color='m', label='Novel Right')
    ax.quiver(*cam_pos, *up, length=1.5, color='c', label='Novel Up')
    ax.quiver(*cam_pos, *forward, length=1.5, color='y', label='Novel forward')

    ax.set_title("Camera Pose (Red=X, Green=Y, Blue=Forward)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=-90)
    ax.legend()

    lim = 5

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    plt.show()
    plt.close()

if __name__ == '__main__':
    # Init model
    model, fine_model, encode, encode_viewdirs, _, _ = init_models()

    data = np.load(DATA_PATH)

    # Load dataset
    images_np = data["images"]
    poses_np = data["poses"]
    focal_np = data["focal"].astype(np.float32)

    # replicate_view(TEST_IDX)

    while True:
        c2w = generate_random_pose()
        generate_novel_view(c2w)