import torch
import numpy as np
import matplotlib.pyplot as plt

from nerf import (
    init_models, get_rays, nerf_forward, near, far, kwargs_sample_hierarchical, kwargs_sample_stratified, n_samples_hierarchical, chunksize, VISUALS_DIR, CHECKPOINTS_DIR, DATA_PATH
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

data = np.load(DATA_PATH)

# Load dataset
images_np = data["images"]
poses_np = data["poses"]
focal_np = data["focal"].astype(np.float32)

# Pick a test index
test_idx = 50
test_img = torch.from_numpy(images_np[test_idx]).to(device)
test_pose = torch.from_numpy(poses_np[test_idx]).to(device)
focal = torch.tensor(focal_np).to(dtype=torch.float32, device=device)

# Init model
model, fine_model, encode, encode_viewdirs, _, _ = init_models()

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