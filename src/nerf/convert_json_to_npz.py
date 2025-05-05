import os
import json
import numpy as np
from PIL import Image
from math import tan
from matplotlib import pyplot as plt

asset_name = "simpleCube"
# === Config ===
json_path = f"assets/{asset_name}/data/data.json"
image_dir = f"assets/{asset_name}/data"
output_npz = f"assets/{asset_name}/data/{asset_name}.npz"
img_extension = ".png"  # Change to ".jpg" if needed
horizontal_aperture = 25.955

# === Load JSON ===
with open(json_path, "r") as f:
    meta = json.load(f)

camera_angle_x = meta["camera_angle_x"]  # in radians
frames = meta["frames"]

# === Load images and poses ===
images = []
poses = []

for frame in frames:
    file_path = os.path.join(image_dir, frame["file_path"])
    print(file_path)
    img = Image.open(file_path).convert("RGB")
    img = np.array(img) / 255.0  # Normalize to [0, 1]

    if images and img.shape != images[0].shape:
        raise ValueError("All images must have the same dimensions!")

    images.append(img)
    poses.append(np.array(frame["transformMatrix"], dtype=np.float32))

images = np.stack(images, axis=0)  # Shape: (N, H, W, 3)
poses = np.stack(poses, axis=0)  # Shape: (N, 4, 4)

dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
origins = poses[:, :3, -1]

idx = 50
print(origins[idx])

ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
_ = ax.quiver(
  origins[..., 0].flatten(),
  origins[..., 1].flatten(),
  origins[..., 2].flatten(),
  dirs[..., 0].flatten(),
  dirs[..., 1].flatten(),
  dirs[..., 2][0].flatten(), length=1.0, normalize=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('z')
plt.show()

plt.imshow(images[idx])
plt.show()

H, W = images.shape[1:3]

# === Compute focal length ===
# Assumes horizontal FoV (angle_x)
focal = horizontal_aperture / (2 * tan(camera_angle_x * 0.5))

print(f"Loaded {len(images)} images with resolution {W}x{H}")
print(f"Computed focal length: {focal:.2f}")

# === Save to .npz ===
np.savez(
    output_npz,
    images=images.astype(np.float32),
    poses=poses.astype(np.float32),
    focal=focal,
    H=H,
    W=W,
)

print(f"Saved to {output_npz}")
