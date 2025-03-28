import os
import json
import numpy as np
from PIL import Image
from math import tan

# === Config ===
json_path = "data/data.json"
image_dir = "data/"
output_npz = "data.npz"
img_extension = ".png"  # Change to ".jpg" if needed

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
poses = np.stack(poses, axis=0)    # Shape: (N, 4, 4)

H, W = images.shape[1:3]

# === Compute focal length ===
# Assumes horizontal FoV (angle_x)
focal = 0.5 * W / tan(0.5 * camera_angle_x)

print(f"Loaded {len(images)} images with resolution {W}x{H}")
print(f"Computed focal length: {focal:.2f}")

# === Save to .npz ===
np.savez(
    output_npz,
    images=images.astype(np.float32),
    poses=poses.astype(np.float32),
    focal=focal,
    H=H,
    W=W
)

print(f"Saved to {output_npz}")
