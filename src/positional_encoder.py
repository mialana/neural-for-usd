import torch
import os
import wget
import numpy as np
import matplotlib.pyplot as plt

deviceName: str = ""
if torch.cuda.is_available():
    deviceName = "cuda"
elif torch.backends.mps.is_available():
    deviceName = "mps"
else:
    deviceName = "cpu"

device: torch.device = torch.device(deviceName)

if not os.path.exists("src/tiny_nerf_data.npz"):
    wget.download(
        "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz", "src/"
    )

data = np.load("src/tiny_nerf_data.npz")
images = data["images"]
poses = data["poses"]
focal = data["focal"]

print(f"Images shape: {images.shape}")
print(f"Poses shape: {poses.shape}")
print(f"Focal length: {focal}")

height, width = images.shape[1:3]
near, far = 2.0, 6.0

n_training = 100
testimg_idx = 101
testimg, testpose = images[testimg_idx], poses[testimg_idx]

plt.imshow(testimg)
plt.savefig('src/test_image.png')
# plt.show()

print("Pose")
print(testpose)