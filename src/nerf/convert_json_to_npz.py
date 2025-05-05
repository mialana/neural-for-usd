# convert_json_to_npz.py
import os
import json
import numpy as np
from PIL import Image
from math import tan
from matplotlib import pyplot as plt
import argparse
import click

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-name", type=str, default="simpleCube")
    return parser.parse_args()

def main():
    horizontal_aperture = 25.955
    
    args = parse_args()
    asset_name = args.asset_name

    json_path = f"assets/{asset_name}/data/data.json"
    image_dir = f"assets/{asset_name}/data"
    output_npz = f"assets/{asset_name}/data/{asset_name}.npz"

    with open(json_path, "r") as f:
        meta = json.load(f)

    camera_angle_x = meta["camera_angle_x"]
    frames = meta["frames"]

    images, poses = [], []

    for frame in frames:
        file_path = os.path.join(image_dir, frame["file_path"])
        img = Image.open(file_path).convert("RGB")
        img = np.array(img) / 255.0
        if images and img.shape != images[0].shape:
            raise ValueError("All images must have the same dimensions!")
        images.append(img)
        poses.append(np.array(frame["transformMatrix"], dtype=np.float32))

    images = np.stack(images, axis=0)
    poses = np.stack(poses, axis=0)

    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]

    idx = 101
    click.secho(f"Test image: ORIGIN ({origins[idx][0]:.3f}, {origins[idx][1]:.3f}, {origins[idx][2]:.3f}) | DIRECTION ({dirs[idx][0]:.3f}, {dirs[idx][1]:.3f}, {dirs[idx][2]:.3f})", fg='yellow')

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

    ax.set_title("Sampled View Directions")
    plt.show()
    plt.close()

    plt.title("Test Image Preview")
    plt.imshow(images[idx])
    plt.show()
    plt.close()

    H, W = images.shape[1:3]
    focal = horizontal_aperture / (2 * tan(camera_angle_x * 0.5))

    np.savez(output_npz, images=images.astype(np.float32), poses=poses.astype(np.float32), focal=focal, H=H, W=W)
    click.secho(f"JSON to NPZ data transfer of {asset_name} successful. Saved to {output_npz}", fg='green')

if __name__ == "__main__":
    main()
