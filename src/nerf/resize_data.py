# resize_data.py
from PIL import Image
import os
from send2trash import send2trash
import argparse
import click

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-name", type=str, default="simpleCube", help="Name of the asset in assets/")
    return parser.parse_args()

def main():
    args = parse_args()
    asset_name = args.asset_name

    input_dir = f"assets/{asset_name}/data/internalVal"
    output_dir = input_dir
    target_size = (100, 100)

    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    xStart = (width - height) / 2
                    xEnd = width - xStart
                    box = (xStart, 0, xEnd, height)
                    cropped_img = img.crop(box)
                    resized_img = cropped_img.resize(target_size, Image.LANCZOS)
                    resized_img.save(output_path)
                    print(f"Resized {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    click.secho(f"All training images of {asset_name} resized successfully!", fg='green')

if __name__ == "__main__":
    main()
