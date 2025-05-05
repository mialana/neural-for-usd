from PIL import Image
import os
from send2trash import send2trash

asset_name = "simpleCube"
input_dir = f"assets/{asset_name}/data/internalVal"     # folder containing original images
output_dir = f"assets/{asset_name}/data/internalVal"  # folder to save resized images
target_size = (100, 100)       # new size (width, height)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all image files
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if len(filename) <= 7:
            send2trash(input_path)
            continue

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
