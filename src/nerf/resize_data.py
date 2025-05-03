from PIL import Image
import os

asset_name = "campfire"
input_dir = f"assets/{asset_name}/data/internalVal"     # folder containing original images
output_dir = f"assets/{asset_name}/data/internalVal"  # folder to save resized images
target_size = (100, 100)       # new size (width, height)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all image files
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(output_path)
                print(f"Resized {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
