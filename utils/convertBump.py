from PIL import Image
import numpy as np
import subprocess

import os

# Load the normal map (RGB) and bump map (grayscale)
normal_map = Image.open('./textures/plane_toy_Plane_toy_Normal.png').convert('RGB')
bump_map = Image.open('./textures/plane_toy_Plane_toy_Bump.png').convert('I;16')

normal_map = np.array(normal_map, dtype=np.float32) / 255.0  # Normalize the normal map to [0, 1]
bump_map = np.array(bump_map, dtype=np.int16)  / 65535.0  # Normalize the bump map to [0, 1]

# Prepare output image array (same size as the normal map)

# Loop through each pixel of the normal map and bump map
height, width, _ = normal_map.shape

bump_strength = 1.0

output_normal_map = np.zeros_like(normal_map)


hRange = int((height - 1) / 4)
wRange = int((width - 1) / 4)


# Process each texel in the height map (excluding the borders)
for y in range(1, hRange):
    for x in range(1, wRange):
        Hg = bump_map[y, x]   # Current texel height
        Hr = bump_map[y, x+1] # Right texel height
        Ha = bump_map[y-1, x] # Above texel height

        # Compute the difference vectors
        diff_right = np.array([1, 0, Hr - Hg])
        diff_up = np.array([0, 1, Ha - Hg])

        # Compute the normal vector as the cross product of the difference vectors
        normal = np.cross(diff_right, diff_up)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        height_diff = np.clip(np.abs(Hr - Hg) +  np.abs(Ha - Hg), 0.0, 1.0)

        print(height_diff)
        
        # Scale bump strength based on height difference (more bump effect where there's more height variation)
        scaled_bump_strength = (1 - height_diff)
        # print(bump_strength, scaled_bump_strength)

        normal[2] = (normal[2] + 1) * 0.5

        # print(normal[0] + 1 / 2.0)

        perturbed_normal = (normal * scaled_bump_strength)

        perturbed_normal = perturbed_normal / np.linalg.norm(perturbed_normal)

        perturbed_normal = np.clip(perturbed_normal, 0.0, 1.0)
        # Store the normal in the normal map
        output_normal_map[y, x] = perturbed_normal

# Map normal map components to the [0, 1] range for RGB storage
# output_normal_map = (output_normal_map + 1) / 2.0

normal_map_uint8 = (output_normal_map * 255).astype(np.uint8)

normal_map_image = Image.fromarray(normal_map_uint8)

normal_map_image.save('./textures/plane_toy_Plane_toy_BumpNorTest.png')


def split_image_into_blocks(image, num_blocks):
    # Split the image height into blocks for parallel processing
    block_height = image.shape[0] // num_blocks
    blocks = []

    for i in range(num_blocks):
        start_row = i * block_height
        end_row = start_row + block_height if i < num_blocks - 1 else image.shape[0]
        blocks.append((start_row, end_row))
    
    return blocks

def process_image_in_parallel(normal_map, bump_map, bump_strength, num_processes=4):
    # Split the image into blocks and process them in parallel
    blocks = split_image_into_blocks(normal_map, num_processes)
    processes = []
    output_paths = []

    for i, (start_row, end_row) in enumerate(blocks):
        output_path = f'output_block_{i}.png'
        output_paths.append(output_path)

        # Call the subprocess to process the block
        process = subprocess.run("")
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    # Combine the results from all blocks
    final_image = np.zeros_like(normal_map)
    for i, (start_row, end_row) in enumerate(blocks):
        output_block = np.array(Image.open(output_paths[i]))
        final_image[start_row:end_row] = output_block

    # Save the final output image
    final_image = (final_image * 255).astype(np.uint8)
    final_image_image = Image.fromarray(final_image)
    final_image_image.save('final_normal_map.png')

    # Clean up temporary output blocks
    for output_path in output_paths:
        os.remove(output_path)