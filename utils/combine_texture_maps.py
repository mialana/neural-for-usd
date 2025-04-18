import numpy as np
from PIL import Image

def call(n1, n2):
    # Define your float4 and float3 as NumPy arrays
    def float4(x, y, z, w):
        return np.array([x, y, z, w])

    def float3(x, y, z):
        return np.array([x, y, z])

    # Example values for n1 and n2 (replace with your actual vectors)
    # n1 = np.array([1, 1, 1, 1])  # Example n1
    # n2 = np.array([1, 1, 1])     # Example n2

    # Apply the operations on n1 and n2 as per the shader code
    n1 = [n1[0], n1[1], n1[2], n1[2]] * float4(2, 2, 2, -2) + float4(-1, -1, -1, 1)
    n2 = n2 * 2 - 1

    # Perform the dot products, correctly indexing the components of n1
    r_x = np.dot([n1[2], n1[0], n1[0]], [n2[0], n2[1], n2[2]])  # dot(n1.zxx, n2.xyz)
    r_y = np.dot([n1[1], n1[2], n1[1]], [n2[0], n2[1], n2[2]])  # dot(n1.yzy, n2.xyz)
    r_z = np.dot([n1[0], n1[1], n1[3]], [-n2[0], -n2[1], -n2[2]]) # dot(n1.xyw, -n2.xyz)

    # Combine into r
    r = float3(r_x, r_y, r_z)

    # Normalize r
    r_normalized = r / np.linalg.norm(r)

    return r_normalized

new_normal = Image.open('./plane_toy_Plane_toy_normal.png').convert('RGB')
orig_normal = Image.open('./origNormal.png').convert('RGB')

new_normal_map = np.array(new_normal, dtype=np.float32) / 255.0 
orig_normal_map = np.array(orig_normal, dtype=np.float32) / 255.0

height, width, _ = new_normal_map.shape

hRange = int((height))
wRange = int((width))

output_normal_map = np.zeros_like(new_normal_map)


# Process each texel in the height map (excluding the borders)
for y in range(hRange):
    for x in range(wRange):
        newfloat4 = np.append(new_normal_map[y, x], 1.)
        origfloat4 = np.append(orig_normal_map[y, x], 1.)
        output_normal_map[y, x] = call(newfloat4, origfloat4)

output_normal_map = (output_normal_map + 1) / 2.0
print(output_normal_map)

normal_map_uint8 = (output_normal_map * 255).astype(np.uint8)

normal_map_image = Image.fromarray(normal_map_uint8)

normal_map_image.save('./textures/result.png')