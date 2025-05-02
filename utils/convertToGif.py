import imageio
import re
import os

# ChatGPT generated
def create_gif(image_folder, gif_path, duration=0.5):
    def sort_key(filename):
        timestamp = filename[84:-4]
        # timestamp = filename[-10:-4]
        return int(timestamp)

    filenames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')],
                       key=sort_key)
    images = [imageio.v2.imread(filename) for filename in filenames]
    imageio.mimsave(gif_path, images, duration=duration)

if __name__ == '__main__':
    # image_folder = "/Users/liu.amy05/Documents/Neural-for-USD/src/nerf/visuals/"
    # gif_path = "/Users/liu.amy05/Documents/Neural-for-USD/assets/japanesePlaneToy/jpt_figures.gif"

    image_folder = "/Users/liu.amy05/Documents/Neural-for-USD/assets/japanesePlaneToy/data/internalVal/"
    gif_path = "/Users/liu.amy05/Documents/Neural-for-USD/assets/japanesePlaneToy/jpt_usdSampling.gif"
    
    create_gif(image_folder, gif_path)
