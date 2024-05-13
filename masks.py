import os
import shutil
from PIL import Image, ImageChops

def move_globals_and_combine_images(source_dir, output_dir_global, output_dir_combined):
    # Ensure output directories exist
    os.makedirs(output_dir_global, exist_ok=True)
    os.makedirs(output_dir_combined, exist_ok=True)

    # Dictionary to hold lists of images for each group and the name of their corresponding global image
    grouped_images = {}

    # Traverse through the directories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Full path to the current file
            full_path = os.path.join(root, file)

            # Check if the file is a global image
            if file.endswith('_global.jpg'):
                # Move global images to the __srcs__ directory
                shutil.move(full_path, os.path.join(output_dir_global, file))
                # Extract key base (BRAND_number_x)
                group_key = '_'.join(file.split('_')[:3])
                # Initialize grouping dictionary
                grouped_images[group_key] = {'global': file, 'images': []}

            else:
                # Split filename to check format and ensure correct processing
                parts = file.split('_')
                if len(parts) > 3 and parts[2].isdigit():  # Verify it matches expected pattern
                    group_key = '_'.join(parts[:3])
                    # Check if this key already exists, meaning we have a corresponding global image
                    if group_key in grouped_images:
                        grouped_images[group_key]['images'].append(full_path)

    # Combine the images for each group using pixel addition
    for group, info in grouped_images.items():
        if info['images']:  # Make sure there are actually images to combine
            base_image = Image.open(info['images'][0]).convert('RGBA')  # Ensure images are in the same mode
            for img_path in info['images'][1:]:
                next_image = Image.open(img_path).convert('RGBA')
                base_image = ImageChops.add(base_image, next_image, scale=2.0, offset=0)  # Scale and offset can adjust blending

            # Save the combined image using the name of the global image in the VEHICULE directory
            base_image.save(os.path.join(output_dir_combined, info['global']))

# Example usage of the function with appropriate paths
move_globals_and_combine_images('path/to/your/folders', 'path/to/__srcs__', 'path/to/VEHICULE')