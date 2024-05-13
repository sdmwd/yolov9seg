import os
import shutil
from PIL import Image

def move_globals_and_combine_images(source_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to hold lists of images for each group
    grouped_images = {}

    # Traverse through the directories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Full path to the current file
            full_path = os.path.join(root, file)

            # Check if the file is a global image
            if file.endswith('_global.jpg'):
                shutil.move(full_path, os.path.join(output_dir, file))
            else:
                # Process other images and group them
                parts = file.split('_')
                if len(parts) >= 3 and parts[2].isdigit():  # Ensure the naming pattern fits
                    group_key = f'{parts[0]}_{parts[1]}_{parts[2]}'
                    if group_key not in grouped_images:
                        grouped_images[group_key] = []
                    grouped_images[group_key].append(full_path)

    # Now combine the images for each group
    for group, images in grouped_images.items():
        # Load images
        loaded_images = [Image.open(img) for img in images]
        # Assuming vertical combination
        total_height = sum(img.height for img in loaded_images)
        max_width = max(img.width for img in loaded_images)
        new_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in loaded_images:
            new_image.paste(img, (0, y_offset))
            y_offset += img.height

        # Save the combined image
        new_image.save(os.path.join(output_dir, f'{group}.jpg'))

# Use the function with appropriate paths
move_globals_and_combine_images('path/to/your/folders', 'path/to/__srcs__')