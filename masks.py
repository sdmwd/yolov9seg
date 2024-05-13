import os
import shutil
import zipfile
from PIL import Image, ImageChops

def process_zip_files(folder_path, output_dir_global, output_dir_combined):
    os.makedirs(output_dir_global, exist_ok=True)
    os.makedirs(output_dir_combined, exist_ok=True)
    grouped_images = {}

    # Walk through the directory to find zip files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract images and process them
                    for member in zip_ref.namelist():
                        if member.endswith('.jpg'):
                            # Extract the file to temp directory for processing
                            zip_ref.extract(member, 'temp')
                            extracted_path = os.path.join('temp', member)
                            # Determine the group key
                            group_key = '_'.join(member.split('_')[:3])
                            if group_key not in grouped_images:
                                grouped_images[group_key] = {'global': None, 'images': []}
                            
                            if member.endswith('_global.jpg'):
                                grouped_images[group_key]['global'] = member
                                shutil.move(extracted_path, os.path.join(output_dir_global, member))
                            else:
                                grouped_images[group_key]['images'].append(extracted_path)

    # Combine the images for each group using pixel addition
    for group, info in grouped_images.items():
        if info['images'] and info['global']:  # Ensure there are images and a corresponding global image
            # Load the first image to establish the base
            base_image = Image.open(info['images'][0]).convert('RGBA')

            # Overlay other images onto the base image
            for img_path in info['images'][1:]:
                next_image = Image.open(img_path).convert('RGBA')
                base_image = ImageChops.add(base_image, next_image)

            # Convert to RGB before saving as JPEG
            base_image = base_image.convert('RGB')

            # Save the combined image using the global image name in the VEHICULE directory
            combined_image_path = os.path.join(output_dir_combined, info['global'])
            base_image.save(combined_image_path)

            # Clean up the extracted images to free up space
            for img_path in info['images']:
                os.remove(img_path)

    # Remove the temporary directory after processing
    shutil.rmtree('temp')

# Usage
process_zip_files('path/to/your/folders', 'path/to/__srcs__', 'path/to/VEHICULE')