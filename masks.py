def move_globals_and_combine_images(source_dir, output_dir_global, output_dir_combined):
    # Ensure output directories exist
    os.makedirs(output_dir_global, exist_ok=True)
    os.makedirs(output_dir_combined, exist_ok=True)

    # Dictionary to hold lists of images for each group
    grouped_images = {}

    # Traverse through the directories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Full path to the current file
            full_path = os.path.join(root, file)

            # Check if the file ends with '_global.jpg' or other formats
            if file.endswith('.jpg'):
                # Extract the group key (BRAND_number_x)
                group_key = '_'.join(file.split('_')[:3])
                
                if group_key not in grouped_images:
                    grouped_images[group_key] = {'global': None, 'images': []}
                
                # Identify global image and separate handling
                if file.endswith('_global.jpg'):
                    grouped_images[group_key]['global'] = file
                    # Move global images directly to the __srcs__ directory
                    shutil.move(full_path, os.path.join(output_dir_global, file))
                else:
                    # Add image to the list in its group
                    grouped_images[group_key]['images'].append(full_path)

    # Combine the images for each group using pixel addition
    for group, info in grouped_images.items():
        if info['images'] and info['global']:  # Ensure there are images and a corresponding global image
            # Load the first image to establish the base
            base_image = Image.open(info['images'][0]).convert('RGBA')

            # Overlay other images onto the base image
            for img_path in info['images'][1:]:
                next_image = Image.open(img_path).convert('RGBA')
                base_image = ImageChops.add(base_image, next_image)

            # Save the combined image using the global image name in the VEHICULE directory
            combined_image_path = os.path.join(output_dir_combined, info['global'])
            base_image.save(combined_image_path)