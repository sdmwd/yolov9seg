import os

def convert_segmentation_to_bbox(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            index = parts[0]
            coordinates = list(map(int, parts[1:]))
            x_coords = coordinates[0::2]
            y_coords = coordinates[1::2]
            xmin = min(x_coords)
            xmax = max(x_coords)
            ymin = min(y_coords)
            ymax = max(y_coords)
            bbox_line = f"{index} {xmin} {ymin} {xmax} {ymax}\n"
            outfile.write(bbox_line)

# Example usage:
input_dir = 'path/to/input/files'
output_dir = 'path/to/output/files'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(input_dir):
    if file_name.endswith('.txt'):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)
        convert_segmentation_to_bbox(input_file_path, output_file_path)
