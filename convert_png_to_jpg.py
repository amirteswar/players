import os
from PIL import Image

# Define the directory containing PNG images
input_directory = r'E:\gpu\Sphase\samedog6\data\football_database'
output_directory = r'E:\gpu\Sphase\samedog6\data\football_database_jpg'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def convert_png_to_jpg(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                png_path = os.path.join(root, file)
                jpg_filename = os.path.splitext(file)[0] + '.jpg'
                jpg_path = os.path.join(output_dir, jpg_filename)
                
                try:
                    with Image.open(png_path) as img:
                        # Convert to RGB if necessary
                        if img.mode == 'RGBA' or img.mode == 'P':
                            img = img.convert('RGB')
                        img.save(jpg_path, 'JPEG')
                    
                    print(f"Converted: {png_path} to {jpg_path}")
                except Exception as e:
                    print(f"Error converting {png_path}: {e}")

# Perform the conversion
convert_png_to_jpg(input_directory, output_directory)
