import os
from PIL import Image
import numpy as np

# Define the directory names
original_dir = '/home/zihao/Desktop/cellcounts/IDCIA v2/images'
processed_dir = '/home/zihao/Desktop/cellcounts/LearningToCountEverything/cell384'

# Create a new directory for processed images if it doesn't exist
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Get all the TIFF files in the original directory
tiff_files = [f for f in os.listdir(original_dir) if f.endswith('.tiff')]

# Process each TIFF file
for tiff_file in tiff_files:
    # Open the image using PIL
    with Image.open(os.path.join(original_dir, tiff_file)) as img:
        # Resize the image
        img_resized = img.resize((512, 384))  # width x height

        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')

        img_resized = Image.eval(img_resized, lambda x: min(x * 3, 255))

        # Save the resized image in JPEG format
        # The file name is the same but with the .jpg extension
        img_resized.save(os.path.join(processed_dir, tiff_file.replace('.tiff', '.jpg')), 'JPEG')

# Return the path of the processed directory and the list of processed files
processed_files = os.listdir(processed_dir)
processed_dir, processed_files


