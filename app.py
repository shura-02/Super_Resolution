import os
import cv2
import numpy as np
from google.colab import files
import shutil
import glob
from google.colab.patches import cv2_imshow

# Clone Real-ESRGAN and enter the Real-ESRGAN directory
!git clone https://github.com/xinntao/Real-ESRGAN.git
%cd Real-ESRGAN

# Set up the environment
!pip install basicsr
!pip install facexlib
!pip install gfpgan
!pip install -r requirements.txt
!python setup.py develop
!pip install gradio

import gradio as gr
import cv2
import os
import glob
import shutil
import numpy as np


from flask import Flask, request


%cd /content
upload_folder = 'upload'
splits_folder = 'splits'
result_folder = 'results'
output_folder = 'output'
direct_floder = 'direct'
if os.path.isdir(upload_folder):
    shutil.rmtree(upload_folder)
if os.path.isdir(splits_folder):
        shutil.rmtree(splits_folder)
if os.path.isdir(result_folder):
    shutil.rmtree(result_folder)
if os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
if os.path.isdir(direct_floder):
    shutil.rmtree(direct_floder)
os.mkdir(upload_folder)
os.mkdir(splits_folder)
os.mkdir(result_folder)
os.mkdir(output_folder)
os.mkdir(direct_floder)

def split_image(splits_folder, image_path, n):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    split_height = height // int(n ** 0.5)
    split_width = width // int(n ** 0.5)

    # Split the image into n**0.5 rows and n**0.5 columns
    splits = []
    for i in range(int(n ** 0.5)):
        for j in range(int(n ** 0.5)):
            split = img[i * split_height: (i + 1) * split_height, j * split_width: (j + 1) * split_width, :]
            splits.append(split)
            # Save the splits in the splits folder
            cv2.imwrite(os.path.join(splits_folder, f'split_{i}_{j}.png'), split)

    return splits_folder


def process_image(input_folder, output_folder):
    input_list = sorted(glob.glob(os.path.join(input_folder, '*')))

    for idx, input_path in enumerate(input_list):
        # Apply ESRGAN to each split
        output_path = os.path.join(output_folder)
        !python /content/Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i $input_path -o $output_path --outscale 3.5 --face_enhance

        # Move the processed image to the 'results' folder
        shutil.move(output_path, os.path.join(output_folder))

    return output_folder


def merge_images(splits_folder, result_folder, n, apply_median_filter=False):
    # Load the processed split images
    splits_list = sorted(glob.glob(os.path.join(splits_folder, '*')))
    merged_rows = []

    if len(splits_list) == 0:
        return None  # No splits available

    for i in range(int(n ** 0.5)):
        if (i * int(n ** 0.5)) >= len(splits_list):
            break  # No more splits to merge
        row_splits = [cv2.imread(split) for split in splits_list[i * int(n ** 0.5): (i + 1) * int(n ** 0.5)]]
        merged_row = np.hstack(row_splits)
        merged_rows.append(merged_row)

    if len(merged_rows) == 0:
        return None  # No rows to merge

    merged = np.vstack(merged_rows)

    if apply_median_filter:
        # Apply median filter to the merged image
        merged = cv2.medianBlur(merged, 3)

    # Save the final output image
    cv2.imwrite(os.path.join(result_folder, 'final_output.png'), merged)

    return "/content/output/final_output.png"


def direct_method(input_image, method, n, apply_filter):
    save_path = "/content/upload/image.jpg"
    bgr_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr_image)

    if method == 'Direct method':
        # Run the first notebook code for direct method
        !python /content/Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 3.5 --face_enhance
        return "/content/results/image_out.jpg"
    elif method == "Split method":
        n = int(n)
        splits_folder = '/content/splits'
        folder = split_image(splits_folder, save_path, n)
        output_folder = "/content/direct"
        out = process_image(folder, output_folder)
        output = '/content/output'
        final = merge_images(out, output, n, apply_median_filter=apply_filter)
        return final
    
    
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    # Your Gradio code goes here
    title = "REAL ESRGAN SUPER RESOLUTION"
    description = "Choose the super resolution method:"
    inputs = [
    gr.inputs.Image(type="numpy", label="Input Image"),
    gr.inputs.Dropdown(["Direct method", "Split method"], label="Method"),
    gr.inputs.Dropdown([4, 9, 16, 25, 36, 49, 64], label="Number of Splits"),
    gr.inputs.Checkbox(label="Apply Median Filter")
    ]

    outputs = gr.outputs.Image(type="numpy", label="Output Image")

    interface = gr.Interface(
    fn=direct_method,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    )
    
    interface.queue()
    interface.launch(debug=True)
    
    # Retrieve the input from the request
    input_data = request.json['input']
    # Process the input and generate the output
    output = process_input(input_data)
    # Return the output as JSON response
    return {'output': output}

if __name__ == '__main__':
    app.run()
