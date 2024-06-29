import os
import random
import cv2
import numpy as np
import imutils

def augment_image(image):
    # Random rotation
    angle = random.uniform(-3, 3)
    image = imutils.rotate_bound(image, angle)

    # Random zoom
    zoom_factor = random.uniform(0.97, 1.03)
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    image = cv2.resize(image, (new_w, new_h))
    if zoom_factor < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        image = cv2.copyMakeBorder(image, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_CONSTANT)
    else:
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        image = image[crop_h:crop_h + h, crop_w:crop_w + w]

    # Adjust brightness
    brightness_factor = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    return image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to read image: {input_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented_image = augment_image(img)
            base_name, ext = os.path.splitext(filename)
            new_filename = f"{base_name}_aug{ext}"
            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            print(f"Saved augmented image: {output_path}")

# Define input and output folders
input_folder = r'chest_xray\chest_xray\train\TEMP'
output_folder = r'chest_xray\chest_xray\train\TEMP'

# Process images
process_images(input_folder, output_folder)

