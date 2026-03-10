import cv2
import numpy as np
import os

# DELETE THE IMPORT LINE THAT WAS HERE

def preprocess_image(image_path): # Use 'preprocess_image' to match app.py
    if not os.path.exists(image_path):
        return np.zeros((512, 512, 3), dtype='float32')

    # 1. Load and Crop
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros((512, 512, 3), dtype='float32'), 0.0
        
    # Simple auto-crop: find non-zero pixels and crop to that bounding box
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 10 # Threshold to ignore near-black borders
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        image = image[y0:y1, x0:x1]

    # 2. Quality Check (Laplacian Variance)
    # Higher variance = sharper image. Below 100 is usually too hazy.
    quality_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 3. Resize & Color Space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    # 4. Refined Ben Graham's Method
    # Reducing sigma to 5-7 helps prevent artifacts in hazy images
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 7), -4, 128)
    
    return image.astype('float32') / 255.0, quality_score