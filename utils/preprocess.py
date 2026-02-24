import cv2
import numpy as np
import os

def preprocess_image(path):
    """
    Standardizes input images to 512x512 with Ben Graham's method.
    This must match the training pipeline to maintain 96% AUC.
    """
    if not os.path.exists(path):
        # Return a blank tensor if the file is missing to prevent crash
        return np.zeros((512, 512, 3), dtype='float32')
    
    # 1. Load and convert color space
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Resize to 512x512 (Matches EfficientNetB3 training)
    image = cv2.resize(image, (512, 512))
    
    # 3. Ben Graham's Preprocessing: Subtract local average to highlight lesions
    # This removes lighting variations and makes blood vessels/spots pop out
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)
    
    # 4. Normalize and add batch dimension
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)