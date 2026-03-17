import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

def apply_clahe(img):
    """Enhances retinal features for better Grad-CAM localization."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def generate_siamese_heatmap(img_path, model, target_side='left'):
    """
    Generates a Grad-CAM heatmap for one side of the Siamese model.
    """
    # 1. Load and Preprocess
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_enhanced = apply_clahe(img_rgb)
    
    img_res = cv2.resize(img_enhanced, (512, 512))
    img_array = np.expand_dims(img_res, axis=0)
    img_preprocessed = preprocess_input(img_array)
    img_tensor = tf.convert_to_tensor(img_preprocessed)

    # 2. Grad-CAM Logic for EfficientNetB3 backbone
    backbone = model.get_layer('efficientnetb3')
    target_layer = backbone.get_layer('top_conv')
    
    # We create a model that maps the backbone input to the target conv layer
    grad_model = tf.keras.models.Model(
        inputs=backbone.inputs,
        outputs=[target_layer.output, backbone.output]
    )

    with tf.GradientTape() as tape:
        # Pass the single eye tensor through the shared backbone
        conv_outputs, predictions = grad_model(img_tensor)
        loss = tf.reduce_max(predictions, axis=-1)

    # 3. Compute Heatmap
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = np.maximum(cam, 0)
    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
    
    # 4. Create Overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Apply standard clinical overlay: 60% original, 40% heatmap
    result = cv2.addWeighted(img_enhanced, 0.6, heatmap_color, 0.4, 0)
    
    # 5. Save and return path
    output_filename = f"heatmap_{target_side}_{os.path.basename(img_path)}"
    output_path = os.path.join('static', 'heatmaps', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert back to BGR for saving with OpenCV
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    return output_path