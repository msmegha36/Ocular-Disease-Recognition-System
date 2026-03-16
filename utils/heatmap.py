import tensorflow as tf
import numpy as np
import cv2
import os

def get_gradcam_heatmap(img_array_left, img_array_right, model, last_conv_layer_name="top_conv", target_side='left'):
    """Generates Grad-CAM for a specific side of a Siamese model."""
    
    # Identify which output head to track: 0 for Left branch, 1 for Right branch
    output_index = 0 if target_side == 'left' else 1
    
    # Target the specific output head from the model's output list
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[output_index]]
    )

    with tf.GradientTape() as tape:
        # Provide BOTH images as a list
        last_conv_layer_output, preds = grad_model([img_array_left, img_array_right])
        
        # 'preds' is now specifically for the side we are targeting
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Calculate gradients of the top class with respect to the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    
    # If gradients are None, the layer name might be wrong or outside the gradient path
    if grads is None:
        print(f"Warning: No gradients found for {last_conv_layer_name}. Check layer name.")
        return np.zeros((last_conv_layer_output.shape[1], last_conv_layer_output.shape[2]))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    max_val = tf.math.reduce_max(heatmap)
    if max_val <= 0:
        return np.zeros(heatmap.shape)

    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, side, alpha=0.4):
    """
    side: 'left' or 'right' to distinguish filenames
    """
    output_dir = os.path.join('static', 'heatmaps')
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(img_path)
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    superimposed_img = jet * alpha + img
    
    # Add side to filename to prevent overwriting
    filename = f"heatmap_{side}_" + os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, superimposed_img)
    
    return f"heatmaps/{filename}"