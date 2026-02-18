import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import os

# --- CONFIGURATION ---
# Ensure this filename matches the one from your training script
WEIGHTS_PATH = "steel_model_weights.h5"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_SIZE = (200, 200)

def build_inference_model():
    """
    Reconstructs the EXACT architecture from the training script.
    This is required to correctly map the numerical weights from the .h5 file.
    """
    # Define Input Layer
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    
    # 1. Preprocessing Layer (Must match training exactly)
    # This processes raw 0-255 pixels into ResNet50-compatible format
    x = keras.applications.resnet50.preprocess_input(inputs)
    
    # 2. ResNet50 Backbone
    # weights=None because we are loading our custom trained weights later
    base_model = keras.applications.ResNet50(
        include_top=False, 
        weights=None, 
        input_shape=(*IMG_SIZE, 3)
    )
    
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # 3. Classification Head (Matching your Dropout 0.3)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    
    return keras.Model(inputs, outputs)

# --- MODEL INITIALIZATION ---
print("[*] Initializing model architecture...")
model = build_inference_model()

if os.path.exists(WEIGHTS_PATH):
    print(f"[*] Loading weights from {WEIGHTS_PATH}...")
    try:
        # Loading only weights avoids all Keras 3 metadata/descriptor issues
        model.load_weights(WEIGHTS_PATH)
        print("[+] Success: Weights loaded successfully.")
    except Exception as e:
        print(f"[!] Error: Could not load weights. {e}")
else:
    print(f"[!] Error: {WEIGHTS_PATH} not found. Please upload the weights file.")

def predict(img):
    """
    Inference pipeline for Gradio.
    """
    if img is None:
        return None
    
    # Convert PIL image to RGB numpy array
    img_array = np.array(img.convert("RGB"))
    
    # Resize to the required input size
    img_res = cv2.resize(img_array, IMG_SIZE)
    
    # Add batch dimension (1, 200, 200, 3)
    img_batch = np.expand_dims(img_res.astype(np.float32), axis=0)
    
    # Perform prediction
    preds = model.predict(img_batch, verbose=0)
    
    # Map probabilities to class names
    return {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}

# --- GRADIO INTERFACE ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Steel Surface Photo"),
    outputs=gr.Label(num_top_classes=3, label="Defect Classification"),
    title="Steel Defect Detection System",
    description="A Deep Learning tool based on ResNet50 for automated quality control of steel surfaces."
)

if __name__ == "__main__":
    # Launching the app
    demo.launch()