import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import os

# --- CONFIGURATION ---
WEIGHTS_PATH = "steel_model_weights.weights.h5"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_SIZE = (200, 200)

def build_inference_model():
    """
    Reconstructs the EXACT architecture from the training script.
    """
    inputs = keras.Input(shape=(200, 200, 3))
    
    # Preprocessing Layer
    x = keras.applications.resnet50.preprocess_input(inputs)
    
    # ResNet50 Backbone
    base_model = keras.applications.ResNet50(
        include_top=False, 
        weights=None, 
        input_shape=(200, 200, 3)
    )
    
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Classification Head
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    
    return keras.Model(inputs, outputs)

# --- MODEL INITIALIZATION ---
print("[*] Initializing model architecture...")
model = build_inference_model()

if os.path.exists(WEIGHTS_PATH):
    print(f"[*] Loading weights from {WEIGHTS_PATH}...")
    try:
        model.load_weights(WEIGHTS_PATH)
        print("[+] Success: Weights loaded successfully.")
    except Exception as e:
        print(f"[!] Error: Could not load weights. {e}")
else:
    print(f"[!] Error: {WEIGHTS_PATH} not found. Upload the .weights.h5 file!")

def predict(img):
    if img is None:
        return None
    
    img_array = np.array(img.convert("RGB"))
    img_res = cv2.resize(img_array, IMG_SIZE)
    img_batch = np.expand_dims(img_res.astype(np.float32), axis=0)
    
    preds = model.predict(img_batch, verbose=0)
    
    return {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}

# --- GRADIO INTERFACE ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Steel Surface Photo"),
    outputs=gr.Label(num_top_classes=3, label="Defect Classification Result"),
    title="Steel Defect Detection System (Keras 3)",
    description="Automated quality control using ResNet50 Transfer Learning."
)

if __name__ == "__main__":
    demo.launch()