import tensorflow as tf
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

app = FastAPI(
    title="Industrial Steel Defect Detection API",
    description="High-performance inference API for NEU Surface Defect Database using Custom ResNet and Transfer Learning models.",
    version="1.0.0"
)

# --- Model Loading Strategy ---
# Professional Note: Using absolute-like paths or env variables is preferred in Docker
try:
    # We load both models to allow potential A/B testing or ensemble logic
    MODEL_CUSTOM_PATH = "models/resnet_custom.keras"
    MODEL_TL_PATH = "models/resnet50_pretrained.keras"
    
    model_custom = tf.keras.models.load_model(MODEL_CUSTOM_PATH)
    # model_tl = tf.keras.models.load_model(MODEL_TL_PATH) # Optional: Load TL model if needed
    
    CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
except Exception as e:
    print(f"Critical Error: Failed to load models. {e}")

@app.post("/predict", tags=["Inference"])
async def predict_defect(file: UploadFile = File(...)):
    """
    Asynchronous endpoint to perform inference on industrial steel surface images.
    Input: Multipart form-data image file.
    Output: JSON containing predicted class and softmax confidence score.
    """
    
    # 1. Validate file extension to prevent processing non-image data
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # 2. Non-blocking read of the uploaded byte stream
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")

        # 3. Preprocessing Pipeline
        # Note: resize() requires a tuple (width, height). 
        # Normalization (1/255) must match the training-time preprocessing.
        image = image.resize((224, 224)) 
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # 4. Model Inference
        # In a remote-ready setup, consider using model.predict_on_batch for single samples
        predictions = model_custom.predict(img_array, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        # 5. Result Serialization
        return {
            "defect_type": CLASSES[class_index],
            "confidence": round(confidence, 4),
            "model_version": "custom_resnet_v1"
        }

    except Exception as e:
        # High-level error logging for production monitoring
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/health", tags=["System"])
async def health_check():
    """Service health check for Docker/K8s liveness probes."""
    return {"status": "healthy", "model_loaded": model_custom is not None}