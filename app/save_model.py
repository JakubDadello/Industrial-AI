import bentoml
import tensorflow as tf

# --- Standardized path for model artifacts ---
MODEL_PATH = "models/resnet50_pretrained.keras"

# --- Load the trained Keras model into memory ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Register the model in the BentoML Local Model Store ---
bentoml.tensorflow.save_model("resnet50_model", model)

