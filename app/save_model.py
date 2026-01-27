import bentoml
import tensorflow as tf

MODEL_PATH = "models/resnet_custom.keras"

model = tf.keras.models.load_model(MODEL_PATH)

bentoml.tensorflow.save_model("resnet_custom_model", model)