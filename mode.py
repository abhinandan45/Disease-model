import tensorflow as tf
import os

# Define the path to your original .h5 model
H5_MODEL_PATH = os.path.join('models', 'plant_disease_model.h5')

# Define the path where the quantized .tflite model will be saved
TFLITE_MODEL_PATH = os.path.join('models', 'plant_disease_model_quantized.tflite')

def convert_h5_to_tflite_quantized(h5_model_path, tflite_model_path):
    """
    Converts a Keras .h5 model to a quantized TensorFlow Lite model.
    """
    if not os.path.exists(h5_model_path):
        print(f"Error: .h5 model not found at {h5_model_path}")
        return

    try:
        # 1. Load the original Keras .h5 model
        print(f"Loading Keras model from: {h5_model_path}")
        model = tf.keras.models.load_model(h5_model_path)
        print("Keras model loaded successfully.")

        # 2. Create a TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # 3. Enable optimizations (quantization)
        # This will apply dynamic range quantization, which quantizes weights and
        # optionally activations to 8-bit integers during inference.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Optional: If you want full integer quantization (requires a representative dataset)
        # This provides the best performance but is more complex to implement.
        # For now, dynamic range quantization is a good starting point.
        # converter.representative_dataset = representative_data_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8


        # 4. Convert the model
        print("Converting model to TFLite with dynamic range quantization...")
        tflite_model = converter.convert()
        print("Model conversion complete.")

        # 5. Save the converted model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Quantized TFLite model saved successfully to: {tflite_model_path}")

    except Exception as e:
        print(f"An error occurred during model conversion: {e}")

if __name__ == "__main__":
    convert_h5_to_tflite_quantized(H5_MODEL_PATH, TFLITE_MODEL_PATH)