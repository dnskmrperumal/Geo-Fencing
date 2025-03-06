import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="median_model.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test input (random size)
#test_input = np.array([
#    [37.7749, -122.4194],
#    [40.7128, -74.0060],
#    [34.0522, -118.2437],
#    [51.5074, -0.1278],
#    [35.6895, 139.6917],
#], dtype=np.float32)

test_input = np.array([
        [9.881047,78.073264],
        [9.881132,78.072591],
        [9.880693,78.072962],
        [9.881047,78.073264],
        [9.881494,78.073929],
        [9.881596,78.073992],
        [9.881530,78.073157],
        [9.881758,78.073135],
        [9.882394,78.073076],
        [9.881908,78.072223],
    ], dtype=np.float32)



# Resize input tensor dynamically
interpreter.resize_tensor_input(input_details[0]['index'], (1, test_input.shape[0], 2))
interpreter.allocate_tensors()  # Must re-allocate after resizing

# Reshape and set input
input_data = np.expand_dims(test_input, axis=0)  
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("✅ Correct Median Coordinates:", output_data)

