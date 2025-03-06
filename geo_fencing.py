import tensorflow as tf

class MedianModel(tf.keras.Model):
    def call(self, inputs):
        # Sort inputs based on latitude (axis=1)
        sorted_inputs = tf.sort(inputs, axis=1)  

        # Compute the median index
        num_elements = tf.shape(sorted_inputs)[1]
        median_index = num_elements // 2

        # Compute the median correctly
        def compute_median():
            return sorted_inputs[:, median_index, :]  

        def compute_mean_of_medians():
            median1 = sorted_inputs[:, median_index, :]
            median2 = sorted_inputs[:, median_index - 1, :]
            return (median1 + median2) / 2.0  

        median = tf.cond(
            tf.equal(num_elements % 2, 0),
            true_fn=compute_mean_of_medians,
            false_fn=compute_median
        )

        return tf.expand_dims(median, axis=1)  

# Define input with dynamic size (N, 2)
inputs = tf.keras.Input(shape=(None, 2))  
outputs = MedianModel()(inputs)

# Create model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  
tflite_model = converter.convert()

# Save model
with open("median_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model successfully created: median_model.tflite")

