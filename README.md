Median Calculation Model (TensorFlow Lite)

This repository contains a TensorFlow model that calculates the median of a dynamic set of latitude-longitude coordinates and converts it into a TensorFlow Lite (TFLite) model for efficient inference on edge devices.

Features

Dynamic Input Support: Accepts any number of coordinate pairs (N, 2).

Efficient Median Calculation: Uses TensorFlow operations to compute the median correctly for both odd and even input sizes.

TFLite Conversion: Converts the model into a .tflite file for lightweight deployment.

Requirements

Before running the script, install the required dependencies:

pip install tensorflow

Usage

Running the Script

To generate the TFLite model, run:

python geo_fencing.py

This will create a median_model.tflite file in the current directory.

TensorFlow Model Overview

The model:

Accepts a (N, 2) input representing latitude-longitude pairs.

Sorts the input values.

Computes the median, handling both even and odd cases.

Converts and saves the model in the .tflite format.

Expected Output

After running the script, you should see:

 TFLite model successfully created: median_model.tflite

Deployment

You can use the generated median_model.tflite in TensorFlow Lite-supported environments, such as Android, Raspberry Pi, or embedded systems, for efficient median calculation.

License

This project is released under the MIT License.
