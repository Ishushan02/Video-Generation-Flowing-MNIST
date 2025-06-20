# Video-Generation-Flowing-MNIST

This project explores deep learning-based video generation using a hybrid model that combines **Convolutional Neural Networks (CNNs)** and **Long Short-Term Memory (LSTM)** units. The model is trained on the **Moving MNIST** dataset to predict future frames of handwritten digits in motion.

## Dataset: Moving MNIST

The **Moving MNIST** dataset consists of 10,000 video sequences. Each sequence contains:

* **20 grayscale frames**, each of size **64×64 pixels**
* **Two handwritten digits** moving independently and randomly within the frame
* Digits **bounce off the frame boundaries** and often **intersect or overlap**

### Example Input Video

Below is a sample sequence from the Moving MNIST dataset:

![Moving MNIST Sample](datavisualization/moving_mnist_2.gif)

---

## Approach 1: ConvLSTM Model

The first approach uses a **ConvLSTM model**, which integrates convolutional operations into LSTM units, allowing the model to capture both spatial and temporal dynamics effectively.

### Model Details

* Input: First 10 frames of a sequence
* Output: Next 10 predicted frames
* Architecture: Convolutional-LSTM network
* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Trained for: 70 Epochs

### Generated Video

Below is a 20-frame video result (10 input + 10 generated) from the trained ConvLSTM model:

![ConvLSTM Output](datavisualization/output-70-epochs.gif)

---
