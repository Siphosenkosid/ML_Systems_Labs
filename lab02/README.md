# Lab 2: A Neural Network from Scratch

## Objective

The aim of this lab was to build a neural network from scratch using NumPy. The lab covers dataset preparation, network architecture, forward propagation, loss calculation, backpropagation, training, and performance evaluation.

## Task 1: Dataset Preparation

The `load_digits` dataset from Scikit-learn was used. It contains 1,797 images of handwritten digits, with 64 features per image.

The input values were divided by 16 to normalize them between 0 and 1. The labels were one-hot encoded into 10 classes.

The dataset was divided into 70% training and 30% testing using a stratified split with `random_state=42`.

**Expected dataset dimensions:**
- Training inputs: (1257, 64)
- Testing inputs: (540, 64)
- Training labels: (1257, 10)
- Testing labels: (540, 10)

## Task 2: Neural Network Architecture

A neural network with two hidden layers was designed.

The architecture was:

Input (64) → Hidden (W) → ReLU → Hidden (W) → ReLU → Output (10) → Softmax

Two hidden widths were considered: W = 16 and W = 64.

The weight dimensions were:
- W1: (64, W)
- W2: (W, W)
- W3: (W, 10)

The biases had dimensions (W,), (W,), and (10,).

Weights were initialized using a normal distribution with standard deviation 0.01.

## Task 3: Forward Propagation

Forward propagation was implemented using matrix multiplication, bias addition, ReLU activation, and softmax.

The network produces probabilities for the 10 digit classes.

The output was checked to ensure that its dimensions were `(batch_size, 10)` and that the probabilities in each row summed to 1 within a tolerance of 1e-6.

## Task 4: Cross-Entropy Loss

Cross-entropy loss was implemented to measure the difference between the predicted probabilities and the correct labels.

The expected loss for an untrained network was approximately:

**ln(10) = 2.3026**

This was used as a sanity check before training.

## Task 5: Backpropagation and Gradient Checking

Backpropagation was implemented using the chain rule to calculate gradients for all weights and biases.

Numerical gradient checking was performed by changing 10 randomly selected weights by ±0.00001.

The analytical and numerical gradients were compared using relative error.

The requirement was for at least 9 out of 10 checked weights to have a relative error below 1e-6.

**Result:** See `results/task5_results.txt` for the actual gradient-checking output.

## Task 6: Neural Network Training

The network was trained using mini-batch gradient descent.

Training settings:
- Batch size: 32
- Epochs: 50
- Learning rates: 0.001, 0.01, 0.1
- Hidden widths: 16 and 64

The training loss was recorded after each epoch, and final test accuracy was evaluated.

The loss curves were plotted to compare the learning rates and hidden-layer widths.

**Results:**
- `results/task6_results.txt`
- `results/loss_curves.png`

The best-performing configuration should be identified from the actual test accuracy results.

## Task 7: Forward and Backward Pass Timing

The execution times of forward propagation and backward propagation were measured separately.

Each operation was timed over 100 runs, and the median execution time was calculated.

The timings were compared to determine which operation required more computation.

Backward propagation can take longer because it involves additional matrix multiplications to calculate gradients for the network parameters.

**Result:** See `results/task7_results.txt` for the measured times and comparison.

## Connection to Reddi Chapter 3

This lab applies the neural network fundamentals described in Chapter 3 of Reddi's Deep Learning Primer.

- **Section 3.4:** Neural network fundamentals and architecture.
- **Sections 3.5.1–3.5.5:** Training, forward propagation, loss functions, backpropagation, and optimization.
- **Section 3.6:** Prediction phase.

These concepts were applied to understand how a neural network processes inputs, calculates errors, updates parameters, and makes predictions.

## Conclusion

This lab demonstrates the main stages of implementing a neural network from scratch, from preparing the dataset to evaluating accuracy and computational performance.

The experiments allow comparison of how learning rate and hidden-layer width influence training behavior, accuracy, and execution time.

The actual outputs are stored in the `results` folder.