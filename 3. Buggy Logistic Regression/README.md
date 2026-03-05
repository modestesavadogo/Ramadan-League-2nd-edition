## Overview

In this problem, you will debug an implementation of **Logistic Regression** trained using **Gradient Descent**.
The provided code contains multiple bugs that prevent correct learning.

## Background

Logistic Regression predicts the probability of class 1 using:

p(y=1|x) = sigmoid(w0 + w1*x)

Where:

sigmoid(z) = 1 / (1 + exp(-z))

The model is trained by minimizing **Binary Cross-Entropy (Log Loss)**:

L = -mean( y*log(p) + (1-y)*log(1-p) )

## The Challenge

The file `buggy_logistic_regression.py` contains bugs inside:
- the sigmoid function
- the loss computation
- the gradient descent update

### Your tasks

1. Identify and fix the bugs.
2. Run the script after fixing it.
3. Your output should match the expected values (small floating error is acceptable).

## Expected Output

- Trained parameters: w0 = -0.6723, w1 = 2.0394
- Accuracy: 0.8600
- LogLoss: 0.3081

## Notes

- You are not allowed to change the dataset generation (seed must remain 42).
- You may add small numerical stabilizers (like eps) if needed.
- Focus on correctness, not performance.