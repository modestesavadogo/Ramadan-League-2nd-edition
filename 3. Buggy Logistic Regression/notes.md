First, we changed 
        1 / (1 + np.exp(z)) 

to      1 / (1 + np.exp(-z)) 
in the sigmoid function on line 10.

Then we changed 
        loss = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

to      loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) 
in the compute_loss function on line 28.

Finally, we changed 
        grad_w0 = (1 / m) * np.sum(errors)
        grad_w1 = (1 / m) * np.sum(errors - X)

to      grad_w0 -= (1 / m) * self.compute_loss(X, y)
        grad_w1 -= (1 / m) * np.sum(errors * X)
in the gradient_descent function on lines 39-40.


Our output is now :
    Trained parameters: w0 = -0.6723, w1 = 2.0394
Accuracy: 0.8600
LogLoss: 0.3081
