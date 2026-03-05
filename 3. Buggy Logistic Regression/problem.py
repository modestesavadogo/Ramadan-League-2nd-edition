import numpy as np

class LogisticRegressionGD:
    def __init__(self):
        self.w0 = 0.0
        self.w1 = 0.0

    def sigmoid(self, z):
        # BUG: wrong sign in exponent (will break training direction)
        return 1 / (1 + np.exp(z))

    def predict_proba(self, X):
        """
        X: shape (m,) 1D feature vector
        returns: probabilities shape (m,)
        """
        z = self.w0 + self.w1 * X
        return self.sigmoid(z)

    def compute_loss(self, X, y):
        """
        Binary cross-entropy loss:
        L = -mean(y*log(p) + (1-y)*log(1-p))
        """
        p = self.predict_proba(X)

        # BUG: missing negative sign and using sum instead of mean
        loss = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        return loss

    def gradient_descent(self, X, y, alpha=0.1, iterations=2000):
        m = len(y)

        for _ in range(iterations):
            p = self.predict_proba(X)
            errors = p - y

            # BUG: wrong gradients (sign / formula issues)
            grad_w0 = (1 / m) * np.sum(errors)
            grad_w1 = (1 / m) * np.sum(errors - X)

            self.w0 = self.w0 + alpha * grad_w0
            self.w1 = self.w1 - alpha * grad_w1

    def fit(self, X, y, alpha=0.1, iterations=2000):
        self.w0 = 0.0
        self.w1 = 0.0
        self.gradient_descent(X, y, alpha=alpha, iterations=iterations)
        return self

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    np.random.seed(42)

    # --- synthetic dataset ---
    X = np.random.randn(200) * 2
    true_w1 = 1.7
    true_w0 = -0.4

    logits = true_w0 + true_w1 * X
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(200) < probs).astype(int)

    model = LogisticRegressionGD()
    model.fit(X, y, alpha=0.2, iterations=2000)

    p = model.predict_proba(X)
    y_hat = model.predict(X)

    acc = np.mean(y_hat == y)
    # log loss
    logloss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()

    print(f"Trained parameters: w0 = {model.w0:.4f}, w1 = {model.w1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"LogLoss: {logloss:.4f}")