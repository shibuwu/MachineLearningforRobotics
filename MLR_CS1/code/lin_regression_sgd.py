import numpy as np
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from visualizer import plot_linear_trajectory


class SGDLinearRegression:
    """Linear regression implementation using stochastic gradient descent optimization."""

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))

    def _compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def _compute_gradients(self, X, y_true, y_pred):
        n = X.shape[0]
        dw = (2 / n) * (X.T @ (y_pred - y_true))
        db = (2 / n) * np.sum(y_pred - y_true, axis=0, keepdims=True)
        return dw, db

    def fit(self, X, y, batch_size=32, epochs=100):
        n_samples, input_dim = X.shape
        output_dim = y.shape[1]
        self._initialize_parameters(input_dim, output_dim)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = X_batch @ self.weights + self.bias
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        return X @ self.weights + self.bias


if __name__ == "__main__":
    use_engineered_features = True 


    #############Your CODE STARTS HERE##############

    # Load data
    X_train, X_test, y_train, y_test = prepare_dataset("data/ur10_dataset.csv")

    # Convert to numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    if use_engineered_features:
        from feature_engineering import engineer_features
        X_train = engineer_features(X_train)
        X_test = engineer_features(X_test) 

    # Train model
    model = SGDLinearRegression(learning_rate=0.01)
    model.fit(X_train, y_train)

    #############Your CODE ENDS HERE##############

    plot_linear_trajectory(model, use_engineered_features=use_engineered_features)