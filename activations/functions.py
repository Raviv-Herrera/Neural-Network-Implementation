import numpy as np


class Relu:

    @staticmethod
    def forward(X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    @staticmethod
    def prime(X):
        return np.where(X > 0, 1.0, 0.0)


class Softmax:

    @staticmethod
    def forward(X: np.ndarray) -> np.ndarray:
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Sigmoid:

    @staticmethod
    def forward(X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def prime(X):
        y = Sigmoid.forward(X) * (1 - Sigmoid.forward(X))
        return y
