import numpy as np


class CategoricalCrossEntropy:

    @staticmethod
    def forward(y_predict, y_true, weights):
        epsilon = 10 ** (-100)
        decay_param = 0.0001
        m = y_predict.shape[0]
        return (-1) * (1 / m) * np.sum((y_true * np.log(y_predict + epsilon))) + decay_param * np.sum(weights**2)
