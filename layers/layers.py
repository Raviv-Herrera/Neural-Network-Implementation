import numpy as np
from activations.functions import Relu, Softmax, Sigmoid


class Layer:

    def __init__(self, n_inputs: int, n_outputs: int, activation_function: str, dropout_rate: float = 0):

        self._weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
        self.vdw = np.zeros_like(self._weights)
        self.sdw = np.zeros_like(self._weights)

        self._biases = np.zeros((1, n_outputs))
        self.vdb = np.zeros_like(self._biases)
        self.sdb = np.zeros_like(self._biases)

        self._activation_function_name: str = activation_function
        self._outputs = []
        self._outputs_after_activation = []

        self._dropout_rate: float = dropout_rate if dropout_rate > 0 else 0

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs:
        :return:
        """
        if self._dropout_rate:
            d_o = DropOut(dropout_rate=self._dropout_rate)
            self._outputs = d_o.forward(inputs=inputs, weights=self.weights, biases=self.biases)
        else:
            self._outputs = inputs @ self._weights + self._biases
        return self._outputs

    def activation(self) -> np.ndarray:

        if self._activation_function_name == 'Relu':
            self._outputs = Relu.forward(X=self._outputs)
        elif self._activation_function_name == 'Softmax':
            self._outputs = Softmax.forward(X=self._outputs)
        elif self._activation_function_name == 'Sigmoid':
            self._outputs = Sigmoid.forward(X=self._outputs)
        else:
            raise Exception("No such function <!>")

        return self._outputs

    def prime(self, X) -> np.ndarray:

        if self._activation_function_name == 'Relu':
            prime = Relu.prime(X=X)
        elif self._activation_function_name == 'Sigmoid':
            prime = Sigmoid.prime(X=X)
        else:
            raise Exception("No such function <!>")

        return prime

    @property
    def outputs(self):
        return self._outputs

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, x):
        self._weights = x

    @property
    def biases(self) -> np.ndarray:
        return self._biases

    @biases.setter
    def biases(self, x):
        self._biases = x


class Dense(Layer):

    def __init__(self, n_inputs: int, n_outputs: int, activation_function, dropout_rate: float = 0):
        super().__init__(n_inputs=n_inputs, n_outputs=n_outputs, activation_function=activation_function,
                         dropout_rate=dropout_rate)


class DropOut:

    def __init__(self, dropout_rate: float):
        self._dropout_rate = dropout_rate
        self._binary_mask = None

    def forward(self, inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        self._binary_mask = np.random.binomial(1, 1 - self._dropout_rate, size=inputs.shape)
        inputs = inputs * self._binary_mask
        return inputs @ weights + biases
