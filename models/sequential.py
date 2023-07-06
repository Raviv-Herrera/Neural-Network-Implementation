import numpy as np
from timeit import default_timer
from layers.layers import Layer
from alive_progress import alive_bar
from loss.loss_functions import CategoricalCrossEntropy
from optimizers.optimizers import RMSProp, Adam, AdaMax, GD
from sklearn.metrics import accuracy_score
from loguru import logger
from time import sleep


class Sequential:

    def __init__(self):
        self._layers = []
        self._learning_rate: float = 0.001
        self._cost: float = .0
        self._loss_name: str = ""
        self._optimizer: str = ""
        self.Loss_list = []
        self.epochs_list = []
        self.accuracy_list = []
        self.val_accuracy_list = []
        self.val_accuracy_plot = []
        self.val_accu_eps = .01

    def add(self, layer: Layer):
        self._layers.append(layer)

    def compile(self, loss: str, optimizer: str, learning_rate: float):
        self._loss_name = loss
        self._optimizer = optimizer
        self._learning_rate = learning_rate

    def forward_propagation(self, a: np.ndarray, outputs_for_back_propagation: list,
                            activations_for_back_propagation: list):

        weights = []

        for j, eachLayer in enumerate(self._layers):
            z = eachLayer.forward(a)
            outputs_for_back_propagation.append(z)
            a = eachLayer.activation()
            activations_for_back_propagation.append(a)
            weights = eachLayer.weights

        return a, outputs_for_back_propagation, activations_for_back_propagation, weights

    def back_propagation(self, a: np.ndarray, y: np.ndarray, outputs_for_back_propagation: list,
                         activations_for_back_propagation: list, m, current_epoch):

        deliver_delta = a - y
        for layer_index in reversed(range(1, self._layers.__len__())):

            dw = (1 / m) * np.dot(activations_for_back_propagation[layer_index - 1].T, deliver_delta)
            db = (1 / m) * np.sum(deliver_delta)

            deliver_delta = np.dot(deliver_delta, self._layers[layer_index].weights.T) * \
                self._layers[layer_index - 1].prime(outputs_for_back_propagation[layer_index - 1])

            updated_w, updated_b = [], []

            if self._optimizer == 'RMSProp':
                updated_w, updated_b = RMSProp.optimize(beta=0.0, dw=dw, db=db,
                                                        sec_order_EMA_w=self._layers[layer_index].sdw,
                                                        sec_order_EMA_b=self._layers[layer_index].sdb,
                                                        learning_rate=self._learning_rate)

            if self._optimizer == 'Adam':
                updated_w, updated_b = Adam.optimize(beta1=0.9, beta2=0.999, dw=dw, db=db,
                                                     first_order_EMA_w=self._layers[layer_index].vdw,
                                                     first_order_EMA_b=self._layers[layer_index].vdb,
                                                     sec_order_EMA_w=self._layers[layer_index].sdw,
                                                     sec_order_EMA_b=self._layers[layer_index].sdb,
                                                     current_epoch=current_epoch,
                                                     learning_rate=self._learning_rate)
            if self._optimizer == 'AdaMax':
                updated_w, updated_b = AdaMax.optimize(beta1=0.9, beta2=0.999, dw=dw, db=db,
                                                       first_order_EMA_w=self._layers[layer_index].vdw,
                                                       first_order_EMA_b=self._layers[layer_index].vdb,
                                                       sec_order_EMA_w=self._layers[layer_index].sdw,
                                                       sec_order_EMA_b=self._layers[layer_index].sdb,
                                                       current_epoch=current_epoch,
                                                       learning_rate=self._learning_rate)

            if self._optimizer == 'GD':
                updated_w, updated_b = GD.optimize(dw=dw, db=db, learning_rate=self._learning_rate)

            self._layers[layer_index].weights -= updated_w
            self._layers[layer_index].biases -= updated_b

    def predict(self, X: np.ndarray):

        a = X
        for j, eachLayer in enumerate(self._layers):
            _ = eachLayer.forward(a)
            a = eachLayer.activation()

        return a

    def get_validation_accuracy(self, X_validation: np.ndarray, y_validation: np.ndarray) -> float:

        validation_predict: np.ndarray = self.predict(X=X_validation)
        return accuracy_score(np.argmax(y_validation, axis=-1), np.argmax(validation_predict, axis=-1)) * 100

    @staticmethod
    def evaluate(y_test, y_predict):
        accuracy = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(y_predict, axis=-1)) * 100
        logger.info(f"Testing accuracy: {accuracy} %")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray,
            batch_size: int, num_epochs: int):

        m = X_train.shape[0]
        _outputs_for_back_propagation, _activations_for_back_propagation = [], []
        for i in range(num_epochs):
            current_epoch = i
            start = default_timer()
            with alive_bar(m // batch_size, force_tty=True) as bar:
                for p in range(m // batch_size):

                    k = p * batch_size
                    l = (p + 1) * batch_size
                    a = X_train[k:l]
                    y = y_train[k:l]

                    a, _outputs_for_back_propagation, _activations_for_back_propagation, weights =\
                        self.forward_propagation(a=a, outputs_for_back_propagation=_outputs_for_back_propagation,
                                                 activations_for_back_propagation=_activations_for_back_propagation)

                    loss = CategoricalCrossEntropy.forward(y_predict=a, y_true=y,
                                                           weights=weights)

                    self.back_propagation(a=a, m=m, y=y, outputs_for_back_propagation=_outputs_for_back_propagation,
                                          activations_for_back_propagation=_activations_for_back_propagation,
                                          current_epoch=current_epoch)

                    accuracy = accuracy_score(np.argmax(y, axis=-1), np.argmax(a, axis=-1)) * 100
                    validation_accuracy = self.get_validation_accuracy(X_validation=X_validation,
                                                                       y_validation=y_validation)

                    bar()
                    sleep(0.001)
            end = default_timer()

            logger.info(f"\nepochs:" + str(i + 1) + " | " +
                        f"runtime: {float(round(end - start, 3))} s" + " | " +
                        f"Loss:" + str(loss) + " | " +
                        f"Accuracy: {float(round(accuracy, 3))} %" + " | " +
                        f"Validation accuracy: {float(round(validation_accuracy, 3))} %")

            self.val_accuracy_list.append(validation_accuracy)  # very important <!> for validation, each epoch counts

            if i % 3 == 0:
                self.accuracy_list.append(accuracy)  # for plotting
                self.Loss_list.append(loss)  # for plotting
                self.epochs_list.append(i)  # for plotting
                self.val_accuracy_plot.append(validation_accuracy)
                if current_epoch > 1:
                    diff_validation = self.val_accuracy_list[-1] - self.val_accuracy_list[-2]
                    if 0 > diff_validation or diff_validation < self.val_accu_eps:
                        logger.info(f"\n \n Early stopping after {current_epoch} epochs \n The reason : diff_ "
                                    f"validation = {diff_validation}")
                        break

        logger.info(f"Training accuracy: {accuracy} % \n ")
