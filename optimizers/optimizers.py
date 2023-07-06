import numpy as np


class RMSProp:

    @staticmethod
    def optimize(beta, dw, db, sec_order_EMA_w, sec_order_EMA_b, learning_rate):
        epsilon = 1e-08
        sec_order_EMA_w = beta * sec_order_EMA_w + (1 - beta) * dw ** 2
        sec_order_EMA_b = beta * sec_order_EMA_b + (1 - beta) * db ** 2

        return (learning_rate / (np.sqrt(sec_order_EMA_w + epsilon))) * dw, \
               (learning_rate / (np.sqrt(sec_order_EMA_b + epsilon))) * db


class Adam:

    @staticmethod
    def optimize(beta1, beta2, dw, db, first_order_EMA_w, first_order_EMA_b, sec_order_EMA_w, sec_order_EMA_b,
                 current_epoch, learning_rate):
        epsilon = 1e-08

        first_order_EMA_w = beta1 * first_order_EMA_w + (1 - beta1) * dw
        first_order_EMA_b = beta1 * first_order_EMA_b + (1 - beta1) * db

        sec_order_EMA_w = beta2 * sec_order_EMA_w + (1 - beta2) * dw ** 2
        sec_order_EMA_b = beta2 * sec_order_EMA_b + (1 - beta2) * db ** 2

        mt_hat_w = first_order_EMA_w / (1 - beta1 ** (current_epoch + 1))  # adaptive mt calc
        mt_hat_b = first_order_EMA_b / (1 - beta1 ** (current_epoch + 1))  # adaptive mt calc

        vt_hat_w = sec_order_EMA_w / (1 - beta2 ** (current_epoch + 1))  # adaptive vt calc
        vt_hat_b = sec_order_EMA_b / (1 - beta2 ** (current_epoch + 1))  # adaptive vt calc

        return (learning_rate / (np.sqrt(vt_hat_w + epsilon))) * mt_hat_w, \
               (learning_rate / (np.sqrt(vt_hat_b + epsilon))) * mt_hat_b


class AdaMax:

    @staticmethod
    def optimize(beta1, beta2, dw, db, first_order_EMA_w, first_order_EMA_b, sec_order_EMA_w, sec_order_EMA_b,
                 current_epoch, learning_rate):
        epsilon = 1e-08

        first_order_EMA_w = beta1 * first_order_EMA_w + (1 - beta1) * dw
        first_order_EMA_b = beta1 * first_order_EMA_b + (1 - beta1) * db

        ut_w = np.maximum(beta2 * sec_order_EMA_w, np.abs(dw))
        ut_b = np.maximum(beta2 * sec_order_EMA_b, np.abs(db))

        eta_learning_rate = learning_rate / (1 - beta1 ** (current_epoch + 1))  # adaptive learning rate calc

        return eta_learning_rate * first_order_EMA_w / (ut_w + epsilon), \
               eta_learning_rate * first_order_EMA_b / (ut_b + epsilon)


class GD:

    @staticmethod
    def optimize(dw, db, learning_rate):
        return learning_rate * dw, learning_rate * db
