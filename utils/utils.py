import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import seaborn as sns
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from loguru import logger
from typing import Tuple
from layers.layers import Dense
from models.sequential import Sequential
np.random.seed(0)


def preprocess_mnist_dataset(is_augmentation: bool):
    """
    This function preprocesses the mnist dataset in purpose to fit the first layer of the CNN.
    :param is_augmentation: (bool) , a flag that indicates whether to use augmentation or not 
    :return:
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if is_augmentation:
        X_train, y_train = augment_dataset(X_train=X_train, y_train=y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

    X_train = X_train / 255
    X_val = X_val / 255
    X_test = X_test / 255

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10).T
    y_val = tensorflow.keras.utils.to_categorical(y_val, 10).T
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10).T

    X_train = X_train.reshape(-1, 28 * 28)
    X_val = X_val.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    y_train = y_train.T
    y_val = y_val.T
    y_test = y_test.T

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_combined_results(loss_lists: list, accuracy_lists: list, opt_names: list, ep_lists: list,
                          val_accuracy_lists: list, lr: float) -> None:
    """
    This function plots the final results of the training phase
    :param lr: (float) Learning rate 
    :param loss_lists: (List) List of lists , each list holds the loss values during the training phase
    :param accuracy_lists: (List) List of lists , each list holds the accuracy values during the training phase
    :param opt_names: (List) List, containing the names of the optimizers
    :param ep_lists: (List) List of the epochs 
    :param val_accuracy_lists: (List) List of lists , each list holds the accuracy values of the validation set during the training phase
    :return:
    """
    colors = ['r', 'b', 'k', 'g', 'yellow']
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.title(f'Accuracy graph - \nlearning rate = {lr}')
    for i, accuracy_list in enumerate(accuracy_lists):
        plt.plot(ep_lists[i], accuracy_list, '--', color=colors[i], label=f"{opt_names[i]}")
        plt.legend()

    plt.subplot(1, 3, 2)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss convergence graph - \nlearning rate = {lr}')
    for i, loss_list in enumerate(loss_lists):
        plt.plot(ep_lists[i], loss_list, '--', color=colors[i], label=f"{opt_names[i]}")
        plt.legend()

    plt.subplot(1, 3, 3)
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.title(f'Validation accuracy graph - \nlearning rate = {lr}')
    for i, val_accuracy_list in enumerate(val_accuracy_lists):
        plt.plot(ep_lists[i], val_accuracy_list, '--', color=colors[i], label=f"{opt_names[i]}")
        plt.legend()

    fig1 = plt.gcf()
    fig1.savefig(f'accuracy_{lr}.png')


def augment_dataset(X_train: np.ndarray, y_train: np.ndarray) -> Tuple: return augmentation(X=X_train, y=y_train)


def augmentation(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param X:
    :param y:
    :return:
    """
    dataGen = ImageDataGenerator(rotation_range=30,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 validation_split=0.2)
    reshaped_X = X.reshape(X.shape[0], 28, 28, 1)

    dataGen.fit(reshaped_X)
    batch_size = 250
    num_iterations = 20
    logger.info(f"Start Augmentation of {batch_size * num_iterations} new samples")
    for i, (X_batch, y_batch) in enumerate(dataGen.flow(reshaped_X, y, batch_size=batch_size)):
        logger.info(f"Start Augmentation id {i} ... ")
        if i < num_iterations:
            for augmented_image in X_batch:
                augmented_image = augmented_image / 255
                X = np.vstack((X, augmented_image.reshape(1, 28, 28)))
            y = np.hstack((y, y_batch))
        else:
            break

    logger.success("Finish")
    return X, y


def plot_heatmaps(y_predicted_lists: list, y_test: np.ndarray, opt_names: list, epochs: int, lr: float):
    plt.figure(figsize=(12, 7))
    dim = len(y_test[0])
    cm = np.zeros((dim, dim), int)

    for index, list_i in enumerate(y_predicted_lists):
        plt.subplot(1, 3, index + 1)
        plt.title(f'Confusion Matrix - \n {opt_names[index]} \n learning rate = {lr} \n After {epochs} epochs')
        for i in range(len(y_test)):
            truth = np.argmax(y_test[i])
            predicted = np.argmax(list_i[i])
            cm[truth, predicted] += 1
        sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 6})
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

    fig1 = plt.gcf()
    fig1.savefig(f'cm_{lr}.png')


def comparison() -> None:
    """

    :return:
    """
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_mnist_dataset(is_augmentation=False)
    optimization_methods = ['RMSProp', 'Adam', 'AdaMax']
    lr_list = [0.0005, 0.001, 0.002, 0.005, 0.01]
    epochs = 200

    for lr in lr_list:

        logger.info(f"Training with learning rate  = {lr}")
        loss_lists = []
        accuracy_lists = []
        opt_names = []
        ep_lists = []
        val_accuracy_lists = []
        y_predicted_lists = []

        for optimization_method in optimization_methods:
            logger.info(f"Training with optimizer  = {optimization_method}")

            model = Sequential()
            model.add(layer=Dense(n_inputs=784, n_outputs=150, activation_function='Relu'))
            model.add(layer=Dense(n_inputs=150, n_outputs=50, activation_function='Relu'))
            model.add(layer=Dense(n_inputs=50, n_outputs=10, activation_function='Softmax'))
            model.compile(loss='CategoricalCrossEntropy', optimizer=optimization_method, learning_rate=lr)
            model.fit(X_train=X_train, y_train=y_train, X_validation=X_val, y_validation=y_val,
                      batch_size=20000, num_epochs=epochs)
            y_predicted = model.predict(X=X_test)
            model.evaluate(y_test=y_test, y_predict=y_predicted)

            loss_lists.append(model.Loss_list)
            accuracy_lists.append(model.accuracy_list)
            opt_names.append(optimization_method)
            ep_lists.append(model.epochs_list)
            val_accuracy_lists.append(model.val_accuracy_plot)
            y_predicted_lists.append(y_predicted)

        plot_heatmaps(y_predicted_lists=y_predicted_lists,
                      y_test=y_test,
                      opt_names=opt_names,
                      epochs=epochs,
                      lr=lr)

        plot_combined_results(loss_lists=loss_lists,
                              accuracy_lists=accuracy_lists,
                              opt_names=opt_names,
                              ep_lists=ep_lists,
                              val_accuracy_lists=val_accuracy_lists,
                              lr=lr)
