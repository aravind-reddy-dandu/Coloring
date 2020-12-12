import PIL
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import pandas as pd


# Relu activation
def relu_activation(z):
    a = np.maximum(0, z)
    return a


# Initiating parameters
def params_initiate(layer_widths):
    parameters = {}
    for i in range(1, len(layer_widths)):
        parameters['W' + str(i)] = np.random.randn(layer_widths[i], layer_widths[i - 1]) * 0.01
        parameters['B' + str(i)] = np.random.randn(layer_widths[i], 1) * 0.01
    return parameters


# Forward propagation
def forward_propagation(X_train, params):
    layers = len(params) // 2
    values = {}
    # Computing each layer with current weights
    for i in range(1, layers + 1):
        if i == 1:
            # If first layer
            values['Z' + str(i)] = np.dot(params['W' + str(i)], X_train) + params['B' + str(i)]
            values['A' + str(i)] = relu_activation(values['Z' + str(i)])
        else:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], values['A' + str(i - 1)]) + params['B' + str(i)]
            if i == layers:
                # If last layer, skipping activation
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['A' + str(i)] = relu_activation(values['Z' + str(i)])
    return values


# Using mean squared error to compute cost
def compute_cost(values, Y_train):
    layers = len(values) // 2
    Y_pred = values['A' + str(layers)]
    cost = 1 / (2 * len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
    return cost


# Using all the derivatives to do back propagation
def backward_propagation(params, values, X_train, Y_train):
    layers = len(params) // 2
    m = len(Y_train)
    grads = {}
    for i in range(layers, 0, -1):
        if i == layers:
            # If last layer, using derivative of mean squared error
            dA = 1 / m * (values['A' + str(i)] - Y_train)
            dZ = dA
        else:
            # Chain rule otherwise
            dA = np.dot(params['W' + str(i + 1)].T, dZ)
            dZ = np.multiply(dA, np.where(values['A' + str(i)] >= 0, 1, 0))
        # Storing directions
        if i == 1:
            # If first layer
            grads['W' + str(i)] = 1 / m * np.dot(dZ, X_train.T)
            grads['B' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1 / m * np.dot(dZ, values['A' + str(i - 1)].T)
            grads['B' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    return grads


# Updating weights
def update_params(params, grads, learning_rate):
    layers = len(params) // 2
    params_updated = {}
    for i in range(1, layers + 1):
        params_updated['W' + str(i)] = params['W' + str(i)] - learning_rate * grads['W' + str(i)]
        params_updated['B' + str(i)] = params['B' + str(i)] - learning_rate * grads['B' + str(i)]
    return params_updated


# Driver code if Neural network
def model(X_train, Y_train, layer_sizes, num_iters, learning_rate):
    # Initiating parameters
    params = params_initiate(layer_sizes)
    # Looping and fitting data
    for i in range(num_iters):
        values = forward_propagation(X_train.T, params)
        cost = compute_cost(values, Y_train.T)
        grads = backward_propagation(params, values, X_train.T, Y_train.T)
        params = update_params(params, grads, learning_rate)
        # Printing cost for every 100 iterations
        if i % 100 == 0:
            print('Cost at iteration ' + str(i + 1) + ' = ' + str(cost) + '\n')
    return params


def compute_accuracy(X_train, X_test, Y_train, Y_test, params):
    values_train = forward_propagation(X_train.T, params)
    train_acc = mean_squared_error(Y_train, values_train['A' + str(len(layer_sizes) - 1)].T)
    test_acc = None
    values_test = None
    if X_test is not None:
        values_test = forward_propagation(X_test.T, params)
        test_acc = mean_squared_error(Y_test, values_test['A' + str(len(layer_sizes) - 1)].T)
    return train_acc, test_acc


# Forward propagating through data and predicting values
def predict(X, params):
    values = forward_propagation(X.T, params)
    predictions = values['A' + str(len(values) // 2)].T
    return predictions


# image_part = load_boston()  # load dataset
# X, Y = image_part["image_part"][:, :12], image_part["target"]  # separate image_part into input and output features
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                     test_size=0.2)  # split image_part into train and test sets in 80-20 ratio
# layer_widths = [12, 5, 5, 1]  # set layer sizes, do not change the size of the first and last layer
# num_iters = 1000  # set number of iterations over the training set(also known as epochs in batch gradient descent context)
# learning_rate = 0.03  # set learning rate for gradient descent
# params = model(X_train, Y_train, layer_widths, num_iters, learning_rate)  # train the model
# train_acc, test_acc = compute_accuracy(X_train, X_test, Y_train, Y_test, params)  # get training and test accuracy
# print('Root Mean Squared Error on Training Data = ' + str(train_acc))
# print('Root Mean Squared Error on Test Data = ' + str(test_acc))


def rgb2gray(rgb):
    # return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
    return np.dot(rgb[..., :3], [0.21, 0.72, 0.07])

# path = 'D:\\Study\\Intro_AI\\Coloring_Assignment\\Images\\reduced.png'
# rgba_image = PIL.Image.open(path)
# rgb_image = rgba_image.convert('RGB')
# img = np.array(rgb_image)
# gray = rgb2gray(img)
# left_actual = img[:, :int(img.shape[1] / 2), :]
# right_actual = img[:, int(img.shape[1] / 2):, :]
# left_half = gray[:, :int(img.shape[1] / 2)]
# right_half = gray[:, int(img.shape[1] / 2):]
#
# left_half_flat = left_half.reshape(-1, 1)
# left_actual = left_actual.reshape(-1, 3)
#
# X = left_half_flat
# colored_right = np.zeros(right_actual.shape)
# store = {}
# for i in range(3):
#     y = left_actual[:, i]
#     layer_widths = [1, 5, 5, 1]  # set layer sizes, do not change the size of the first and last layer
#     num_iters = 1000  # set number of iterations over the training set(also known as epochs in batch gradient descent context)
#     learning_rate = 0.1  # set learning rate for gradient descent
#     params = model(X, y, layer_widths, num_iters, learning_rate)  # train the model
#     train_acc, test_acc = compute_accuracy(X, None, y, None, params)  # get training and test accuracy
#     print('Root Mean Squared Error on Training Data = ' + str(train_acc))
#     print('Root Mean Squared Error on Test Data = ' + str(test_acc))
#     y_pred = find_labels(right_half.reshape(-1, 1), params)
#     y_pred = np.array(y_pred)
#     store[i] = y_pred.reshape(right_half.shape)
#     print(y_pred)
#     colored_right[:, :, i] = y_pred.reshape(right_half.shape)
# plt.imshow(colored_right / 255)
# plt.show()
# plt.imshow(right_actual / 255)
# plt.show()


# X = np.array(pd.read_csv('D:\\Study\\Intro_AI\\Coloring_Assignment\\patches/X_train.csv'))
# y = np.array(pd.read_csv('D:\\Study\\Intro_AI\\Coloring_Assignment\\patches/Y_train.csv'))[:, 0]
# layer_widths = [100, 10, 10, 1]  # set layer sizes, do not change the size of the first and last layer
# num_iters = 1500  # set number of iterations over the training set(also known as epochs in batch gradient descent context)
# learning_rate = 0.1  # set learning rate for gradient descent
# params = model(X, y, layer_widths, num_iters, learning_rate)  # train the model
# train_acc, test_acc = compute_accuracy(X, None, y, None, params)  # get training and test accuracy
# print('Root Mean Squared Error on Training Data = ' + str(train_acc))
# print('Root Mean Squared Error on Test Data = ' + str(test_acc))
# y_pred = find_labels(right_half.reshape(-1, 1), params)
# y_pred = np.array(y_pred)
