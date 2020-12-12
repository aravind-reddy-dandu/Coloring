import PIL
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    # return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
    return np.dot(rgb[..., :3], [0.21, 0.72, 0.07])


# Function to create X image_part as defined
def create_x(size):
    # Values from 1 through 10 are normal random values
    X_1to10 = np.random.normal(0, 1, (size, 10))
    # Defining standard deviation
    sigma = np.sqrt(0.1)
    # Using given formulae for x_11, x12_....
    X_11 = np.asarray(
        [X_1to10[i][1] + X_1to10[i][2] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_12 = np.asarray(
        [X_1to10[i][3] + X_1to10[i][4] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_13 = np.asarray(
        [X_1to10[i][4] + X_1to10[i][5] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_14 = np.asarray([0.1 * X_1to10[i][7] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_15 = np.asarray(
        [2 * X_1to10[i][2] - 10 + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape((-1, 1))
    # x_16 through x_20 are again normal random values
    X_16to20 = np.random.normal(0, 1, (size, 5))
    # Concatenating all values to create one single array of X values
    return np.concatenate((X_1to10, X_11, X_12, X_13, X_14, X_15, X_16to20), axis=1)


# Function to generate true weights for Xs using formula
def generate_true_weights():
    w_actual = []
    # Using formula to generate weights
    for i in range(1, 21):
        if i <= 10:
            w_actual.append(0.6 ** i)
        else:
            w_actual.append(0)
    return w_actual


# Function to generate Y values given X matrix
def create_y(X, size):
    y = []
    sigma = np.sqrt(0.1)
    # Creating y value for each row
    for i in range(size):
        randomness = 10
        for j in range(1, 11):
            randomness += (0.6 ** j) * X[i][j - 1]
        randomness += np.random.normal(loc=0, scale=sigma)
        y.append(randomness)
    # Returns y values in a np array
    return np.asarray(y)


# Function to merge X and Y image_part and create a pandas dataframe
def merge_x_y_data(m):
    X = create_x(m)
    y = create_y(X, m).reshape((m, 1))
    data = pd.DataFrame(np.append(X, y, axis=1), columns=["X" + str(i + 1) for i in range(20)] + ['Y'])
    return data


# This is used to get the X image_part to the center and add bias. Mean centering is recommended to give better results and
# less worry about bias
def mean_center_normalize(X_matrix, len_input):
    # Squashing whole image_part between 0 and 1
    X_matrix = (X_matrix - np.mean(X_matrix, 0)) / np.std(X_matrix, 0)
    # Adding 1s as the first column to represent bias
    X_matrix = np.hstack((np.ones((len_input, 1)), X_matrix))
    return X_matrix


# Class to perform different types of regression
class LinearRegression:
    # Init method. Needs X image_part, Y image_part and other optional parameters
    def __init__(self, X_matrix, Y_matrix, learning_rate=0.01, tot_iterations=1500):
        self.X = X_matrix
        self.Y = Y_matrix
        self.learning_rate = learning_rate
        self.iterations = tot_iterations
        self.num_samples = len(Y_matrix)
        self.num_features = X_matrix.shape[1]
        # Mean centering given image_part
        self.X = mean_center_normalize(self.X, self.num_samples)
        # self.Y = Y_matrix[:, np.newaxis]
        # Initializing weights to a zero vector
        self.weights = np.zeros((self.num_features + 1, 1))
        self.curr_error = 1

    # Simple formula to fit image_part to naive regression. No gradient descent used. Assuming XTX is invertible
    def fit_naive_reg(self):
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.Y)
        return self

    # Using gradient descent in linear regression method
    def fit_linear_reg_gradient(self, st, test_X, test_y):
        error_store = []
        for i in range(self.iterations):
            # Finding gradient and moving towards minimization
            d = self.X.T @ (self.X @ self.weights - self.Y)
            self.weights = self.weights - (self.learning_rate / self.num_samples) * d
            if i != 0:
                curr_error = self.get_error()
                test_error = self.get_error(test_X, test_y)
                error_store.append(curr_error)
                new_df = pd.DataFrame([curr_error])
                if i % 50 == 0:
                    # print('Train Error', curr_error)
                    # print('Test Error', test_error)
                    print(i, curr_error, test_error)
                if st is not None:
                    st.add_rows(new_df)
        return error_store

    # Using simple formula for Ridge regression
    def fit_ridge_reg(self, norm_const):
        error_store = [0]
        n_samples, n_features = self.X.shape
        self.weights = np.dot(
            np.dot(np.linalg.inv(np.dot(self.X.T, self.X) + norm_const * np.identity(n_features)), self.X.T), self.Y)
        return error_store

    def fit_ridge_reg_gradient(self, norm_const, st):
        error_store = [0]
        norm_const = norm_const * self.num_samples
        n_samples, n_features = self.X.shape
        for i in range(self.iterations):
            # Finding gradient and moving towards minimization
            y_pred = self.X @ self.weights
            d = (- (2 * self.X.T.dot(self.Y - y_pred)) +
                 (2 * norm_const * self.weights)) / n_samples
            self.weights = self.weights - self.learning_rate * d
            if i != 0:
                curr_error = self.get_error()
                error_store.append(curr_error)
                new_df = pd.DataFrame([curr_error])
                if st is not None:
                    st.add_rows(new_df)
        return error_store

    # Using formulae given in class notes for Lasso regression
    def fit_lasso(self, norm_const, st):
        error_store = [0]
        norm_const = norm_const * self.num_samples
        n_samples, n_features = self.X.shape
        # calculating bias using formula. No iterations needed as this is independent
        self.weights[0] = np.sum(self.Y - np.dot(self.X[:, 1:], self.weights[1:])) / n_samples
        for i in range(self.iterations):
            for j in range(1, n_features):
                # Maintaining a copy of weights. Not actually needed
                copy_w = self.weights.copy()
                residue = self.Y - np.dot(self.X, copy_w)
                # Computing first value in numerator of formula
                first = np.dot(self.X[:, j], residue)
                # Computing second value in numerator of formula
                second = norm_const / 2
                # These are used to check for conditions
                compare = (-first + second) / np.dot(self.X[:, j].T, self.X[:, j])
                compare_neg = (-first - second) / np.dot(self.X[:, j].T, self.X[:, j])
                # Updating weights based on conditions
                if self.weights[j] > compare:
                    self.weights[j] = self.weights[j] - compare
                elif self.weights[j] < compare_neg:
                    self.weights[j] = self.weights[j] - compare_neg
                else:
                    self.weights[j] = 0
            if i != 0:
                curr_error = self.get_error()
                error_store.append(curr_error)
                new_df = pd.DataFrame([curr_error])
                if st is not None:
                    st.add_rows(new_df)
        return error_store

    # By default returns training error. Takes X and Y image_part as input
    def get_error(self, X_matrix=None, Y_matrix=None):
        # If no image_part given, calculating training error
        # mean-centering
        if X_matrix is None:
            X_matrix = self.X
        else:
            # Mean centering any input image_part as we've found weights after this
            X_matrix = mean_center_normalize(X_matrix, X_matrix.shape[0])

        if Y_matrix is None:
            Y_matrix = self.Y
        else:
            # Y_matrix = Y_matrix[:, np.newaxis]
            Y_matrix = Y_matrix

        # Using the formula to find Y values. Bias is the first weight. X has 1s in the first column
        y_pred = X_matrix @ self.weights
        # Returning scaled score for better understanding. Error is squashed between 0 and 1
        # Example: Error of 0.1 is less. Error of 0.9 is terrible
        score = (((Y_matrix - y_pred) ** 2).sum() / ((Y_matrix - Y_matrix.mean()) ** 2).sum())
        self.curr_error = score
        return score

    # Simple find_labels function.
    def predict(self, X):
        return mean_center_normalize(X, X.shape[0]) @ self.weights

    # Method exposed to return weights after training
    def get_weights(self):
        return self.weights[1:]


# Applies plain regression with single values
def Run_Regression():
    path = 'D:\\Study\\Intro_AI\\Coloring_Assignment\\Images\\reduced.png'
    rgba_image = PIL.Image.open(path)
    rgb_image = rgba_image.convert('RGB')
    img = np.array(rgb_image)
    gray = rgb2gray(img)
    left_actual = img[:, :int(img.shape[1] / 2), :]
    right_actual = img[:, int(img.shape[1] / 2):, :]
    left_half = gray[:, :int(img.shape[1] / 2)]
    right_half = gray[:, int(img.shape[1] / 2):]
    left_half_flat = left_half.reshape(-1, 1)
    left_actual = left_actual.reshape(-1, 3)
    X = left_half_flat
    colored_right = np.zeros(right_actual.shape)
    y = left_actual
    regressor = LinearRegression(X, np.asarray(y), 0.01,
                                 tot_iterations=10000)
    regressor.fit_linear_reg_gradient(None, None, None)
    print(regressor.get_error())
    pred = regressor.predict(right_half.reshape(-1, 1))
    pred = pred.reshape(right_actual.shape)
    print(pred)
    plt.imshow(pred / 255)
    plt.show()


# Run_Regression()

