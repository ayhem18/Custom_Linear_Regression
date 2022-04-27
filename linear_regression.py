import math

import numpy as np
from numpy import linalg as lg


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.intercept = 0.0
        self.coefficients = None
        self.complete_coefficients = None

    def get_x_matrix(self, x_dataframe):
        # the number of columns: the number of independent variables
        number_columns = x_dataframe.shape[1]
        # it is more convenient to produce X^T, then transpose it.
        if self.fit_intercept:
            number_rows = x_dataframe.shape[0]
            # create the column of 1's
            X = [[1 for _ in range(number_rows)]]
            # add each column as a row to the variable X
            for i in range(number_columns):
                X.append(x_dataframe.iloc[:, i].tolist())
            # X now represents the actual variable X in our mathematical model
            X = np.transpose(np.array(X))

        else:
            X = np.transpose(
                np.array([np.array(x_dataframe.iloc[:, i].tolist()) for i in range(number_columns)]))
        return X

    def fit(self, x_dataframe, y_series):
        X = self.get_x_matrix(x_dataframe)

        X_T = np.transpose(X)
        Y = np.array([np.array([y]) for y in y_series.tolist()])

        B = lg.inv(X_T @ X) @ X_T @ Y

        if self.fit_intercept:
            self.intercept = B[0]
            self.coefficients = B[1:]
        else:
            self.coefficients = B
        self.complete_coefficients = B

    def predict(self, x_dataframe):
        X = self.get_x_matrix(x_dataframe)
        number_rows = X.shape[0]
        predictions = np.array(
            [X[i] @ self.complete_coefficients for i in range(number_rows)])
        return predictions

    def mse(self, y_array, prediction_array):
        assert len(y_array) == len(prediction_array)

        y_list = [y[0] for y in y_array]
        prediction_list = [p[0] for p in prediction_array]

        error = 0.0
        for i in range(len(y_list)):
            error += (y_list[i] - prediction_list[i]) ** 2

        return error / len(y_array)

    def rmse(self, y_array, prediction_array):
        return math.sqrt(self.mse(y_array, prediction_array))

    def r2_score(self, y_array, prediction_array):
        """
        This method calculates the R^2 error metrics between the actual results and a set of predictions
        :return: float: the error estimation
        """
        assert len(y_array) == len(prediction_array)

        y_list = [y[0] for y in y_array]

        y_mean_value = np.mean(np.array(y_list))

        mean_difference = 0.0
        for y in y_list:
            mean_difference += (y - y_mean_value) ** 2

        return 1 - (len(y_array) * self.mse(y_array, prediction_array) / mean_difference)

    def coefficients_1d(self):
        return np.array([array[0] for array in self.coefficients])
