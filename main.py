import math

import pandas as pd
import linear_regression as lr
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def compare_models(file_name, explanatory_variable_names, dependent_variable_name):
    # load the data into the dataframe
    data = pd.read_csv(file_name)

    # initialize a linear regression with intercept
    model = LinearRegression(fit_intercept=True)

    custom_model = lr.CustomLinearRegression(fit_intercept=True)

    # train the models
    model.fit(data[explanatory_variable_names], data[dependent_variable_name])
    custom_model.fit(data[explanatory_variable_names], data[dependent_variable_name])

    # predict on the same training data
    model_predictions = model.predict(data[explanatory_variable_names])
    custom_model_predictions = custom_model.predict(data[explanatory_variable_names])

    # evaluate both models: rmse
    model_rmse = math.sqrt(mean_squared_error(data[dependent_variable_name], model_predictions))
    y_array = np.array([np.array([y]) for y in data[dependent_variable_name].tolist()])

    custom_model_rmse = custom_model.rmse(y_array, custom_model_predictions)

    # evaluate both models: r2_score
    model_r2_score = r2_score(data[dependent_variable_name], model_predictions)
    custom_model_r2_score = custom_model.r2_score(y_array, custom_model_predictions)

    # compare between the two models:
    differences = {'Intercept': model.intercept_ - custom_model.intercept,
                   'Coefficient': model.coef_ - custom_model.coefficients_1d(),
                   'R2': model_r2_score - custom_model_r2_score,
                   'RMSE': model_rmse - custom_model_rmse}
    print(differences)


if __name__ == '__main__':
    explanatory_variable_names = ['f1', 'f2', 'f3']
    dependent_variable_name = 'y'
    file_name = "test_data.csv"

    compare_models(file_name, explanatory_variable_names, dependent_variable_name)
