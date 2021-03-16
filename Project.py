import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Can choose between using next closing price as target, or next day's change as target
TARGET = "Next_Close"
FILENAME = "new_VOO_features_prepared_lin"


def split_data(df):
    """
    Returns train and test feature and target splits
    """
    targets = df[TARGET]
    features = df.drop(["Next_Close"], axis=1).drop(["Day_Change"], axis=1)

    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2)

    features_train = np.asarray(features_train)
    features_test = np.asarray(features_test)
    targets_train = np.asarray(targets_train)
    targets_test = np.asarray(targets_test)
  
    return features_train, features_test, targets_train, targets_test


def initialize_Weights(n):
    bias = 0
    weights = np.zeros((n, 1))
    return weights, bias


def propagate(weights, bias, f, t):
    """
    Calculates predictions, then loss, then partials for updating weights and bias for pipelined linreg model
    """

    cases = f.shape[1]

    # Forward Propagation, calculates predictions
    prediction = np.dot(weights.T, f.T) + bias
    cost = np.sum((prediction - t)**2) / (2 * cases)

    # Backward Propagation, calculates partials and loss
    bias_partial = np.sum(prediction - t) / cases
    weights_partial = np.dot(f.T, (prediction - t.T).T) / cases

    partials = {"dw":weights_partial, "db":bias_partial}

    return cost, partials


def optimize(weights, bias, f, t, iterations, learning_rate):
    """
    Trains and optimizes the model over a number of iterations and learning rate, for pipelined linreg model.
    Returns optimized weights and bias.
    """
    costs = []

    for i in range(iterations):
        cost, partials = propagate(weights, bias, f, t)

        partial_weight = partials["dw"]
        partial_bias = partials["db"]
        weights = weights - partial_weight * learning_rate
        bias = bias - partial_bias * learning_rate

        if i != 0 and i % 1000 == 0:
            costs.append(cost)

    params = {"w": weights, "b": bias}

    grads = {"dw": partial_weight, "db": partial_bias}

    return params, grads, costs


def predict(weights, bias, f):
    return np.dot(weights.T, f.T) + bias


def pipelined_linreg(features_train, features_test, targets_train, targets_test, iterations, learning_rate):
    """
    Trains a linear regression model using implemented forward and back propagation.

    """

    weights, bias = initialize_Weights(7)

    parameters, grads, costs = optimize(weights, bias, features_train, targets_train, iterations, learning_rate)
    weights = parameters["w"]
    bias = parameters["b"]

    prediction_train = predict(weights, bias, features_train)
    prediction_test = predict(weights, bias, features_test)

    train_rmse = get_RMSE(targets_train, prediction_train)
    test_rmse = get_RMSE(targets_test, prediction_test)

    print("pipelined LinReg train RMSE: ", train_rmse)
    print("pipelined LinReg test RMSE: ", test_rmse, "\n")

    return weights, bias


def get_RMSE(targets, preds):
    """
    Given a set of targets and predictions, returns the root mean square error
    """
    return np.sqrt(1 / targets.shape[0] * np.sum((preds - targets) ** 2))


def sklearn_linreg(features_train, features_test, targets_train, targets_test):
    """
    Returns a fitted sklearn linreg estimator
    """
    reg = LinearRegression().fit(features_train, targets_train)

    train_rmse = get_RMSE(targets_train, reg.predict(features_train))
    test_rmse = get_RMSE(targets_test, reg.predict(features_test))
    print("sklearn LinReg train RMSE: ", train_rmse)
    print("sklearn LinReg test RMSE: ", test_rmse, "\n")

    return reg


def closed_form_linreg(features_train, features_test, targets_train, targets_test):
    """
    Returns weights for a linreg model found using the closed form solution
    """

    weights = np.dot(np.dot(np.linalg.pinv(np.dot(features_train.T, features_train)), features_train.T), targets_train)

    prediction_train = predict(weights, 0, features_train)
    prediction_test = predict(weights, 0, features_test)

    train_rmse = get_RMSE(targets_train, prediction_train)
    test_rmse = get_RMSE(targets_test, prediction_test)

    print("Closed form LinReg solution train RMSE: ", train_rmse)
    print("Closed form LinReg solution test RMSE: ", test_rmse, "\n")

    return weights



def main():
    df = pd.read_csv(FILENAME + '.csv', index_col='Date', parse_dates=True)
    df.to_numpy()
    features_train, features_test, targets_train, targets_test = split_data(df)

    # The more iterations there are, the longer training will take but the more accurate the model will be
    iterations = 100000
    learning_rate = 1.0e-15
    weights, bias = pipelined_linreg(features_train, features_test, targets_train, targets_test, iterations, learning_rate)

    reg = sklearn_linreg(features_train, features_test, targets_train, targets_test)

    closed_form_weights = closed_form_linreg(features_train, features_test, targets_train, targets_test)

    date = "2021-3-11"
    print("A predictive example, on " + date + "  the index fund VOO closed at: ", df.loc[date]["new_VOO"])
    print("It also had the following features: ")

    features = df.loc[date].drop(["Day_Change", "Next_Close"])
    print(features)
    print()

    pipelined_pred = np.dot(np.array(features), weights) + bias
    scikitlearn_pred = reg.predict(np.array(features).reshape(1, -1))
    closed_form_pred = np.dot(closed_form_weights, np.array(features))

    print("The following are the " + TARGET + " price predictions for the following linreg implementations:")
    print("Piplined model prediction: ", pipelined_pred[0])
    print("sklearn model prediction: ", scikitlearn_pred[0])
    print("Closed form model prediction: ", closed_form_pred)
    print("Actual Price: ", df.loc[date][TARGET])


if __name__ == '__main__':
    main()

