import numpy as np
import threading


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.random.rand(n_h, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.random.rand(n_y, 1)

    print("W1 ========= {}".format(W1))
    print("W2 ========= {}".format(W2))
    print("b1 ========= {}".format(b1))
    print("b2 ========= {}".format(b2))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters.get("W1")
    b1 = parameters.get("b1")
    W2 = parameters.get("W2")
    b2 = parameters.get("b2")

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[0]

    logs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - np.sum(logs) / m

    cost = np.squeeze(cost)

    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters.get("W1")
    b1 = parameters.get("b1")
    W2 = parameters.get("W2")
    b2 = parameters.get("b2")

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def compute_iteration(X, current_Y, desired_Y, parameters):
    A2, cache = forward_propagation(X, parameters)

    cost = compute_cost(A2, desired_Y)
    print("Cost now = {}".format(cost))

    if cost.any() > 0:
        grads = backward_propagation(parameters, cache, X, desired_Y)
        parameters = update_parameters(parameters, grads)

    current_Y = A2
    return current_Y, parameters


def nn_model(X, n_y, n_h, num_iterations=1000, print_cost=False):
    n_x = X.shape[0]

    parameters_for_first_thread = initialize_parameters(n_x, n_h, n_y)
    parameters_for_second_thread = initialize_parameters(n_x, n_h, n_y)

    Y1, cache = forward_propagation(X, parameters_for_first_thread)
    Y2, cache = forward_propagation(X, parameters_for_second_thread)
    for i in range(0, num_iterations):
        Y1, parameters_for_first_thread = compute_iteration(X, Y1, Y2, parameters_for_first_thread)
        Y2, parameters_for_second_thread = compute_iteration(X, Y2, Y1, parameters_for_second_thread)
        thread1 = threading.Thread(target=compute_iteration, args=(X, Y1, Y2, parameters_for_first_thread))
        thread2 = threading.Thread(target=compute_iteration, args=(X, Y2, Y1, parameters_for_second_thread))
        thread1.start()
        thread2.start()
    return parameters_for_first_thread, parameters_for_second_thread, Y1, Y2


model = nn_model(np.random.randn(5, 2), 4, 2)
print(model)
