import numpy as np


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sgn(x):
    return 1 if x > 0 else -1


def vectorized_sgn(X):
    return np.vectorize(sgn)(X)


def get_key_from_weights(W):
    return np.sum((W), axis=1)


def hebian_rule(O, Y, L, W, X):
    if O * Y > 0:
        W = W - O * X
    if np.abs(W) > L:
        W = sgn(W) * L

    return W


def vectorized_hebian_rule(O, Y, L, W, X):
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            W[i][j] = hebian_rule(O, Y[i], L, W[i][j], X[i][j])

    return W


def initialize_parameters(shape, L):
    return np.random.randint(-L, L + 1, shape)


def forward_propagation(X, W):
    Z1 = np.sum((W * X), axis=1)
    Y = vectorized_sgn(Z1)
    O = np.prod(Y)

    cache = {
        "Y": Y,
        "O": O
    }

    return O, cache


def compute_cost(A2, Y):
    m = Y.shape[0]

    logs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - np.sum(logs) / m

    cost = np.squeeze(cost)

    return cost


def update_parameters(W, X, L, cache):
    Y = cache.get("Y")
    O = cache.get("O")

    W = vectorized_hebian_rule(O, Y, L, W, X)

    return W


def nn_model(K, N, L):
    O1 = 0
    O2 = 0
    W1 = initialize_parameters((K, N), L)
    W2 = initialize_parameters((K, N), L)
    for i in range(0, 100000):
        X = np.random.choice([-1, 1], (K, N))
        O1, cache1 = forward_propagation(X, W1)
        O2, cache2 = forward_propagation(X, W2)
        if O1 * O2 < 0:
            W1 = update_parameters(W1, X, L, cache1)
            W2 = update_parameters(W2, X, L, cache2)

        if i % 1000 == 0:
            print("i = {}".format(i))
            print("W1 = {}".format(W1))
            print("W2 = {}".format(W2))

    return W1, W2, O1, O2


K = 3
N = 5
L = 3

W1, W2, O1, O2 = nn_model(K, N, L)
print("W1 = {}".format(W1))
print("W2 = {}".format(W2))
print("O1 = {}".format(O1))
print("O2 = {}".format(O2))
