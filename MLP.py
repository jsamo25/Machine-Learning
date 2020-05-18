import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def Data_Generation():
    X, Y= load_svmlight_file("mushrooms.txt")
    print("the size of the data set is:", len(Y))
    x_train, x_test, y_train, y_test = \
        train_test_split(X, Y, test_size= 0.1)
    return x_train, x_test, y_train, y_test

def NN_MLP(x_test, x_train, y_train, y_test, alpha, r_state):
    mlp = MLPClassifier(hidden_layer_sizes=(120, 100, 50),
                        activation='logistic',
                        alpha= alpha, max_iter = 1000,
                        random_state= r_state)
    mlp.fit(x_train, y_train) #y_pred = mlp.predict(x_test)

    train_score = mlp.score(x_train, y_train)
    test_score = mlp.score(x_test, y_test)
    return train_score, test_score, mlp


def Error():
    x_train, x_test, y_train, y_test = Data_Generation()
    alpha_vec, runs = np.logspace(-10, 10, 30), 5
    Train, Test, A, B = [], [], [], []
    #A & B are only control variables.
    for i in range(len(alpha_vec)):
        for r in np.arange(runs):
            train_score, test_score, mlp = \
                NN_MLP(x_test, x_train, y_train, y_test, alpha_vec[i], r)
            A = np.append(A, train_score)
            B = np.append(B, test_score)
        Train = np.append(Train, np.mean(A))
        Test = np.append(Test, np.mean(B))

    plt.grid(True)
    plt.semilogx(alpha_vec, 1-Train)
    plt.semilogx(alpha_vec, 1-Test)
    plt.legend(['Training', 'Test'])
    plt.ylabel("Error")
    plt.xlabel("Regularization (alpha)")
    plt.show()

Error()