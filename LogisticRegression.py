import os
import pickle

import numpy as np
from scipy import optimize
from sklearn import datasets

"""
a dataset of features(X) or targets(Y), or theta(theta), are
all represented as column vectors

example:
features = [ [ 2 , 3 , 5]
             [ 3 , 4 , 8]
             ]
We have three datasets, each with two features.
Dataset 1: [ 2 , 3 ]
Dataset 2: [ 3 , 4 ]
Dataset 3: [ 5 , 8 ]

"""


class logistic_regression:
    def __init__(self, learning_rate, lamb):
        self.learning_rate = learning_rate
        self.lamb = lamb

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def cost(self, theta, X, Y):
        number_of_datasets = X.shape[1]
        hypothesis = self.sigmoid(theta.T * X)  # row vector
        """
        Logarithm of 0 is infinity, so to prevent this we
        take .0001 from all hypothesis values.
        """
        for i in range(hypothesis.shape[1]):
            if (hypothesis[0, i] == 0):
                hypothesis[0, i] = hypothesis[0, i] + .0001
            elif (hypothesis[0, i] == 1):
                hypothesis[0, i] = hypothesis[0, i] - .0001

        cost_when_target_is_1 = np.log(hypothesis) * Y
        """
        Since any 
        hypothesis can be > 1 after adding .0001, we take
        absolute value of (1.0-hypothesis) to prevent taking
        the log of negative values.
        """
        cost_when_target_is_0 = np.log(abs(1.0 - hypothesis)) * (1 - Y)
        cost = (-1 / number_of_datasets) * (cost_when_target_is_0 + cost_when_target_is_1)
        return cost[0, 0]

    def cost_regularized(self, theta, X, Y):
        number_of_datasets = X.shape[1]
        cost = self.cost(theta, X, Y)
        regularization_cost = self.lamb / (2 * number_of_datasets) * np.sum(np.power(theta, 2))
        return cost + regularization_cost

    # work in progress
    def derivative_of_cost(self, theta, X, Y, theta_number):
        num_of_datasets = X.shape[1]
        hypothesis = self.sigmoid(theta.T * X)
        derivative = X[theta_number] * (hypothesis.T - Y) * (1 / num_of_datasets)
        return derivative

    def derivative_of_cost_regularized(self, theta, X, Y, theta_number):
        num_of_datasets = X.shape[1]
        hypothesis = self.sigmoid(theta.T * X)
        derivative = X[theta_number] * (hypothesis.T - Y) * (1 / num_of_datasets) + (self.lamb / num_of_datasets) * (
            theta[theta_number])
        return derivative

    def descent(self, theta, X, Y, iterations):
        X = np.matrix(X)
        Y = np.matrix(Y)
        theta = np.matrix(theta)
        num_of_features = len(theta)
        for i in range(1, iterations):
            # if (ask):
            #     change_learning_rate_boolean = input("Do you want to change the learning rate? Y for yes , N for no:")
            #     if (change_learning_rate_boolean == "y"):
            #         change_learning_rate_to = input("Enter the new learning rate: ")
            cost = self.cost(theta, X, Y)
           # print("iteration = {:d} , cost = {:f}".format(i, cost))
            error_terms = np.empty([num_of_features, 1])  # learning rate * derivative of error function

            for x in range(num_of_features):
                error_terms[x] = self.learning_rate * self.derivative_of_cost(theta, X, Y, x)

            theta = np.subtract(theta, error_terms)

        return theta

    def descent_regularized(self, theta, X, Y, iterations):

        X = np.matrix(X)
        Y = np.matrix(Y)
        theta = np.matrix(theta)

        num_of_features = len(theta)
        for i in range(1, iterations):
            # if (ask):
            #     change_learning_rate_boolean = input("Do you want to change the learning rate? Y for yes , N for no:")
            #     if (change_learning_rate_boolean == "y"):
            #         change_learning_rate_to = input("Enter the new learning rate: ")
            cost = self.cost_regularized(theta, X, Y)
            #print("iteration = {:d} , cost = {:f}".format(i, cost))
            error_terms = np.empty([num_of_features, 1])  # learning rate * derivative of error function

            for x in range(num_of_features):
                error_terms[x] = self.learning_rate * self.derivative_of_cost_regularized(theta, X, Y, x)

            theta = np.subtract(theta, error_terms)

        return theta

    def train(self, theta, X, Y):
        params0 = theta
        res = optimize.minimize(self.cost, params0, method='BFGS', args=(X, Y))
        print(res)

    def training_results(self, theta, X_testing, Y_testing):

        theta = np.matrix(theta)
        X_testing = np.matrix(X_testing)
        Y_testing = np.matrix(Y_testing)


        incorrect = correct = 0;

        hypothesis = self.sigmoid(theta.T * X_testing)

        for i in range(hypothesis.shape[1]):
            if (hypothesis[0, i] >= .5):
                hypothesis[0, i] = 1
            else:
                hypothesis[0, i] = 0
            if (hypothesis[0, i] == Y_testing[i, 0]):
                correct += 1
            else:
                incorrect += 1

        print("correct: {:d} ".format(correct))
        print("incorrect: {:d} ".format(incorrect))

    def training_results1(self, theta, X_testing, Y_testing):
        theta = np.matrix(theta)
        X_testing = np.matrix(X_testing)
        Y_testing = np.matrix(Y_testing)
        incorrect = correct = 0;

        hypothesis = self.sigmoid(theta.T * X_testing)
        hyp = np.argmax(hypothesis,axis=0)
        hyp = hyp.T
        print(Y_testing.shape)
        print(hyp.shape)
        print(hyp[:,:100])

        for i in range(Y_testing.shape[0]):
            if(hyp[i,0] == Y_testing[i,0]):
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        print("correct: {:d} ".format(correct))
        print("incorrect: {:d} ".format(incorrect))


def classify_iris():
    theta = np.random.rand(4, 3)
    iris = datasets.load_iris()
    X = iris.data[:, :]
    Y = iris.target
    num_of_datasets = len(Y)
    Y = Y.reshape(num_of_datasets,1)
    length_of_each_fold = (int)(Y.shape[0] / 5)  # each fold of cross validation, equally partitioned
    lr = logistic_regression(.01, .5)

    print(num_of_datasets)
    print(length_of_each_fold )

    # for each iris type
    for a in range(3):
        # intial folds
        test_start_index = 0
        test_end_index = length_of_each_fold
        train_start_index = test_end_index
        train_end_index = num_of_datasets
        # cross validation
        for g in range(5):
            print("iteration %d"%(g))
            train_class1 = np.asmatrix(
                [X[i] for i in range(test_start_index) or range(train_start_index, train_end_index) if Y[i] == a]).T
            target_class1 = np.ones((train_class1.shape[1], 1))
            train_class0 = np.asmatrix(
                [X[i] for i in range(test_start_index) or range(train_start_index, train_end_index) if Y[i] != a]).T
            target_class0 = np.zeros((train_class0.shape[1], 1))
            test_class1 = np.asmatrix([X[i] for i in range(test_start_index, test_end_index) if Y[i] == a]).T
            test_target_class1 = np.ones((test_class1.shape[1], 1))
            test_class0 = np.asmatrix([X[i] for i in range(test_start_index, test_end_index) if Y[i] != a]).T
            test_target_class0 = np.zeros((test_class0.shape[1], 1))
            theta_a = np.asmatrix([row[a] for row in theta]).T

            theta_a = lr.descent_regularized(theta_a, train_class0, target_class0, 3)
            #theta_a = lr.descent_regularized(theta_a, train_class1, target_class1, 3)

            for d in range(0, 4):
                theta[d][a] = theta_a[d][0]

            test_start_index = test_end_index
            test_end_index = test_end_index + length_of_each_fold
            train_start_index = test_end_index
            # lr.training_results(theta_a, test_class0, test_target_class0)
            # lr.training_results(theta_a, test_class1, test_target_class1)
    lr.training_results1(theta, X.T, Y)

def classify_digits():
    # theta = np.zeros(
    #     (64, 10))  # each column i , represents its respective digit. each digit has 64 weights initialized at zero

    theta = np.random.rand(64, 10)
    digits = datasets.load_digits()
    #X and Y are row datasets, and will be transposed during cross validation
    X = digits.images[:1795, :]
    Y = digits.target[:1795]
    X = (X.reshape(1795, -1))  # turn 3d features vector into 2d, by flattening 8x8 array to 64 array
    Y = Y.reshape(Y.shape[0], 1)
    length_of_each_fold = (int)(Y.shape[0] / 5)  # each fold of cross validation, equally partitioned
    num_of_datasets = Y.shape[0]
    lr = logistic_regression(.01, .5)


    # for the weights of each digit
    for a in range(10):
        # intial folds
        test_start_index = 0
        test_end_index = length_of_each_fold
        train_start_index = test_end_index
        train_end_index = num_of_datasets
        # cross validation
        for g in range(5):

            train_class1 = np.asmatrix(
                [X[i] for i in range(test_start_index) or range(train_start_index, train_end_index) if Y[i] == a]).T
            target_class1 = np.ones((train_class1.shape[1], 1))
            train_class0 = np.asmatrix(
                [X[i] for i in range(test_start_index) or range(train_start_index, train_end_index) if Y[i] != a]).T
            target_class0 = np.zeros((train_class0.shape[1], 1))
            test_class1 = np.asmatrix([X[i] for i in range(test_start_index, test_end_index) if Y[i] == a]).T
            test_target_class1 = np.ones((test_class1.shape[1], 1))
            test_class0 = np.asmatrix([X[i] for i in range(test_start_index, test_end_index) if Y[i] != a]).T
            test_target_class0 = np.zeros((test_class0.shape[1], 1))
            theta_a = np.asmatrix([row[a] for row in theta]).T

            theta_a = lr.descent_regularized(theta_a, train_class0, target_class0, 3)
            theta_a = lr.descent_regularized(theta_a, train_class1, target_class1, 3)

            for d in range(0, 64):
                theta[d][a] = theta_a[d][0]

            print(test_start_index)
            print(test_end_index)
            print(train_start_index)
            print(train_end_index)

            test_start_index = test_end_index
            test_end_index = test_end_index + length_of_each_fold
            train_start_index = test_end_index
            #train_end_index = test_start_index
            # lr.training_results(theta_a, test_class0, test_target_class0)
            # lr.training_results(theta_a, test_class1, test_target_class1)
    lr.training_results1(theta, X.T , Y)

def classify_cifar():
    cifar_10_dir = 'cifar-10-batches-py/'
    xs = []
    ys = []

    for i in range(1, 6):
        batch_dir = os.path.join(cifar_10_dir, 'data_batch_%d' % (i,))
        with open(batch_dir, 'rb') as f:
            dict = pickle.load(f, encoding='latin1')
            X = dict['data']
            X = np.matrix(X)
            print(X.shape)
            xs.append(X)
            Y = dict['labels']
            Y = np.matrix(Y)

            ys.append(Y.T)
    Xtr = np.concatenate(xs).T
    Ytr = np.concatenate(ys)
    print(Xtr.shape)
    print(Ytr.shape)


def main():
    classify_iris()
    #classify_digits()
    # classify_cifar()


if __name__ == "__main__":
    main()
