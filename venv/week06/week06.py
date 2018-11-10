import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import Neuron
from NeuralNetwork import NeuralNetworkClassifier
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def get_data():
    filename = "adult.data"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names)

    translated_values = {'income' : {' <=50K' : 0, ' >50K' : 1}}
    data_frame.replace(translated_values, inplace=True)

    input = data_frame[column_names[0:-1]]
    target = data_frame[column_names[-1]]

    input = pd.get_dummies(input)

    # A smaller test data set to manually verify results

    # filename = 'test.csv'
    # column_names = ['column1', 'column2', 'column3', 'column4']
    #
    # data_frame = pd.read_csv(filename, names=column_names)
    # input = data_frame[column_names[0:-1]]
    # target = data_frame[column_names[-1]]

    # include some standard scaling...
    scaler = preprocessing.StandardScaler()

    # transform to numpy arrays
    return scaler.fit_transform(input.values),  target.values
    # return input.values, target.values

def get_iris():
    filename = "iris.data"
    column_names = ["sepal length","sepal width","pedal length","pedal width","iris"]

    data_frame = pd.read_csv(filename, names=column_names)

    translated_values = {'iris' : {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}}
    data_frame.replace(translated_values, inplace=True)

    input = data_frame[column_names[0:-1]]
    targets = data_frame[column_names[-1]]

    scaler = preprocessing.StandardScaler()

    return scaler.fit_transform(input.values), targets.values

def get_autism_data():
    filename = "Autism-Adult-Data.data"
    column_names=['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                  'age', 'gender', 'ethnicity', 'jundice', 'autism', 'country_of_res', 'used_app_before', 'result', 'age_desc', 'relation',
                  'Class/ASD']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names, na_values=['?'])
    data_frame.dropna(inplace=True)

    # Replace some values with numerics
    cleanup_values = {'gender' : {'f' : 0, 'm' : 1 },
                      'jundice' : {'no' : 0, 'yes' : 1},
                      'autism' : {'no' : 0, 'yes' : 1},
                      'used_app_before' : {'no' : 0, 'yes' : 1}}
    data_frame.replace(cleanup_values, inplace=True)

    # Separate input, target
    input = data_frame[column_names[1:]]
    target = data_frame[column_names[0]]

    # One hot transformation
    input = pd.get_dummies(input)

    # Transform to numpy arrays
    return input.values, target.values

def show_accuracy(prediction, target, classifier):
    accuracy = prediction == target
    correct = np.count_nonzero(accuracy)
    total = np.size(prediction)
    print(type(classifier).__name__, ": Accuracy: ", correct, "/", total, " - ", "{:.2f}".format(correct * 100 / total), "%")

def main(args):
    data, targets = get_data()

    data_train, data_test, target_train, target_test = model_selection.train_test_split(data, targets, test_size=.3, random_state=12)

    # My Neural Network
    network = NeuralNetworkClassifier([5,5], .05)
    network.fit(data_train, target_train)
    show_accuracy(network.predict(data_test), target_test, network)

    # Sklearn's Neural Network
    libNetwork = MLPClassifier(hidden_layer_sizes=(5,5), learning_rate='adaptive', learning_rate_init=.1)
    libNetwork.fit(data_train, target_train)
    show_accuracy(libNetwork.predict(data_test), target_test, libNetwork)

    # K Nearest Neighbors
    kNeighbors = KNeighborsClassifier(n_neighbors=3)
    kNeighbors.fit(data_train, target_train)
    show_accuracy(kNeighbors.predict(data_test), target_test, kNeighbors)

    # Nieve Bayes
    bayes = GaussianNB()
    bayes.fit(data_train, target_train)
    show_accuracy(bayes.predict(data_test), target_test, bayes)

    # Decision Tree
    tree = DecisionTreeClassifier()
    tree.fit(data_train, target_train)
    show_accuracy(tree.predict(data_test), target_test, tree)

    print("show learning graph? (y/n)")
    showGraph = input()

    if (showGraph == 'y'):
        x,y = network.get_error_lists()
        plt.title("Learning Accuracy")
        plt.xlabel("Iteration (100 rows)")
        plt.ylabel("Accuracy (%)")
        plt.plot(x,y*100)
        plt.show()

if __name__ == "__main__":
    main(sys.argv)
