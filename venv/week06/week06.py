import sys
import pandas as pd
import numpy as np
from NeuralNetwork import Neuron
from NeuralNetwork import NeuralNetworkClassifier
from sklearn import model_selection
from sklearn import preprocessing

def get_data():
    filename = "adult.data"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names)
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

def main(args):
    data, targets = get_data()

    data_train, data_test, target_train, target_test = model_selection.train_test_split(data, targets, test_size=.3, random_state=12)

    network = NeuralNetworkClassifier([2])
    network.fit(data_train, target_train)
    prediction = network.predict(data_test)
    print("Predictions: ", prediction)
    print("Actual: ", target_test)

    prediction_accuracy = prediction == target_test
    correct = np.count_nonzero(prediction_accuracy)
    total = np.size(prediction)
    print(type(network).__name__, ": Accuracy: ", correct, "/", total, " - ", "{:.2f}".format(correct * 100 / total), "%")

if __name__ == "__main__":
    main(sys.argv)
