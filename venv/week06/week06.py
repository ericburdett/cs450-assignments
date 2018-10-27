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
    input = data_frame[column_names[0:-1]] # change back to: input = data_frame[column_names[0:-1]]
    target = data_frame[column_names[-1]]


    input = pd.get_dummies(input)

    # include some standard scaling...
    scaler = preprocessing.StandardScaler()

    # transform to numpy arrays
    return scaler.fit_transform(input.values),  target.values

def main(args):
    data, targets = get_data()

    data_train, data_test, target_train, target_test = model_selection.train_test_split(data, targets, test_size=.3, random_state=12)

    network = NeuralNetworkClassifier()
    network.fit(data_train, target_train, [-1], 1)
    print(network.predict(data_test))

    # net = NeuralNetworkClassifier()
    # net.fit(data, targets)
    # print("Prediction: ", net.predict(np.array([[1,2,3,4]])))


if __name__ == "__main__":
    main(sys.argv)
