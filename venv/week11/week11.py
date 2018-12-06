import sys

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier


def get_spam_data():
    filename = "spambase.data"
    column_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                    '51', '52', '53', '54', '55', '56', '57', '58']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names)

    input = data_frame[column_names[0:-1]]
    target = data_frame[column_names[-1]]

    return input.values, target.values

def get_wine_data():
    filename = "wine.data"
    column_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                    '11', '12', '13', '14']

    data_frame = pd.read_csv(filename, names=column_names)

    input = data_frame[column_names[1:]]
    target = data_frame[column_names[0]]

    return input.values, target.values

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

    # scaler = preprocessing.StandardScaler()
    # return scaler.fit_transform(input.values), target.values

    # Transform to numpy arrays
    return input.values, target.values

def get_house_data():
    filename = "house-votes-84.data"
    column_names = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
                    'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                    'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
                    'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names)
    input = data_frame[column_names[1:]] # change back to: input = data_frame[column_names[0:-1]]
    target = data_frame[column_names[0]]

    input = pd.get_dummies(input)

    # transform to numpy arrays
    return input.values, target.values

def show_accuracy(prediction, target, classifier):
    accuracy = prediction == target
    correct = np.count_nonzero(accuracy)
    total = np.size(prediction)
    print(type(classifier).__name__, ": Accuracy: ", correct, "/", total, " - ", "{:.2f}".format(correct * 100 / total), "%")

def print_accuracy(classifier, data, targets):
    k_fold = KFold(n_splits=10, random_state=12)

    result = cross_val_score(classifier, data, targets, cv=k_fold, scoring='accuracy')
    print(type(classifier).__name__, ": Accuracy: {:.2f}%".format(result.mean() * 100))


def main(args):
    data, targets = get_wine_data()

    samples = .6
    features = .6

    # Neural Network
    network = MLPClassifier(hidden_layer_sizes=(10,10,10), learning_rate='adaptive', learning_rate_init=.1, max_iter=500)
    print_accuracy(network, data, targets)

    baggingNetwork = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(10,10,10), learning_rate='adaptive', learning_rate_init=.1, max_iter=500), max_samples=samples, max_features=features)
    print_accuracy(baggingNetwork, data, targets)

    print("")

    # K Nearest Neighbors
    kNeighbors = KNeighborsClassifier(n_neighbors=5)
    print_accuracy(kNeighbors, data, targets)

    bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), max_samples=samples, max_features=features)
    print_accuracy(bagging, data, targets)

    print("")

    bayes = GaussianNB()
    print_accuracy(bayes, data, targets)

    baggingBayes = BaggingClassifier(GaussianNB(), max_samples=samples, max_features=features)
    print_accuracy(baggingBayes, data, targets)

    print("")

    # Decision Tree
    tree = DecisionTreeClassifier()
    print_accuracy(tree, data, targets)

    baggingTree = BaggingClassifier(DecisionTreeClassifier(), max_samples=samples, max_features=features)
    print_accuracy(baggingTree, data, targets)

    print("")

    # Random Forest
    forest = RandomForestClassifier(n_estimators=200)
    print_accuracy(forest, data, targets)

    # Ada Boost
    ada = AdaBoostClassifier(n_estimators=200)
    print_accuracy(ada, data, targets)


if __name__ == "__main__":
    main(sys.argv)