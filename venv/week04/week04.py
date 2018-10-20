import sys
import numpy as np
import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn import model_selection
from sklearn import tree
from anytree import Node, RenderTree

def get_data(perform_one_hot):
    filename = "house-votes-84.data"
    column_names = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
                    'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                    'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
                    'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names)
    input = data_frame[column_names[1:]] # change back to: input = data_frame[column_names[0:-1]]
    target = data_frame[column_names[0]]

    if (perform_one_hot):
        input = pd.get_dummies(input)

    # transform to numpy arrays
    return input.values, target.values, column_names[1:]

def get_prediction_numbers(predictions, target_test):
    prediction_accuracy = predictions == target_test
    correct = np.count_nonzero(prediction_accuracy)
    total = len(prediction_accuracy)

    return correct, total

def main(args):

    # sklearn's classifier - Requires one-hot transformation
    sk_data, sk_targets, sk_column_names = get_data(True)
    data_train, data_test, target_train, target_test = model_selection.train_test_split(sk_data, sk_targets, test_size=.30, random_state=42)

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(data_train, target_train)
    predictions = classifier.predict(data_test)

    # Get prediction numbers
    sk_correct, sk_total = get_prediction_numbers(predictions, target_test)

    # My Classifier - Without one-hot transformation
    data, targets, column_names = get_data(False)
    data_train, data_test, target_train, target_test = model_selection.train_test_split(data, targets, test_size=.30, random_state=42)

    classifier = DecisionTreeClassifier()
    classifier.fit(data_train, target_train, column_names)
    predictions = classifier.predict(data_test)

    # Get prediction numbers
    correct, total = get_prediction_numbers(predictions, target_test)

    # Display the tree
    root = classifier.get_root_node()
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))

    print("\nMy Classifier:")
    print("Total ({} / {}) accuracy: {:.2f}".format(correct, total ,correct / total))

    print("\nSklearn's Classifier")
    print("Total ({} / {}) accuracy: {:.2f}".format(sk_correct, sk_total ,sk_correct / sk_total))

if __name__ == "__main__":
    main(sys.argv)