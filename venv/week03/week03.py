import sys
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

def get_classifier():
    return KNeighborsClassifier(n_neighbors=11)

def get_regressor():
    return KNeighborsRegressor(n_neighbors=11)

def get_car_data():
    filename = "car.data"
    column_names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names)
    input = data_frame[column_names[0:-1]]
    target = data_frame[column_names[-1]]

    # One hot transformation
    input = pd.get_dummies(input)

    # transform to numpy arrays
    return input.values, target.values

def get_auto_mpg_data():
    filename = "auto-mpg.data"
    column_names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

    # Read data from csv, separate on whitespace
    data_frame = pd.read_csv(filename, names=column_names, sep='\s+', na_values=['?'])

    # Remove rows with missing data
    data_frame = data_frame.dropna()

    # Separate input, target
    input = data_frame[column_names[1:]]
    target = data_frame[column_names[0]]

    # One hot transformation
    input = pd.get_dummies(input)

    # transform to numpy arrays
    return input.values, target.values


def get_autism_data():


def test_and_print_accuracy(function, classifier):
    input, target = function()
    input_train, input_test, target_train, target_test = model_selection.train_test_split(input, target, test_size=.30, random_state=12)

    print ("Using ", type(classifier).__name__)
    model = classifier.fit(input_train, target_train)

    prediction = model.predict(input_test)
    prediction_accuracy = prediction == target_test

    correct = np.count_nonzero(prediction_accuracy)
    total = np.size(prediction)

    print("Accuracy: ", correct, "/", total, " - ", "{:.2f}".format(correct * 100 / total), "%")


def test_and_print_comparison(function, regressor, display_results):
    input, target = function()
    input_train, input_test, target_train, target_test = model_selection.train_test_split(input, target, test_size=.30, random_state=12)

    print ("Using ", type(regressor).__name__)
    model = regressor.fit(input_train, target_train)

    prediction = model.predict(input_test)

    diff_list = []

    for i in range(len(prediction)):
        diff = prediction[i] - target_test[i]
        diff_list.append(diff)

        if display_results:
            print("Predicted: ", "{:.1f}".format(prediction[i]), " Actual: ", target_test[i], " Diff: ", "{:.1f}".format(diff))

    print("Average difference: {:.2f} mpg".format(np.average(diff_list)))

def main(args):
    print("Car Data")
    test_and_print_accuracy(get_car_data, get_classifier())

    print("\nAutism Data")
    test_and_print_accuracy(get_autism_data(), get_classifier())

    print("\nAuto-MPG Data")
    test_and_print_comparison(get_auto_mpg_data, get_regressor(), False)



if __name__ == "__main__":
    main(sys.argv)