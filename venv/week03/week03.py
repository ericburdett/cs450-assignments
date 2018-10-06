import sys
import warnings
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def get_classifier():
    return KNeighborsClassifier(n_neighbors=11)
    # return LinearDiscriminantAnalysis()

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
    data_frame.dropna(inplace=True)

    # Separate input, target
    input = data_frame[column_names[1:]]
    target = data_frame[column_names[0]]

    # One hot transformation
    input = pd.get_dummies(input)

    # transform to numpy arrays
    return input.values, target.values


def get_autism_data():
    filename = "Autism-Adult-Data.data"
    column_names=['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                  'age', 'gender', 'ethnicity', 'jundice', 'autism', 'country_of_res', 'used_app_before', 'result', 'age_desc', 'relation',
                  'Class/ASD']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names, na_values=['?'])

    ### Strategies for filling missing data ###

    # Remove rows with missing data
    # This method gives about 76% accuracy with KNeighborsClassifier
    data_frame.dropna(inplace=True)

    # Replace missing values with the mean
    # This method gives about 73% accuracy with KNeighborsClassifier
    # data_frame.fillna(data_frame.mean(), inplace=True)

    # Replace missing values with the median
    # This method gives about 73% accuracy with KNeighborsClassifier
    # data_frame.fillna(data_frame.median(), inplace=True)

    # Replace missing values with the most_frequent value
    # This method gives about 73% accuracy with KNeighborsClassifier
    # (This code should be placed below after all other preprocessing has taken place)
    # (Imputer from sklearn was required to fill missing values with the mode)
    # input_values = input.values
    # target_values = target.values
    # imputer = Imputer(strategy='most_frequent')
    # input_values = imputer.fit_transform(input_values)

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

def test_and_print_accuracy(function, classifier):
    input, target = function()
    input_train, input_test, target_train, target_test = model_selection.train_test_split(input, target, test_size=.30, random_state=12)

    print ("Using ", type(classifier).__name__)

    k_fold = KFold(n_splits=10, random_state=12)

    result = cross_val_score(classifier, input, target, cv=k_fold, scoring='accuracy')

    print("Accuracy: {:.2f}%".format(result.mean()))

    # model = classifier.fit(input_train, target_train)
    #
    # prediction = model.predict(input_test)
    # prediction_accuracy = prediction == target_test
    #
    # correct = np.count_nonzero(prediction_accuracy)
    # total = np.size(prediction)
    #
    # print("Accuracy: ", correct, "/", total, " - ", "{:.2f}".format(correct * 100 / total), "%")


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
    # Ignore warnings so we can easily see the accuracy between the three data sets.
    # Warnings receiving during cross validation (LinearDiscriminantAnalysis) saying variables are collinear
    warnings.filterwarnings('ignore')

    print("Car Data")
    test_and_print_accuracy(get_car_data, get_classifier())

    print("\nAutism Data")
    # get_autism_data()
    test_and_print_accuracy(get_autism_data, get_classifier())

    print("\nAuto-MPG Data")
    test_and_print_comparison(get_auto_mpg_data, get_regressor(), False)


if __name__ == "__main__":
    main(sys.argv)