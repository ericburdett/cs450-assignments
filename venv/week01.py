import sys
import pandas
import numpy
import csv
import HardCoded
from sklearn import datasets
from sklearn import model_selection
from sklearn import naive_bayes
from numpy import genfromtxt

def get_classifier():
    # return HardCoded.HardCodedClassifier()
    return naive_bayes.GaussianNB()

def main(argv):
    print("Enter filename for the data file")
    data_filename = input()
    print("Enter filename for the target")
    target_filename = input()

    data = numpy.zeros(0)
    target = numpy.zeros(0)

    if data_filename == '':
        print("Using pre-loaded dataset")
        iris = datasets.load_iris()
        data = iris.data
        target = iris.target
    else:
        try:
            raw_data = open(data_filename, "rt")
            data = numpy.genfromtxt(raw_data, delimiter= ',', dtype = 'float')
            raw_data = open(target_filename, "rt")
            target = numpy.genfromtxt(raw_data, delimiter=',', dtype='int')
        except FileNotFoundError:
            print("File not found")
            return

    data_train, data_test, target_train, target_test = model_selection.train_test_split(data, target, test_size=.30, random_state=42)

    classifier = get_classifier()
    print ("Using ", type(classifier).__name__)
    model = classifier.fit(data_train, target_train)

    prediction = model.predict(data_test)
    prediction_accuracy = prediction == target_test

    correct = numpy.count_nonzero(prediction_accuracy)
    total = numpy.size(prediction)

    print("Accuracy: ", correct, "/", total, " - ", "{:.2f}".format(correct * 100 / total), "%")

if __name__ == "__main__":
    main(sys.argv)