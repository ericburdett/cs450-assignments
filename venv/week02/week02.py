import sys
import pandas
import numpy
from sklearn import datasets
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from KNearestNeighbors import KNearestNeighbors

def get_classifier():
    return KNearestNeighbors(k=5)
    # return KNeighborsClassifier(n_neighbors=3)

def main(argv):
    data = numpy.zeros(0)
    target = numpy.zeros(0)

    print("Using pre-loaded dataset")
    iris = datasets.load_iris()
    data = StandardScaler().fit_transform(iris.data) # Standardize, so that we can work with different types of data
    target = iris.target

    print("Targets: ", target)

    data_train, data_test, target_train, target_test = model_selection.train_test_split(data, target, test_size=.30, random_state=42)

    for i in range(1,100):
        print("K=", i)
        print_accuracy(KNearestNeighbors(k=i), data_test, data_train, target_test, target_train)
        print_accuracy(KNeighborsClassifier(n_neighbors=i), data_test, data_train, target_test, target_train)

def print_accuracy(classifier, data_test, data_train, target_test, target_train):
    model = classifier.fit(data_train, target_train)
    prediction = model.predict(data_test)
    prediction_accuracy = prediction == target_test
    correct = numpy.count_nonzero(prediction_accuracy)
    total = numpy.size(prediction)
    print(type(classifier).__name__, ": Accuracy: ", correct, "/", total, " - ", "{:.2f}".format(correct * 100 / total), "%")

if __name__ == "__main__":
    main(sys.argv)