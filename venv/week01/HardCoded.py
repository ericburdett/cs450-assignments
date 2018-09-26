import numpy

class HardCodedModel:
    def predict(self, test_array):
        prediction_list = []
        
        for x in test_array:
            prediction_list.append(0)
        
        return numpy.array(prediction_list)

class HardCodedClassifier:
    def fit(self, data, target):
        return HardCodedModel()
