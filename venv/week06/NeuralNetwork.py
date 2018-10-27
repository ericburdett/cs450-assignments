import numpy as np

class NeuralNetworkClassifier:
    def fit(self, data, targets, bias, layers):
        self.data = data
        self.targets = targets
        self.bias = bias
        self.layers = layers

        self.create_neurons()

    def predict(self, data):
        output_neuron_results = []

        for row in data:
            output_neuron_results.append(self.get_output_neuron_results(row))

        return np.array(output_neuron_results)

    def get_output_neuron_results(self, row):
        output = []
        layerNum = 0

        for neuron in self.neurons:
            # Prepend the bias node to the given row
            neuron.values = np.insert(row,0, self.bias[0])
            output.append(neuron.does_fire())

        return np.array(output)

    def create_neurons(self):
        # this is also the number of columns in the data, +1 to account for bias
        number_of_edges_per_neuron = self.data.shape[1] + 1

        self.output_names = []
        self.neurons = []

        # Create the output nodes, 1 for each output
        for target_name in np.unique(self.targets):
            # Store the names in the output list
            self.output_names.append(target_name)

            # Create a neuron, initialize weights and values to 0,
            # and add it to the neuron list
            self.neurons.append(Neuron(number_of_edges_per_neuron))

    def train(self):
        # do something
        return

class Neuron:
    def __init__(self, number_of_connections):
        # For now, we'll just initialize the weights to random values
        # This is more interesting that looking at zeros for the outputs
        self.values = np.zeros(number_of_connections)
        self.weights = np.random.uniform(-1, 1, number_of_connections)

    def does_fire(self):
        # multiple weights by the values, then sum
        # if the sum exceeds the threshold of 0, then the neuron fires
        return np.sum(self.values * self.weights) > 0

    def __repr__(self):
        return "(weights: {} values: {})".format(np.array2string(self.weights), np.array2string(self.values))
