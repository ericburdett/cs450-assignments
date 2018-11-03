import numpy as np
import math

class NeuralNetworkClassifier:
    def __init__(self, hiddenLayerCounts):
        # Use the hidden layer counts given, and add a single node for the final output layer
        hiddenLayerCounts.append(1)
        self.layerCounts = hiddenLayerCounts

    def fit(self, data, targets):
        self.data = data
        self.targets = targets

        # Create all nodes needed for the network, given the counts
        self.create_neurons()

        # Train the network based on the data given
        self.train()


    def predict(self, data):
        output_neuron_results = []

        for row in data:
            output_neuron_results.append(self.get_output_neuron_results(row))

        return np.array(output_neuron_results)

    def train(self):
        for row in self.data:
            output = self.feed_forward(row)

            print("Final Output: ", output)

            # calculate error
            # back-propagate and update neural network

    def feed_forward(self, row):
        # Insert input values into first layer
        for neuron in self.neuronList[0]:
            neuron.values = np.insert(row, 0, -1)

        for i in range(0, len(self.neuronList) - 1):
            activation_values = []

            # Calculate activation values
            for j in range(0, len(self.neuronList[i])):
                # calculate activation
                activation_values.append(self.neuronList[i][j].activation())

            # Add bias node to front
            activation_values.insert(0, -1)

            # Insert activation values into the next layer
            for j in range(0, len(self.neuronList[i+1])):
                self.neuronList[i+1][j].values = np.array(activation_values)

        # return the activation for the final output node
        return self.neuronList[-1][0].activation()

    def create_neurons(self):
        neuronList = []
        # Iterate through each layer
        for i in range(0, len(self.layerCounts)):
            incoming_connections = 0
            # if first layer, incoming connections is equal to number of inputs
            if i == 0:
                # add one for the bias node
                incoming_connections = self.data.shape[1] + 1
            # incoming connections is equal to the # of nodes on the previous layer
            else:
                # add one for the bias node
                incoming_connections = self.layerCounts[i - 1] + 1

            neurons = []

            # Iterate through each node in the layer
            for j in range(0, self.layerCounts[i]):
                # Add neuron to the list for this layer
                neurons.append(Neuron(incoming_connections))

            # Add list of neurons for this layer to list for entire network
            if len(neurons) != 0:
                neuronList.append(neurons)

        # print("NeuronList")

        # convert to numpy array
        self.neuronList = np.array(neuronList)

class Neuron:
    def __init__(self, number_of_connections):
        # For now, we'll just initialize the weights to random values
        # This is more interesting that looking at zeros for the outputs
        self.values = np.zeros(number_of_connections)
        self.weights = np.random.uniform(-1, 1, number_of_connections)

    def activation(self):
        # multiple weights by the values, then sum
        sum = np.sum(self.values * self.weights)

        # use the sigmoid activation function
        return 1 / (1 + math.exp(-sum))

    def __repr__(self):
        return "(weights: {} values: {})".format(np.array2string(self.weights), np.array2string(self.values))





