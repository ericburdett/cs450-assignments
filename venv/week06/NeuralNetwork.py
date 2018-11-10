import numpy as np
import math

class NeuralNetworkClassifier:
    def __init__(self, hiddenLayerCounts, learningRate):
        # Use the hidden layer counts given, and add a single node for the final output layer
        hiddenLayerCounts.append(1)
        self.layerCounts = hiddenLayerCounts
        self.learningRate = learningRate

    def fit(self, data, targets):
        self.data = data
        self.targets = targets

        # Create all nodes needed for the network, given the counts
        self.create_neurons()

        # Train the network based on the data given
        self.train()

    def predict(self, data):
        output_neuron_results = []

        # uniqueTargets = np.unique(self.targets)

        predictions = []

        for row in data:
            activation_value = self.feed_forward(row)
            # prediction = uniqueTargets[0] if activation_value < .5 else uniqueTargets[1]
            prediction = 0 if activation_value < .5 else 1
            predictions.append(prediction)

        return np.array(predictions)

    def predict_from_activation(self, value):
        prediction = 0 if value < .5 else 1
        return np.array(prediction)

    def train(self):
        self.error_x = []
        self.error_y = []
        for i in range(0, len(self.data)):
            if i % 1000 == 0:
                print("Training Progress: ", i / len(self.data))

            output = self.feed_forward(self.data[i])

            # Add to our error lists so that we can visualize the learning progress later
            self.error_x.append(i)
            self.error_y.append(self.predict_from_activation(output))

            self.back_propagate(output, self.targets[i])

        # Update lists
        correct = self.error_y == self.targets
        self.error_y = []
        self.error_x = []

        for i in range(0, len(correct), 100):
            endValue = i + 100 if i + 100 < len(correct) else len(correct)
            sublist = correct[i:endValue]
            accuracy = np.count_nonzero(sublist) / len(sublist)
            self.error_x.append(i / 100)
            self.error_y.append(accuracy)

    def back_propagate(self, output, target):
        self.error_list = []
        output_layer_error_list = []

        # Output Layer Error
        error = output * (1 - output) * (output - target)

        output_layer_error_list.append(error)

        self.error_list.append(output_layer_error_list)

        # Hidden Layer Error

        # Each Layer
        for i in range(len(self.neuronList) - 2, -1, -1):
            layer_error_list = []

            # Each Node within a layer
            for j in range(0, len(self.neuronList[i])):
                neuron = self.neuronList[i][j]
                activation_value = neuron.activation()

                sum = 0
                # calculate sum...
                for k in range(0, len(self.neuronList[i + 1])):
                    upper_layer_error = self.error_list[len(self.neuronList) - i - 2][k]
                    neuron = self.neuronList[i + 1][k]
                    weight = neuron.weights[i]

                    sum += weight * upper_layer_error

                neuron_error = activation_value * (1 - activation_value) * sum

                # Add the neuron's error to the layer error list
                layer_error_list.append(neuron_error)

            # Add the layer's error to the master error list
            self.error_list.append(layer_error_list)

        # Update the weights based on the calculated errors

        # Each layer
        for i in range(len(self.neuronList) - 1, -1, -1):

            # Each neuron in a layer
            for j in range(0, len(self.neuronList[i])):
                neuron = self.neuronList[i][j]

                error = self.error_list[len(self.neuronList) - i - 1][j]

                # Each weight in a neuron
                for k in range(0, len(neuron.weights)):
                    neuron.weights[k] -= self.learningRate * error * self.valueList[i][k]

    def feed_forward(self, row):
        self.valueList = []

        self.valueList.append(np.insert(row, 0, -1).tolist())

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

            self.valueList.append(activation_values)

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

        # convert to numpy array
        self.neuronList = np.array(neuronList)

    def get_error_lists(self):


        return np.array(self.error_x), np.array(self.error_y)

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





