import numpy as np
import math
from anytree import Node, RenderTree

class DecisionTreeClassifier:
    def fit(self, data, targets, column_names):
        self.data = data
        self.targets = targets
        self.column_names = column_names
        self.tree_root = None

        self._build_tree(None, data, targets, column_names)

        # build tree
        # specify "None" for currentNode when starting algorithm

        return self

    def get_root_node(self):
        return self.tree_root

    def predict(self, data):
        # do something
        return np.zeros(0)

    def _build_tree(self, currentNode, data, targets, column_names):
        if len(column_names) == 1:
            self._handle_leaf_nodes(currentNode, data, targets)
            return

        column_to_split = self._pick_column_to_split(data, targets)

        newNode = Node(column_names[column_to_split], parent=currentNode)
        column_names = np.delete(column_names, column_to_split)
        if currentNode == None:
            self.tree_root = newNode

        for dataValue in np.unique(data[column_to_split]):
            # Remove column and recurse
            self._build_tree(newNode, np.delete(data, np.s_[column_to_split], 1), targets, column_names)


    def _handle_leaf_nodes(self, currentNode, data, targets):
        for dataValue in np.unique(data):
            targetList = list(np.take(targets, np.where(data == dataValue)[0]))
            bestValue = None
            bestProbability = 0

            for targetValue in np.unique(targets):
                probability = targetList.count(targetValue) / len(targetList)
                if probability > bestProbability:
                    bestProbability = probability
                    bestValue = targetValue

            Node(bestValue, parent=currentNode)


    def _pick_column_to_split(self, data, targets):
        best_column = 0
        best_column_entropy = 1000

        for index in range(0, data.shape[1]):
            column_entropy = self._get_weighted_entropy(data[index], targets)
            # print("Entropy for column ", index, ": ", column_entropy)
            if column_entropy < best_column_entropy:
                best_column = index
                best_column_entropy = column_entropy

        return best_column

    def _get_weighted_entropy(self, dataColumn, targets):
        weighted_entropy = 0

        for dataValue in np.unique(dataColumn):
            probabilityList = []
            targetList = list(np.take(targets, np.where(dataColumn == dataValue))[0])
            dataListSize = dataColumn.tolist().count(dataValue)

            for targetValue in np.unique(targets):
                probability = targetList.count(targetValue) / len(targetList)
                probabilityList.append(probability)

            entropy = self.__get_entropy(probabilityList) * (dataListSize / len(dataColumn))

            # print("Entropy for ", dataValue, " ", self.__get_entropy(probabilityList), " Weighted: ", entropy)

            weighted_entropy += entropy

        return weighted_entropy

    def __get_entropy(self, probailityList):
        entropySum = 0

        for probability in probailityList:

            # Define log2(0) as 0
            if probability == 0:
                continue

            entropySum += (-probability * math.log2(probability))

        return entropySum
