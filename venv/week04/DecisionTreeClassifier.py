import numpy as np
import math
from anytree import Node, RenderTree

class DecisionTreeClassifier:
    def fit(self, data, targets, column_names):
        self.data = data
        self.targets = targets
        self.column_names = column_names
        self.tree_root = Node([None, None])

        self._build_tree(self.tree_root, data, targets, column_names)

        return self

    def get_root_node(self):
        return self.tree_root

    def predict(self, data):
        prediction_set = []
        for row in data:
            prediction_set.append(self._make_prediction(row))

        return prediction_set

    def _make_prediction(self, row):
        node = self.tree_root

        while not node.is_leaf:
            node_name = node.name[1]
            row_value = self._get_row_value(row, node_name)
            node = self._pick_next_node(node, row_value)

        return node.name[1]

    def _get_row_value(self, row, column_name):
        index = self.column_names.index(column_name)
        return row[index]

    def _pick_next_node(self, node, row_value):
        for child in node.children:
            if child.name[0] == row_value:
                return child

        # Default case, if none are found, pick the first child
        return node.children[0]

    def _build_tree(self, currentNode, data, targets, column_names):

        # Find the next column to split
        column_to_split = self._pick_column_to_split(data, targets)

        # Set the value in the tree of the column to split
        currentNode.name[1] = column_names[column_to_split]

        # Remove the column name from this set since it it's already been used
        column_names = np.delete(column_names, column_to_split)

        column_data = data[:,column_to_split]

        # Create n number of branches depending on how many unique values there are in the chosen column
        for dataValue in np.unique(column_data):

            # Include only rows with this dataValue in the target set and the data set
            newTargets = targets[np.where(column_data == dataValue)]
            newData = data[np.where(column_data == dataValue)]

            # Delete column from data set.
            newData = np.delete(newData, np.s_[column_to_split], 1)

            # Create new Node that represents the edge. The next column name will be set on next iteration
            newNode = Node([dataValue, None], parent=currentNode)

            # If there are no more elements to split on, then pick the most common class
            # Or if all remaining rows are of one class, then pick that class
            if (len(column_names) == 0) or (len(np.unique(newTargets)) == 1):
                class_choice = self._pick_most_common_element(newTargets)
                newNode.name[1] = class_choice
            # Recurse and continue to build tree if we have more elements to split
            else:
                self._build_tree(newNode, newData, newTargets, column_names)

    def _pick_most_common_element(self, array):
        (values, counts) = np.unique(array, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def _pick_column_to_split(self, data, targets):
        best_column = 0
        best_column_entropy = 1000

        for index in range(0, data.shape[1]):
            column_entropy = self._get_weighted_entropy(data[:,index], targets)
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
