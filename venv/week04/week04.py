import sys
import numpy as np
import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier
from anytree import Node, RenderTree

def get_data():
    # filename = "house-votes-84.data"
    # column_names = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
    #                 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
    #                 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
    #                 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']

    filename = "test.csv"
    column_names = ['Class', 'Country Music', 'Religious', 'Gender']

    # Read data from csv
    data_frame = pd.read_csv(filename, names=column_names)
    input = data_frame[column_names[1:]] # change back to: input = data_frame[column_names[0:-1]]
    target = data_frame[column_names[0]]

    # transform to numpy arrays
    return input.values, target.values, column_names[1:]

def main(args):
    data, targets, column_names = get_data()

    classifier = DecisionTreeClassifier()
    classifier.fit(data, targets, column_names)
    root = classifier.get_root_node()
    print(root)

    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))



if __name__ == "__main__":
    main(sys.argv)