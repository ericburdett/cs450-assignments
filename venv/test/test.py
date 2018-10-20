import sys
import numpy as np
import math
from anytree import Node, RenderTree

def get_entropy(p1, p2):
    if p1 == 0:
        p1 = 0.0000000000000000001
    if p2 == 0:
        p2 = 0.0000000000000000001
    return (-(p1) * math.log2(p1)) + (-(p2) * math.log2(p2))

def main(argv):

    while True:
        p1 = float(input("Probability 1:"))
        p2 = float(input("Probability 2:"))

        if p1 == -1:
            break;

        print("Entropy:", get_entropy(p1, p2))

if __name__ == "__main__":
    main(sys.argv)