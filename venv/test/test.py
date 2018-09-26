import sys
import numpy as np


def main(argv):
    list = (8, 2, 5, 4, 5, 6)
    a = np.array(list)
    print(a)

    sa = np.sort(a)
    print(sa)

    indices = np.where(a == sa[1])
    print(indices)


if __name__ == "__main__":
    main(sys.argv)