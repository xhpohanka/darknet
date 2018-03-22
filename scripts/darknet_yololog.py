#!/usr/bin/python2

import sys
import re
import numpy as np
from matplotlib import pyplot as plt


def main(argv):
    fname = argv[0]

    l = []

    with open(fname) as f:
        matchchars = re.compile('[a-z\ ]*')
        for line in f:
            if re.findall('^[0-9]+:', line):
                line = line.replace(':', ',')
                line = matchchars.sub('', line)
                l.append(np.fromstring(line, sep=','))

    r = np.asarray(l)

    plt.plot(r[:, 0], r[:, 1])
    plt.plot(r[:, 0], r[:, 2])
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
