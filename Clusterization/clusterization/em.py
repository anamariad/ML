import sys
import random
import numpy as np
import utils as utils

def expectation_maximization(points, K, variance):
    # The initial means are random
    old_means = tuple()
    means = random.sample(points, K)
    expectations = []
    while not utils.convergence(means, old_means):
        old_means = means
        # the E step

    return means, expectations # returns a tuple of them



# arg1: input filename
# arg2: number of clusters
input_filename = sys.argv[1]
K = int(sys.argv[2])

input_points = utils.read_points(input_filename)
variance = np.var(input_points, True)

clusterization = expectation_maximization(input_points, K, variance)



