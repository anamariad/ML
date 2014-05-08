import sys
import random
import numpy as np
import utils as utils
import scipy.stats as stats


def estimation_step(K, means, points, stddev):
    points_size = len(points)
    expectations = np.zeros((points_size, K))
    for i in range(0, points_size):
        total = np.zeros(2)
        current_point = points[i]
        for j in range(0, K):
            total += stats.norm(means[j], stddev).pdf(current_point)
        for j in range(0, K):
            expectations[i][j] = stats.norm(means[j], stddev).pdf(current_point) / total

    return expectations


def maximization_step(K, expectations, means, points):
    points_size = len(points)
    for j in range(0, K):
        m_step_numerator = np.zeros(2)
        m_step_denominator = np.zeros(2)
        for i in range(0, points_size):
            m_step_numerator += expectations[i][j] * points[i]
            m_step_denominator += expectations[i][j]
        means.append(m_step_numerator / m_step_denominator)
    return means


def expectation_maximization(points, K, stddev):
    # The initial means are random
    old_means = []
    means = random.sample(points, K)


    while not utils.convergence(means, old_means):
        old_means = means

        # the E step
        expectations = estimation_step(K, means, points, stddev)
        # the M step
        means = maximization_step(K, expectations, old_means, points)

    return means, expectations # returns a tuple of them



# arg1: input filename
# arg2: number of clusters
input_filename = sys.argv[1]
K = int(sys.argv[2])

input_points = utils.read_points(input_filename)
stddev = np.std(input_points, axis = 0)

print input_points
# clusterization = expectation_maximization(input_points, K, stddev)

# print clusterization[0]




