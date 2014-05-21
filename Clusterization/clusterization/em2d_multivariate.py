from cmath import sqrt
import sys
import getopt
import random
import numpy as np
from numpy.matrixlib.defmatrix import matrix
import utils as utils


def multivariate_distribution(cov, current_point, mean):
    mean_point = matrix(current_point - mean)
    return 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov)) \
           * np.exp(-1 / 2 * mean_point.T * np.linalg.inv(cov) * mean_point)


def estimation(K, means, points, cov):
    points_size = len(points)
    expectations = np.zeros((points_size, K))

    for i in range(points_size):
        total = 0
        current_point = points[i]
        for j in range(K):
            total += multivariate_distribution(cov[j], current_point, means[j])
        for j in range(K):
            expectations[i][j] = multivariate_distribution(cov[j], current_point, means[j]) / total

    return expectations


def maximization(K, expectations, points):
    points_size = len(points)
    means = []
    cov = []
    for j in range(K):
        means_numerator = np.zeros(2)
        m_step_denominator = 0
        for i in range(points_size):
            means_numerator += expectations[i][j] * points[i]
            m_step_denominator += expectations[i][j]

        # updating the mean
        curr_mean = means_numerator / m_step_denominator
        means.append(curr_mean)

        # updating the covariance
        cov_numerator = np.zeros(2)
        for i in range(points_size):
            cov_numerator += expectations[i][j] * (points[i] - curr_mean) * (points[i] - curr_mean).T

        cov.append(cov_numerator / m_step_denominator)


    return np.array(means), cov


def expectation_maximization(points, K, means):
    old_means = np.zeros(means.shape)
    transposed_points = points.T
    cov = [np.cov(transposed_points) for _ in range(K)]

    while not utils.convergence(means, old_means):
        old_means = means

        # the E step
        expectations = estimation(K, means, points, cov)
        # the M step
        means, cov = maximization(K, expectations, points)

    return means, expectations  # returns a tuple of them


error_msg = 'em2d_multivariate.py -f <inputfile> -k <number of clusters> -m <comma-separated initial K means values>'
try:
    opts, args = getopt.getopt(sys.argv[1:], "f:k:m:", ["file=", "means="])
except getopt.GetoptError:
    print error_msg
    sys.exit(2)

input_filename = None
K = 0
means = None

for opt, arg in opts:
    if opt in ('-f', '--file'):
        input_filename = arg
    elif opt == '-k':
        K = int(arg)
    elif opt in ('-m', '--means'):
        means_string = arg.split(":")
        first_mean = [float(m) for m in means_string[0].split(",")]
        second_mean = [float(m) for m in means_string[1].split(",")]
        means = np.array([first_mean, second_mean])


if input_filename is None or K == 0:
    print error_msg
    sys.exit(2)

input_points = utils.read_points(input_filename)

if means is None:
    means = np.array(random.sample(input_points, K))



clusterization = expectation_maximization(input_points, K, means)

print "centroids: {} \n expectations:\n {}".format(clusterization[0], clusterization[1])




