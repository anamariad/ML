import sys
import getopt
import random
import numpy as np
import utils as utils


def estimation(K, means, points, stddev):
    points_size = len(points)
    expectations = np.zeros((points_size, K))
    for i in range(0, points_size):
        total = 0
        current_point = points[i]
        for j in range(0, K):
            total += np.exp(-1 / (2 * (stddev ** 2)) * (np.linalg.norm(current_point - means[j]) ** 2))
        for j in range(0, K):
            expectations[i][j] = np.exp(-1 / (2 * (stddev ** 2)) * (np.linalg.norm(current_point - means[j]) ** 2))  / total

    return expectations


def maximization(K, expectations, points):
    points_size = len(points)
    means = []
    for j in range(0, K):
        m_step_numerator = np.zeros(2)
        m_step_denominator = 0
        for i in range(0, points_size):
            m_step_numerator += expectations[i][j] * points[i]
            m_step_denominator += expectations[i][j]
        means.append(m_step_numerator / m_step_denominator)
    return means


def expectation_maximization(points, K, stddev, means):
    old_means = np.zeros(means.shape)
    expectations = None
    clusters = None
    while not utils.convergence(means, old_means):
        old_means = means

        # the E step
        expectations = estimation(K, means, points, stddev)
        # the M step
        means = np.array(maximization(K, expectations, points))
        clusters = assign_points_to_clusters(points, expectations, K)

    return means, expectations, clusters  # returns a tuple of them


def assign_points_to_clusters(points, expectations, K):
    clusters = {}
    for i in range(len(points)):
        best_cluster_key = max([(j, expectations[i][j]) for j in range(K)], key = lambda t: t[1])[0]

        if best_cluster_key in clusters:
            clusters[best_cluster_key].append(points[i])
        else:
            clusters[best_cluster_key] = [points[i]]
    return clusters

error_msg = 'em2d.py -f <inputfile> -k <number of clusters> -m <comma-separated initial K means values> -s <stddev>'
try:
    opts, args = getopt.getopt(sys.argv[1:], "f:k:m:s:t", ["file=", "means=", "stddev="])
except getopt.GetoptError:
    print error_msg
    sys.exit(2)

input_filename = None
K = 0
means = None
stddev = None

for opt, arg in opts:
    if opt in ('-f', '--file'):
        input_filename = arg
    elif opt == '-k':
        K = int(arg)
    elif opt in ('-s', '--stddev'):
        stddev = float(arg)
    elif opt in ('-m', '--means'):
        means_string = arg.split(":")
        first_mean = [float(m) for m in means_string[0].split(",")]
        second_mean = [float(m) for m in means_string[1].split(",")]
        means = np.array([first_mean, second_mean])


if input_filename is None or K == 0:
    print error_msg
    sys.exit(2)

input_points = utils.read_points(input_filename)

if stddev is None:
    stddev = np.std(input_points)

if means is None:
    means = np.array(random.sample(input_points, K))

clusterization = expectation_maximization(input_points, K, stddev, means)

centroids = clusterization[0]
expectations = clusterization[1]
clusters = clusterization[2]

print "centroids:\n {} \n expectations:\n {}".format(centroids, expectations)

utils.plot_clusters(centroids, clusters)




