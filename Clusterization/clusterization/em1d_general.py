import sys
import getopt
import random
import numpy as np
import utils as utils
import scipy.stats as stats


def expectation(K, means, ratios, points, stddevs):
    points_size = len(points)
    expectations = np.zeros((points_size, K))
    for i in range(points_size):
        total = 0
        current_point = points[i]
        for j in range(K):
            total += stats.norm(means[j], stddevs[j]).pdf(current_point) * ratios[j]
        for j in range(K):
            expectations[i][j] = stats.norm(means[j], stddevs[j]).pdf(current_point) * ratios[j] / total

    return expectations


def maximization(K, expectations, points):
    points_size = len(points)
    means = np.zeros(K)
    ratios = np.zeros(K)
    stddevs = np.zeros(K)
    for j in range(K):
        m_step_numerator = 0.0
        m_step_denominator = 0.0
        stddevs_numerator = 0.0
        for i in range(points_size):
            m_step_numerator += expectations[i][j] * points[i]
            m_step_denominator += expectations[i][j]
        means[j] = m_step_numerator / m_step_denominator
        ratios[j] = m_step_denominator / points_size

        for i in range(points_size):
            stddevs_numerator += expectations[i][j] * ((points[i] - means[j]) ** 2)
        stddevs[j] = stddevs_numerator / m_step_denominator
    return means, stddevs, ratios

def q_function(K, stddev, points, centroids, expectations):

    q = 0.0

    # for i in range(len(points)):
    #     for j in range(K):
    #         q += np.log(1.0 / np.sqrt(2 * np.pi * (stddevs[j] ** 2)))
    #         q += np.log(ratios[j]) - 1.0 / (2 * stddev ** 2) * expectations[i][j] * ((points[i] - centroids[j]) ** 2)

    return q

def expectation_maximization(points, K, stddev, means, ratios, threshold):
    old_means = np.zeros(means.shape)
    expectations = None
    clusters = None
    while not utils.convergence(means, old_means, threshold):

        old_means = means

        # the E step
        expectations = expectation(K, means, ratios, points, stddev)
        # the M step
        means, stddevs, ratios = maximization(K, expectations, points)

        clusters = assign_points_to_clusters(points, expectations, K)

    return means, expectations, stddevs, clusters, ratios  # returns a tuple of them

def assign_points_to_clusters(points, expectations, K):
    clusters = {}
    for i in range(K):
        clusters[i] = []

    for i in range(len(points)):
        best_cluster_key = max([(j, expectations[i][j]) for j in range(K)], key = lambda t: t[1])[0]

        clusters[best_cluster_key].append(points[i])
    return clusters

error_msg = 'em1d.py -i <inputfile> -k <number of clusters> -m <comma-separated initial K means values> ' \
            '-s <stddev> -t <threshold> -o <outputfile>'
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:k:m:s:t:o:r:", ["inputfile=", "means=", "stddevs=", "threshold=", "outputfile=", "ratios="])
except getopt.GetoptError:
    print error_msg
    sys.exit(2)

input_filename = None
K = 0
means = None
stddevs = None
threshold = None
output_filename = None
ratios = None

for opt, arg in opts:
    if opt in ('-i', '--inputfile'):
        input_filename = arg
    elif opt in ('-o', '--outputfile'):
        output_filename = arg
    elif opt == '-k':
        K = int(arg)
    elif opt in ('-s', '--stddevs'):
        stddevs_string = arg.split(',')
        stddevs = np.array([float(s) for s in stddevs_string])
    elif opt in ('-t', '--threshold'):
        threshold = float(arg)
    elif opt in ('-m', '--means'):
        means_string = arg.split(',')
        means = np.array([float(m) for m in means_string])
    elif opt in ('-r', '--ratios'):
        ratios_string = arg.split(',')
        ratios = np.array([float(r) for r in ratios_string])

if input_filename is None or K == 0:
    print error_msg
    sys.exit(2)

if threshold is None:
    threshold = 0.01

if output_filename is None:
    output_filename = "em.out"

output_file = open(output_filename, 'w')

input_points = utils.read_points(input_filename)

if stddevs is None:
    stdev_all = np.std(input_points)
    stddevs = np.array(stdev_all for _ in range(K))

if means is None:
    means = np.array(random.sample(input_points, K))

if ratios is None:
    ratios = np.array(1.0 / K for _ in range(K))


# writing the standard deviations to file
utils.print_array_to_file(stddevs, output_file)

centroids, expectations, stddevs, clusters, ratios = expectation_maximization(input_points, K, stddevs, means, ratios, threshold)

# writing the standard deviations to file
utils.print_array_to_file(stddevs, output_file)

print "centroids:\n {} \n expectations:\n {}".format(centroids, expectations)

# outputting q function to file
output_file.write(str(q_function(K, stddevs, input_points, centroids, expectations)))
output_file.write('\n')

# outputting ratios to file
utils.print_array_to_file(ratios, output_file)

# outputting centroids to file
utils.print_array_to_file(centroids, output_file)

# outputting expectations to file
utils.print_matrix_to_file(expectations, output_file)

output_file.close()

utils.plot_clusters_1d(centroids, clusters)




