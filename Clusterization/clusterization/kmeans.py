import getopt
import sys
import random
import numpy as np
import utils as utils

# Lloyd's algorithm:
#
# c_1,..., c_k = randomly chosen centroids
# while (centroids do not converge):
#   S_1,...S_k = empty clusters
#   for i in range(1, n):
#       j = argmin_t (||x_i - c_t|| ^ 2)
#       add x_i to S_j
#   for j in range(1, k):
#       c_j = (1 / |S_j|) * sum(x_i, x_i from S_j)

def lloyd_kmeans(points, K):
    # The initial centroids are random
    centroids = np.array(random.sample(points, K))
    old_centroids = np.zeros(centroids.shape)

    while not utils.convergence(centroids, old_centroids):
        old_centroids = centroids
        # Assign all points to clusters
        clusters = assign_points_to_clusters(points, centroids)
        # Reevaluate centers
        centroids = np.array(recompute_centroids(clusters))
    return centroids, clusters # returns a tuple of them


def assign_points_to_clusters(points, centroids):
    clusters = {}
    for point in points:
        best_centroid = min([(i, np.linalg.norm(point - centroids[i]) ** 2) for i in range(len(centroids))],
                            key = lambda t: t[1])
        best_centroid_key = best_centroid[0]

        if best_centroid_key in clusters:
            clusters[best_centroid_key].append(point)
        else:
            clusters[best_centroid_key] = [point]
    return clusters


def recompute_centroids(clusters):
    new_centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        new_centroids.append(np.mean(clusters[k], axis = 0))
    return new_centroids


error_msg = 'kmeans.py -f <inputfile> -k <number of clusters>'
try:
    opts, args = getopt.getopt(sys.argv[1:], "f:k:", ["file="])
except getopt.GetoptError:
    print error_msg
    sys.exit(2)

input_filename = None
K = 0

for opt, arg in opts:
    if opt in ('-f', '--file'):
        input_filename = arg
    elif opt == '-k':
        K = int(arg)

if input_filename is None or K == 0:
    print error_msg
    sys.exit(2)

input_points = utils.read_points(input_filename)
clusterization = lloyd_kmeans(input_points, K)

centroids = clusterization[0]
clusters = clusterization[1]

print "centroids: {}".format(centroids)

utils.plot_clusters(centroids, clusters)
