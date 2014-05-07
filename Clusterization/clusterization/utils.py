__author__ = 'Annouk'

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

def read_points(csv_file_name):
    points = np.loadtxt(csv_file_name, delimiter = ',')
    return points


def plot_clusters(centroids, clusters):
    # we assume centroids is a list of points and clusters is a dictionary of arrays
    # we transpose them, so we could get 2 different arrays, representing the X and Y coordinates
    clusters_no = len(centroids)

    centroids_transposed = np.array(centroids).transpose()
    x_centroids = centroids_transposed[0]
    y_centroids = centroids_transposed[1]

    for i in range(clusters_no):
        cluster = np.array(clusters[i]).transpose()
        x_cluster = cluster[0]
        y_cluster = cluster[1]

        # setting a different color for each cluster
        plt.scatter(x_cluster, y_cluster, c = np.random.rand(1, 5))

    plt.scatter(x_centroids, y_centroids, c = 'r')
    plt.show()


def convergence(centroids, oldcentroids):
    return set([tuple(c) for c in centroids]) == set([tuple(c) for c in oldcentroids])

