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

    centroids_transposed = np.array(centroids).T
    x_centroids = centroids_transposed[0]
    y_centroids = centroids_transposed[1]

    for i in range(clusters_no):
        cluster = np.array(clusters[i]).transpose()
        x_cluster = cluster[0]
        y_cluster = cluster[1]

        # setting a different color for each cluster
        plt.scatter(x_cluster, y_cluster, c = np.random.rand(1, 3))

    plt.scatter(x_centroids, y_centroids, c = 'r')
    plt.show()

def plot_clusters_1d(centroids, clusters):
    # adapting the above function to work with one dimensional data
    clusters_no = len(centroids)
    x_centroids = np.array(centroids)
    y_centroids = np.zeros(clusters_no)

    for i in range(clusters_no):
        x_cluster = np.array(clusters[i])
        y_cluster = np.zeros(len(clusters[i]))

        # setting a different color for each cluster
        plt.scatter(x_cluster, y_cluster, c = np.random.rand(1, 3))

    plt.scatter(x_centroids, y_centroids, c = 'r')
    plt.show()


def convergence(centroids, oldcentroids):
    return np.linalg.norm(centroids - oldcentroids) < 0.01