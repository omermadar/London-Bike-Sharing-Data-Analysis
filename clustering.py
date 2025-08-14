import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as a numpy array of shape (n, 2)
    :return: data and noise, where noise~N(0,0.00001^2)
    """
    noise = np.random.normal(loc=0, scale=1e-5, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as a numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]

def min_max_scaling(data):
    """
    This function performs min-max scaling on the NumpyArr.
    :param data: dataset as a numpy array of shape (n, 2)
    :return: numpy array scaled between [0,1]
    """
    min_vals = data.min(axis=0)  # min per column
    max_vals = data.max(axis=0)  # max per column
    scaled = (data - min_vals) / (max_vals - min_vals)
    return scaled

# ====================
def transform_data(df, features):
    """
    performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as a numpy array of shape (n, 2)
    """
    data = df[features].to_numpy()
    data = min_max_scaling(data)
    add_noise(data)

    return data


def find_closest_centroid(vector, centroids):
    """
    Finds the index of the closest centroid to the given vector.
    :param vector: 1D np.array of shape (d)
    :param centroids: 2D np.array of shape (k, d)
    :return: int index of the closest centroid
    """
    # Initialize with the distance to the first centroid
    min_dist = dist(vector, centroids[0])
    closest_idx = 0

    # Loop through all centroids by index
    for idx in range(1, len(centroids)):
        d = dist(vector, centroids[idx])
        if d < min_dist:
            min_dist = d
            closest_idx = idx

    return closest_idx


def kmeans(data, k):
    """
    running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of clusters
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    #data = transform_data(data, ['cnt', 't1']) # Find out what features we need
    prev_centroids = choose_initial_centroids(data, k)
    labels = assign_to_clusters(data, prev_centroids)
    centroids = recompute_centroids(data, labels, k)
    while not np.array_equal(centroids, prev_centroids):
        prev_centroids = centroids
        labels = assign_to_clusters(data, centroids)
        centroids = recompute_centroids(data, labels, k)

    return labels, centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model and saving the figure.
    :param data: data as a numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as a numpy array of size n
    :param centroids: the final centroids of kmeans, as a numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """

    ind1, ind2 = 0,1

    plt.figure(figsize=(8, 8))
    # plot the points
    plt.scatter(data[:, ind1], data[:, ind2], c = labels.astype(int))
    #plot the centroid
    for i in range(len(centroids)):
        plt.scatter(centroids[i, ind1], centroids[i, ind2], color='white', edgecolors='black', marker='*', linewidth=2,
                    s=200, alpha=0.85, label=f'Centroid' if i == 0 else None)

    plt.xlabel('cnt')
    plt.ylabel('t1')
    plt.title(f'Results for kmeans with k = {len(centroids)}')
    plt.show()

    print(np.array_str(centroids, precision=3, suppress_small=True))
    plt.savefig(path)
    plt.close('all')


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the Euclidean distance
    """
    pass
    return np.sqrt(np.sum((x - y) ** 2))


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as a numpy array of shape (n, 2)
    :param centroids: current centroids as a numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    length = data.shape[0]
    labels = np.zeros(length)
    for i in range(length):
        closest_centroid = find_closest_centroid(data[i], centroids)
        labels[i] = closest_centroid
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as a numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        new_centroids[i] = np.mean(data[labels == i], axis=0)

    return new_centroids


