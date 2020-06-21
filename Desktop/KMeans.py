import numpy as np
import pandas as pd
from sklearn import datasets


def distancefunction(centroid, element, k):
    "Calculates the euclidean distance between centroid and an element"
    distance = 0
    for i in range(k):
        distance += (element[i] - centroid[i]) * (element[i] - centroid[i])
    return distance


iris = datasets.load_iris()
x = iris.data
dataset = pd.DataFrame(x)  # Train dataset
pd.set_option('display.max_rows', dataset.shape[0] + 1)
n = 5  # Number of clusters
k = 4  # Dimension
centroid = np.random.rand(n, k)
for i in range(k):
    centroid[:, i] *= dataset[i].mean() * 2
# print("Initial centroid:")
# print(centroid)
dataset['Cluster'] = np.zeros((150,), dtype=int)

mindf = 0
while True:
    olddataset = dataset.copy()
    for i in range(dataset.shape[0]):
        mindf = distancefunction(centroid[0], dataset.loc[i, :], k)
        cluster = 0
        for j in range(1, n):
            newdf = distancefunction(centroid[j], dataset.loc[i, :], k)
            if mindf > newdf:
                mindf = newdf
                cluster = j
        dataset.loc[i, 'Cluster'] = cluster
    # print("Next dataset is:")
    # print(dataset)
    for i in range(n):
        clusters = dataset.loc[dataset.Cluster == i]
        if clusters.empty:
            for j in range(k):
                centroid[i][j] = dataset.loc[:, j].mean() * 2 * np.random.rand()
        else:
            for j in range(k):
                centroid[i][j] = clusters.loc[:, j].mean()
    # print("Next centroid:")
    # print(centroid)
    if dataset.equals(olddataset):
        break
print("Final dataset is:")
print(dataset)
print("\n\n")
print("Final centroid is:")
print(centroid)


