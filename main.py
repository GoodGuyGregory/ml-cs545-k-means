import numpy as np
import matplotlib.pyplot as plt
import random


class KMeansClassifier:
    def __init__(self, k, iterations, dataSet):
        # number of clusters
        self.K = k
        self.iterations = iterations
        self.dataSet = dataSet

        # stores mean values of the centroids
        self.centroids = np.array([])
        # dataframe that holds the centroid minArgResults from kClassifyCall
        self.centroidPoints = None

    def randomizeKCentroids(self):
        # randomly pick K centroids from the data values
        temp = list(range(self.dataSet.shape[0]))
        idx = random.sample(temp, self.K)
        self.centroids = self.dataSet[idx]

    def plotData(self, labels):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.dataSet[:, 0], self.dataSet[:, 1], c=labels[:len(self.dataSet)], cmap='viridis', marker='.')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c="black", marker='+', label="Centroids")
        plt.show()

    def kMeansClassify(self):
        # determine distance from the closest cluster

        for _ in range(self.iterations):
            clusterClasses = []
            # calculate the distance metric per point
            # E step assign random points to the centroids
            for point in self.dataSet:
                # calculate the euclidean distances for each point and the centroid.
                closestValueDistance = np.argmin(np.sqrt(np.sum((point - self.centroids)**2, axis=1)))
                # add the smallest value index from the list of centroids
                clusterClasses.append(closestValueDistance)


            # M step - calculate the mean and adjust the centroid distances
            clusterAssignments = np.array(clusterClasses)
            # print("cluster Assignments")
            # print(clusterAssignments.shape)
            # print(clusterAssignments)

            clusterValues = []

            for i in range(self.K):
                # append where the index of the points match the kth cluster
                # this will pull the values for later adjustment of the centers for each point
                clusterValues.append(np.argwhere(clusterAssignments == i))
            # list to hold the new adjusted centers
            clusterCenters = []
            # iterate for each of the cluster we need to pull the specific data points
            # to recalculate the new center values
            for i,item in enumerate(clusterValues):
                print("cluster values: " + str(i))
                print(item.shape)
                print(item)
            self.plotData(clusterClasses)
            for i, values in enumerate(clusterValues):
                # case where there are no values assigned to this cluster.
                if len(values) == 0:
                    # no need to change the values
                    clusterCenters.append(self.centroids[i])
                else:
                    # calculate the averages for all points from the cluster classes assigned values
                    # print(self.dataSet.shape)
                    # print(values)
                    clusterCenters.append(np.mean(self.dataSet[values], axis=0)[0])
                    # print(np.mean(self.dataSet[values],axis=0).shape)
                # if the whole collection of centroids is not changed or has a slightly small change
                # re-assign the centroids for new round of E step
            self.centroids = np.array(clusterCenters)
        # gather and return the cluster labels
        return clusterClasses

class FuzzyCMeansClassifier:
    def __init__(self, k, m, iterations, dataSet):
        self.K = k
        self.M = m
        self.iterations = iterations
        self.dataSet = dataSet
        self.centroids = np.array([])
        self.Weights = None
        self.centroidsPoints = None


    def plotData(self, labels):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.dataSet[:, 0], self.dataSet[:, 1], c=labels[:len(self.dataSet)], cmap='viridis', marker='.')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c="black", marker='+', label="Centroids")
        plt.show()


    def randomizeWeights(self):
        self.Weights = np.random.uniform(0.1,1, size=(self.dataSet.shape[0], self.K))

    def fuzzyCMeansCluster(self):
        self.randomizeWeights()

        # distance between each data point and the centroids
        for iteration in range(self.iterations):

            # calculate the cluster membership
            # cK is done first.
            clusterAssignment = []


            for i in range(self.K):
                clusterAssignment.append(np.sum(self.Weights[:, i].reshape(-1, 1) * self.dataSet ** self.M * self.dataSet, axis=0) \
                                        / np.sum(self.Weights[:, i].reshape(-1, 1) * self.dataSet ** self.M, axis=0))

            self.centroids = np.array(clusterAssignment)

            clusterClasses = []
            # calculate the distance metric per point
            # E step assign random points to the centroids
            for point in self.dataSet:
                # calculate the euclidean distances for each point and the centroid.
                closestValueDistance = np.argmin(np.sqrt(np.sum((point - self.centroids) ** 2, axis=1)))
                # add the smallest value index from the list of centroids
                clusterClasses.append(closestValueDistance)
            self.plotData(clusterClasses)


            # initialize an empty np array
            distancesPerCluster = np.zeros((self.dataSet.shape[0], self.K))
            for i in range(self.K):
                distancesPerCluster[:, i] = np.linalg.norm(self.dataSet - self.centroids[i, :], axis=1)

            # reset the weight ij
            weightAssignmentPerCluster = []
            for i in range(self.K):
                weightAssignmentPerCluster.append(1 / np.sum((distancesPerCluster[:, i].reshape(-1, 1) / distancesPerCluster)**(2/(self.M - 1)), axis=1))
            self.Weights = np.array(weightAssignmentPerCluster)
            # transpose to maintain additional iteration
            self.Weights = np.transpose(self.Weights, (1,0))



def readData():
    clusterData = np.loadtxt("545_cluster_dataset programming 3.txt")
    return clusterData

def main():
    clusterPoints = readData()
    # kmeans = KMeansClassifier(5,10, clusterPoints)
    # kmeans.randomizeKCentroids()
    # labeledClusters = kmeans.kMeansClassify()

    cmeans = FuzzyCMeansClassifier(5,2,100, clusterPoints)
    cmeans.fuzzyCMeansCluster()

main()