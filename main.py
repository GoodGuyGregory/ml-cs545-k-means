import numpy as np
import matplotlib.pyplot as plt


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
        self.centroids = np.random.uniform(np.amin(self.dataSet, axis=0), np.amax(self.dataSet, axis=0), size=(self.K, self.dataSet.shape[1]))
        # print(self.centroids)

    def plotData(self, labels):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.dataSet[:, 0], self.dataSet[:, 1], c=labels[:len(self.dataSet)], cmap='viridis', marker='*')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c="black", marker='+', label="Centroids")
        plt.show()

    def kMeansClassify(self):
        # determine distance from the closest cluster
        clusterClasses = []

        for _ in range(self.iterations):
            # calculate the distance metric per point
            # E step assign random points to the centroids
            for point in self.dataSet:
                # calculate the euclidean distances for each point and the centroid.
                closestValueDistance = np.argmin(np.sqrt(np.sum((point - self.centroids)**2, axis=1)))
                # add the smallest value index from the list of centroids
                clusterClasses.append(closestValueDistance)

            # M step - calculate the mean and adjust the centroid distances
            clusterAssignments = np.array(clusterClasses)

            clusterValues = []

            for i in range(self.K):
                # append where the index of the points match the kth cluster
                # this will pull the values for later adjustment of the centers for each point
                clusterValues.append(np.argwhere(clusterAssignments == i))
            # list to hold the new adjusted centers
            clusterCenters = []
            # iterate for each of the cluster we need to pull the specific data points
            # to recalculate the new center values
            self.plotData(clusterClasses)
            for i, values in enumerate(clusterValues):
                # case where there are no values assigned to this cluster.

                if len(values) == 0:
                    # no need to change the values
                    clusterCenters.append(self.centroids[i])
                else:
                    # calculate the averages for all points from the cluster classes assigned values
                    clusterCenters.append(np.mean(self.dataSet[clusterAssignments[values]], axis=0)[0])
                # if the whole collection of centroids is not changed or has a slightly small change
                if np.max(self.centroids - np.array(clusterCenters[i])) < 0:
                    # done moving.
                    break
                else:
                    # re-assign the centroids for new round of E step
                    self.centroids = np.array(clusterCenters)
        # gather and return the cluster labels
        return clusterClasses

    def fuzzyCmeansClassify(self):
        return None




def readData():
    clusterData = np.loadtxt("545_cluster_dataset programming 3.txt")
    return clusterData

def main():
    clusterPoints = readData()
    kmeans = KMeansClassifier(8,10, clusterPoints)
    kmeans.randomizeKCentroids()
    labeledClusters = kmeans.kMeansClassify()


main()