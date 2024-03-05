import pandas as pd
import numpy as np


class KMeansClassifier:
    def __init__(self, k, iterations, dataSet):
        # number of clusters
        self.K = k
        self.iterations = iterations
        self.dataSet = dataSet

        # stores mean values of the centroids
        self.centroids = []
        # dataframe that holds the centroid minArgResults from kClassifyCall
        self.centroidPoints = None

    def randomizeKCentroids(self):
        # randomly pick K centroids from the data values
        self.centroids = np.random.uniform(np.amin(self.dataSet), np.amax(self.dataSet), size=(self.K, self.dataSet.shape[1]))
        print(self.centroids)


    def kMeansClassify(self):
        # determine distance from the closest cluster
        clusterClasses = []

        # calculate the distance metric per point
        for point in self.dataSet:
            closestValue = np.argmin(np.sqrt(np.sum((point - self.centroids)**2, axis=1)))
            closetCluster = closestValue
            clusterClasses.append(closetCluster)





    def fuzzyCmeansClassify(self):
        return None




def readData():
    clusterData = np.loadtxt("545_cluster_dataset programming 3.txt")
    return clusterData

def main():
    clusterPoints = readData()
    kmeans = KMeansClassifier(5,10, clusterPoints)
    kmeans.randomizeKCentroids()
    kmeans.kMeansClassify()


main()