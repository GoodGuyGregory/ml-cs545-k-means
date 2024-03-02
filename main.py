import pandas as pd
import numpy as np


class KMeansClassifier:
    def __int__(self, k, iterations, dataSet):
        # number of clusters
        self.K = k
        self.iterations = iterations
        self.dataSet = dataSet

        # stores mean values of the centroids
        self.centroids = []
        # dataframe that holds the centroid minArgResults from kClassifyCall
        self.centroidPoints =

    def randomizeKCentroids(self):
        # randomly pick K centroids from the data values
        for k in self.K:
            # choose a random sample point
            randomCentroid = self.dataSet.sample(frac=1)
            self.centroids.append(randomCentroid)


    def kMeansClassify(self):
        # determine distance from the closest cluster
        pass;



    def fuzzyCmeansClassify(self):
        return None




def readData():
    clusterData = pd.read_csv("545_cluster_dataset programming 3.txt", sep='\t', header=None)
    clusterData[['Feature 1', 'Feature 2']] = clusterData[0].str.split(expand=True, n=1)
    clusterData.drop(0, axis=1, inplace=True)
    return clusterData

def main():
    clusterPoints = readData()
    print(clusterPoints)


main()