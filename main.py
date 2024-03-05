import pandas as pd
import numpy as np


class KMeansClassifier:
    def __init__(self, k, iterations, dataSet):
        # number of clusters
        self.K = k
        self.iterations = iterations
        self.dataSet = np.asarray(dataSet, dtype=float)

        # stores mean values of the centroids
        self.centroids = []
        # dataframe that holds the centroid minArgResults from kClassifyCall
        self.centroidPoints = None

    def randomizeKCentroids(self):
        # randomly pick K centroids from the data values
        self.centroids = np.random.uniform(np.amin(self.dataSet), np.amax(self.dataSet), size=self.K)
        print(self.centroids)



    def kMeansClassify(self):
        # determine distance from the closest cluster
        pass



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
    kmeans = KMeansClassifier(5,10, clusterPoints)
    kmeans.randomizeKCentroids()


main()