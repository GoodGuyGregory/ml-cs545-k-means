import pandas as pd
import numpy as np

def readData():
    clusterData = pd.read_csv("545_cluster_dataset programming 3.txt", sep='\t', header=None)
    clusterData[['Feature 1', 'Feature 2']] = clusterData[0].str.split(expand=True, n=1)
    clusterData.drop(0, axis=1, inplace=True)
    return clusterData

def main():
    clusterPoints = readData()
    print(clusterPoints)


main()