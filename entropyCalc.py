import math
import numpy as np


def calculateEntrophy(cluster):
    entrophy = 0
    totalInstances = len(cluster)
    values = []
    # get the number of unique values
    for x in cluster:
        if x not in values:
            values.append(x)


    for value in values:
        count = 0
        # count value in the cluster
        for x in cluster:
            if x == value:
                count += 1
        probability = count / totalInstances
        entrophy += (probability * math.log2(probability))
    return -1 * entrophy


def calculateMeanEntrophy():
    pass
def main():
    k = input('how many clusters do you need calculated? ')
    x = 0
    while x is not k:
        clusters = input('provide your list of clusters: ')
        print('calculating entrophy per value in cluster')
        print(calculateEntrophy(clusters))
        x += 1


main()