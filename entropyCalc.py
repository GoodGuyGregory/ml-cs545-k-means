import math

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


def main():
    k = int(input('how many clusters do you need calculated? '))
    meanClusters = []
    entropyPerCluster = []
    for x in range(k):
        clusters = input('provide your list of clusters: ')
        meanClusters.append(clusters)
        print('calculating entrophy per value in cluster')
        clusterEntropy = round(calculateEntrophy(clusters),3)
        print(clusterEntropy)
        entropyPerCluster.append(clusterEntropy)


main()