import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Read the csv file and return a list of dictionaries with the column headers as keys and the row elements as values.
def load_data(filepath):
    data_list = []  

    # open the file and read
    with open('countries.csv', newline = '') as file:
        reader = csv.DictReader(file)
        # the loop traverses each row and append each row to data_list
        for row in reader:  
            data_list.append(dict(row)) 
            
    return data_list


# Extract the statisics data from dictionary and covert them to float
def calc_features(row):
    
    # extract the value of specific columns from dictionary and converts their data type to float
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    
    # combine these float values into a numpy array and make the shape of this array (6,)
    feature_array = np.array([x1, x2, x3, x4, x5, x6], dtype = np.float64)
    feature_array = np.reshape(feature_array,(6,))
    
    return feature_array


def hac(features):
    
    # initialize the number of input features 
    # and matrix that stores distances between pair of feature vectors
    n = len(features) 
    distance_matrix = np.zeros((n, n))
    
    # the nested loop computes the Euclidean distance between each pair of feature vectors
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
    
    # make a dictionary whose keys are from 0 to n-1         
    clusters = {}  
    for i in range(n):
        clusters[i] = [i]
    
    # a linkage matrix to store the result of the complete linkage clustering
    Z = np.zeros((n - 1, 4))
    
    # a unique identifier to ensure new cluster after each merge
    cluster_index = n
    
    # the outer loop to merge clusters
    for loop in range(n - 1):
        
        # initialize minimum distance and merge to cluster
        minimum_distance = float('inf')
        mergeCluster = (-1, -1)
        
        # the double to compare all pairs of cluster
        for i in clusters.keys():
            for j in clusters.keys():
                if i < j:
                    # compute the distance between two clusters
                    cluster_distance = max([distance_matrix[x][y] for x in clusters[i] for y in clusters[j]])
                    # check and update the minimum distance and merge cluster
                    if cluster_distance < minimum_distance:
                        minimum_distance = cluster_distance
                        mergeCluster = (i, j)
        
        # store the information of the cluster in a linkage matrix
        (clusterX, clusterY) = mergeCluster
        Z[loop, 0] = clusterX
        Z[loop, 1] = clusterY
        Z[loop, 2] = minimum_distance
        Z[loop, 3] = len(clusters[clusterX]) + len(clusters[clusterY])
        
        # merge clusters together and delete the previous clusters since they are now merged
        clusters[cluster_index] = clusters[clusterX] + clusters[clusterY]
        del clusters[clusterX]
        del clusters[clusterY]
        
        cluster_index += 1
        
    return Z

# visualize the result of complete linkage clustering    
def fig_hac(Z, names):
    fig = plt.figure()
    
    # creates a dendrogram based on the given linkage matrix
    dendrogram(Z, labels = names, leaf_rotation = 90)

    plt.tight_layout()  
    plt.show()
    
    return fig

    
# normalize the features on the given dataset
def normalize_features(features):
    features_array = np.array(features)
    
    # compute the mean and standard deviation for each feature
    feature_mean = np.mean(features_array, axis = 0)
    feature_sd = np.std(features_array, axis = 0)
    
    # perform the normalization on the features 
    normalized_features = ((features_array - feature_mean) / feature_sd)
    normalized_list = [row for row in normalized_features]
    
    return normalized_list

if __name__ == "__main__":
    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)

    n = 10
    print(f"Testing for n = {n}")
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    
    fig_hac(Z_raw, country_names[:n])
    fig_hac(Z_normalized, country_names[:n]) 