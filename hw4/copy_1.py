# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import dendrogram


# # Read the csv file and return a list of dictionaries with the column headers as keys and the row elements as values.
# def load_data(filepath):
#     data_list = []  

#     with open('countries.csv', newline='') as file:
        
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
            
#             data_list.append(dict(row)) 
            
#     return data_list


# # Extract the statisics data from dictionary and covert them to float
# def calc_features(row):
    
#     x1 = float(row['Population'])
#     x2 = float(row['Net migration'])
#     x3 = float(row['GDP ($ per capita)'])
#     x4 = float(row['Literacy (%)'])
#     x5 = float(row['Phones (per 1000)'])
#     x6 = float(row['Infant mortality (per 1000 births)'])
    
#     feature_array = np.array([x1, x2, x3, x4, x5, x6], dtype = np.float64)
#     feature_array = np.reshape(feature_array,(6,))
    
#     return feature_array


# def hac(features):
#     # Number of feature vectors
#     n = len(features)
    
#     # Initialize distance matrix with euclidean distances
#     distance_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
                
#     # Initialize clusters
#     clusters = [{i} for i in range(n)]
    
#     # Initialize Z with zeros
#     Z = np.zeros((n-1, 4))
    
#     for iteration in range(n-1):
#         # Find the pair of clusters with the minimum distance
#         min_distance = float('inf')
#         clusters_to_merge = (-1, -1)
        
#         for i in range(len(clusters)):
#             for j in range(i+1, len(clusters)):
#                 cluster_distance = max([distance_matrix[a][b] for a in clusters[i] for b in clusters[j]])
                
#                 if cluster_distance < min_distance:
#                     min_distance = cluster_distance
#                     clusters_to_merge = (i, j)
        
#         # Fill the Z matrix
#         cluster_a, cluster_b = clusters_to_merge
#         Z[iteration, 0], Z[iteration, 1] = sorted([min(clusters[cluster_a]), min(clusters[cluster_b])])
#         Z[iteration, 2] = min_distance
#         Z[iteration, 3] = len(clusters[cluster_a]) + len(clusters[cluster_b])
        
#         # Merge the clusters
#         clusters[cluster_a] = clusters[cluster_a].union(clusters[cluster_b])
#         del clusters[cluster_b]
        
#     return Z
    
# def fig_hac(Z, names):
#     fig = plt.figure()
#     dendrogram(Z, labels=names, leaf_rotation=90)

#     plt.xlabel("Countries")
#     plt.ylabel("Distance")
#     plt.title("Hierarchical Agglomerative Clustering")
#     plt.tight_layout()
    
#     return fig
    

# def normalize_features(features):
#     features_array = np.array(features)
    
#     column_means = np.mean(features_array, axis=0)
#     column_stddevs = np.std(features_array, axis=0)
    
#     normalized_features = (features_array - column_means) / column_stddevs
    
#     normalized_features_list = [row for row in normalized_features]
    
#     return normalized_features_list