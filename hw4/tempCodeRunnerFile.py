if __name__ == '__main__':
    # Load data from 'countries.csv' using your 'load_data' function
    country_data = load_data('countries.csv')

    # Extract features for each country and store them in a list
    features_list = [calc_features(country) for country in country_data]

    # Convert the list of features to a NumPy array
    features_array = np.array(features_list)

    # Call the 'hac' function with the features
    result = hac(features_array)

    # Print the result
    print("Hierarchical Agglomerative Clustering Result:")
    print(result)