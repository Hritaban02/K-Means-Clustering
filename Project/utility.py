# Import the necessary libraries
import numpy as np
from models import KMeans


def wangs_method_of_cross_validation(X, n_clusters=8, max_iter=300, distance_metric='euclidean', algorithm='lloyd', c=20):
    """
    The wangs_method_of_cross_validation function takes in six input parameters X, n_clusters, max_iter, distance_metric, algorithm, and c
    and returns the average number of disagreements between the two clustering in wangs_method_of_cross_validation.

    Inputs:
        'X': X is n x d numpy array where n is the number of data points and m is the number of features.
            It is the feature matrix.

        'n_clusters': It is the number of clusters into which the data points must be segregated.
                The default value of this parameter is 8 which is arbitrary.

        'max_iter': It is the maximum number of iterations up to which the K-Means Clustering Model
            can improve the positions of the cluster centres and thereby improving the clustering.
            The default value of this parameter is 300 which is arbitrary.

        'distance_metric': It is the metric which is used to measure the distance between the data points.
            The default value of this parameter is 'euclidean'.

        'algorithm': It is the initialization algorithm by which the cluster centres are to be initialized.
            Permissible methods of initialization are 'lloyd' and 'kmeans++'.
            The default value of this parameter is 'kmeans++'.

        'c': It is the number of permutations for which the dataset X must be shuffled.
            The default value of the parameter is 20.

        NOTE:   distance_metric other than 'euclidean' is not implemented in this algorithm,
                this parameter is a design choice and has been mentioned for extension purposes only.
                For all purposes in this algorithm distance will be measured as 'euclidean'.

    Outputs:
        'average_disagreements': It is the average number of disagreements between the two clustering in wangs_method_of_cross_validation.
    """
    # Initialize the total number of disagreements to 0
    summation_disagreements = 0

    # Perform the computation of disagreements for c number of times, each time shuffling the dataset X
    for i in range(0, c):
        # Get the new shuffled data
        permuted_data = np.random.permutation(X)

        # Split the data into three nearly equal parts
        [S1, S2, S3] = np.array_split(permuted_data, 3)

        # Perform k-means clustering on S1 based on the parameters provided
        kmeans_1 = KMeans(n_clusters, max_iter, distance_metric, algorithm)
        kmeans_1.fit(S1)

        # Use the first model to predict the labels of S3
        y_predict_1 = kmeans_1.predict(S3)

        # Perform k-means clustering on S2 based on the parameters provided
        kmeans_2 = KMeans(n_clusters, max_iter, distance_metric, algorithm)
        kmeans_2.fit(S2)

        # Use the second model to predict the labels of S3
        y_predict_2 = kmeans_2.predict(S3)

        # Get the indices of the numpy array S3
        indices = np.array(range(0, len(S3)))

        # Compute the pairs of indices and store them in a array
        pairs_of_indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2)

        # For each pair of indices, increment the number of total disagreements if the conditions are satisfied
        for j in range(0, len(pairs_of_indices)):
            # If the indices are not equal (There is no point in checking a single data point)
            if pairs_of_indices[j][0] != pairs_of_indices[j][1]:
                # If model 1 says that the pair of data points belong to the same cluster while model two says they do not, then there is a disagreement
                if y_predict_1[pairs_of_indices[j][0]] == y_predict_1[pairs_of_indices[j][1]] and y_predict_2[pairs_of_indices[j][0]] != y_predict_2[pairs_of_indices[j][1]]:
                    summation_disagreements += 1
                # Else if model 1 says that the pair of data points do not belong to the same cluster while model two says they do , then there is a disagreement
                elif y_predict_1[pairs_of_indices[j][0]] != y_predict_1[pairs_of_indices[j][1]] and y_predict_2[pairs_of_indices[j][0]] == y_predict_2[pairs_of_indices[j][1]]:
                    summation_disagreements += 1

    # Compute the average number of disagreements
    average_disagreements = summation_disagreements/(2*c)

    # Return the average number of disagreements
    return average_disagreements
