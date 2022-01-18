# Import the necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from models import KMeans


def test_A(n_clusters, X, y, max_iter=50):
    """
    The test_A function takes in four input parameters - n_clusters, X, y and max_iter and performs the Test_A as specified.

    Inputs:
        'n_clusters': It is the number of clusters into which the data points must be segregated.

        'X': X is n x d numpy array where n is the number of data points and m is the number of features.
                It is the feature matrix.

        'y': y is the ground truth labels of the data points.

        'max_iter': It is the maximum number of iterations up to which the Test must be performed.
            The default value of this parameter is 50 which is arbitrary.
    """
    # Select K random points from X to act as the initial centres
    index = np.random.choice(X.shape[0], n_clusters, replace=False)
    k_random_points = X[index]

    # Initialize the average value of all metrics to be 0
    homogeneity_score_avg = 0
    adjusted_rand_score_avg = 0
    normalized_mutual_info_score_avg = 0
    fowlkes_mallows_score_avg = 0

    # For each iteration of the test, split the data randomly into a 80:20 ratio
    # Train the model 80% of the dataset
    # Then, get the predicted labels on the rest of the 20% of the dataset - X_test
    # Measure all the metrics using these predicted labels and y_test as the ground truth
    for i in range(0, max_iter):
        # Split the data randomly into a 80:20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the model 80% of the dataset with the k_random_points as initial centres
        kmeans = KMeans(n_clusters=n_clusters, initial_cluster_centres=k_random_points)
        kmeans.fit(X_train)

        # Get the predicted labels on the rest of the 20% of the dataset - X_test
        y_predict = kmeans.predict(X_test)

        # Measure all the metrics using these predicted labels y_predict and y_test as the ground truth
        homogeneity_score_avg += homogeneity_score(y_test, y_predict)
        adjusted_rand_score_avg += adjusted_rand_score(y_test, y_predict)
        normalized_mutual_info_score_avg += normalized_mutual_info_score(y_test, y_predict)
        fowlkes_mallows_score_avg += fowlkes_mallows_score(y_test, y_predict)

    # Compute the average of the metric over the iterations
    homogeneity_score_avg /= max_iter
    adjusted_rand_score_avg /= max_iter
    normalized_mutual_info_score_avg /= max_iter
    fowlkes_mallows_score_avg /= max_iter

    # Return the average value of the metrics
    return homogeneity_score_avg, adjusted_rand_score_avg, normalized_mutual_info_score_avg, fowlkes_mallows_score_avg
