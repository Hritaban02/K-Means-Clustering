# Import the necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from models import KMeans


def improved_test_A(n_clusters, X, y, max_iter=50, distance_metric="euclidean"):
    """
    The improved_test_A function takes in five input parameters - n_clusters, X, y, max_iter and distance_metric
    and performs the Improved_Test_A which uses the kmeans++ heuristic.

    Inputs:
        'n_clusters': It is the number of clusters into which the data points must be segregated.

        'X': X is n x d numpy array where n is the number of data points and m is the number of features.
                It is the feature matrix.

        'y': y is the ground truth labels of the data points.

        'max_iter': It is the maximum number of iterations up to which the Test must be performed.
            The default value of this parameter is 50 which is arbitrary.

        'distance_metric': It is the metric which is used to measure the distance between the data points.

        NOTE:   distance_metric other than 'euclidean' is not implemented in this algorithm,
            this parameter is a design choice and has been mentioned for extension purposes only.
            For all purposes in this algorithm distance will be measured as 'euclidean'.
    """
    # Select the first cluster centre randomly

    # Grab a single index randomly from the numpy array X
    first_random_choice_for_centre = np.random.choice(X.shape[0], 1, replace=False)
    # Initialise current_cluster_centres with the cluster centre selected randomly
    current_cluster_centres = X[first_random_choice_for_centre]

    # Iteratively keep adding a new cluster centre according to the probability distribution defined by the kmeans++ heuristic
    for i in range(2, n_clusters + 1):

        # Initialize the Sum of Squared of Minimum of the distances from the already chosen clusters to 0
        sum_of_square_of_min_distances = 0

        # This list stores the probability of the data point being a cluster centre
        probability_of_x_being_a_centre = []

        # For every x in the X compute the probability of it being the next cluster centre and add it to the list, until all the cluster centres are obtained
        for x in X:
            # Get the squared of the distance from the closest cluster centre
            _, distance_squared = KMeans.closest_centre(x, current_cluster_centres, distance_metric)

            # Append it to the list
            probability_of_x_being_a_centre.append(distance_squared)

            # Update the sum
            sum_of_square_of_min_distances += distance_squared

        # In order to compute the probability convert the list to an numpy array and then divide each element of the array by the sum
        probability_of_x_being_a_centre = np.array(probability_of_x_being_a_centre, dtype=float)
        probability_of_x_being_a_centre = np.true_divide(probability_of_x_being_a_centre,
                                                         sum_of_square_of_min_distances)

        # Add the x which has the highest probability of being the next cluster centre to the list
        current_cluster_centres = np.append(current_cluster_centres,
                                            [X[np.where(probability_of_x_being_a_centre == np.amax(
                                                probability_of_x_being_a_centre))[0][0]]], axis=0)

    # Store the initial cluster centres chosen from the data points according to the kmeans++ heuristic
    k_means_plus_plus_points = current_cluster_centres

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
        kmeans = KMeans(n_clusters=n_clusters, distance_metric=distance_metric,
                        initial_cluster_centres=k_means_plus_plus_points)
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
