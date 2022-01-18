# Import the necessary libraries
import numpy as np
import copy


# KMeans Class
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, distance_metric='euclidean', algorithm='lloyd', initial_cluster_centres=None):
        """
        The __init__ method of the KMeans class takes in five input parameters - n_clusters, max_iter, distance_metric,
        algorithm, and initial_cluster_centres and sets the parameters of the K-Means Clustering model.

        Inputs:
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

            'initial_cluster_centres': It is the set of initial cluster centres that can be provided by the user.
                It overrides the algorithm parameter. The default value of this parameter is None.

            NOTE:   distance_metric other than 'euclidean' is not implemented in this algorithm,
                this parameter is a design choice and has been mentioned for extension purposes only.
                For all purposes in this algorithm distance will be measured as 'euclidean'.

            NOTE:   If number of clusters in initial_cluster_centres is not equal to n_clusters,
                then, this parameter is ignored and set to None.

        Attributes:
            'n_clusters': It stores the number of clusters into which the data points must be segregated.

            'max_iter': It stores the maximum number of iterations up to which the K-Means Clustering Model
                can improve the positions of the cluster centres and thereby improving the clustering.

            'distance_metric': It stores the metric which is used to measure the distance between the data points.

            NOTE:   distance_metric other than 'euclidean' is not implemented in this algorithm,
                this parameter is a design choice and has been mentioned for extension purposes only.
                For all purposes in this algorithm distance will be measured as 'euclidean'.

            'algorithm': It stores the initialization algorithm by which the cluster centres are to be initialized.
                Permissible methods of initialization are 'lloyd' and 'kmeans++'.

            'initial_cluster_centres': It is the set of initial cluster centres that can be provided by the user.
                It overrides the algorithm parameter.

            NOTE:   If number of clusters in initial_cluster_centres is not equal to n_clusters,
                then, this parameter is ignored and set to None.

            'cluster_centres_': It stores the final centres of the clusters after iteratively improving the clustering.
                It is initialized to None.

            'labels_': It stores the final list of labels (cluster indices) of the data points after iteratively improving the clustering.
                It is initialized to None.

            'inertia_': It stores the sum of square of distances of each data point from its respective cluster centre.
                It is initialized to None.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        # If the value of algorithm is not permissible then output a warning and set it to a permissible value.
        if algorithm != "lloyd" and algorithm != "kmeans++":
            print("\n Warning: Permissible methods of initialization are 'lloyd' and 'kmeans++'. algorithm value ignored and set to \'kmeans++\'.")
            algorithm = "kmeans++"
        self.algorithm = algorithm

        # If the number of initial_cluster_centres is not valid then output a warning and set it to None.
        if initial_cluster_centres is not None:
            if len(initial_cluster_centres) != self.n_clusters:
                print("\n Warning: Number of centres in initial_cluster_centres is not equal to n_clusters. initial_cluster_centres ignored and set to None.")
                initial_cluster_centres = None
        self.initial_cluster_centres_ = initial_cluster_centres
        self.cluster_centres_ = None
        self.labels_ = None

        # If the value of distance_metric is not permissible then output a warning and set it to a permissible value.
        if distance_metric != "euclidean":
            print("\n Warning: Permissible metrics are 'euclidean' only. distance_metric value ignored and set to \'euclidean\'.")
            distance_metric = "euclidean"
        self.distance_metric = distance_metric
        self.inertia_ = None

    def lloyd_cluster_centre_initialization(self, X):
        """
        The lloyd_cluster_centre_initialization method of the KMeans class takes in one input parameter X
        and returns the initial cluster centres chosen randomly from the data points.

        Inputs:
            'X': X is n x d numpy array where n is the number of data points and m is the number of features.
                It is the feature matrix.

        Outputs:
            'cluster_centres': Numpy array of initial cluster centres chosen randomly from the data points.
        """
        # Grab random indices from the numpy array X, n_clusters being the number of indices
        index = np.random.choice(X.shape[0], self.n_clusters, replace=False)

        # Return numpy array of initial cluster centres chosen randomly from the data points.
        return X[index]

    def kmeans_plus_plus_cluster_centre_initialization(self, X):
        """
        The kmeans_plus_plus_cluster_centre_initialization method of the KMeans class takes in one input parameter X
        and returns the initial cluster centres chosen from the data points according to the kmeans++ heuristic.

        Inputs:
            'X': X is n x d numpy array where n is the number of data points and d is the number of features.
                It is the feature matrix.

        Outputs:
            'cluster_centres': Numpy array of initial cluster centres chosen from the data points according to the kmeans++ heuristic.
        """
        # Select the first cluster centre randomly

        # Grab a single index randomly from the numpy array X
        first_random_choice_for_centre = np.random.choice(X.shape[0], 1, replace=False)
        # Initialise current_cluster_centres with the cluster centre selected randomly
        current_cluster_centres = X[first_random_choice_for_centre]

        # Iteratively keep adding a new cluster centre according to the probability distribution defined by the kmeans++ heuristic
        for i in range(2, self.n_clusters + 1):

            # Initialize the Sum of Squared of Minimum of the distances from the already chosen clusters to 0
            sum_of_square_of_min_distances = 0

            # This list stores the probability of the data point being a cluster centre
            probability_of_x_being_a_centre = []

            # For every x in the X compute the probability of it being the next cluster centre and add it to the list, until all the cluster centres are obtained
            for x in X:
                # Get the squared of the distance from the closest cluster centre
                _, distance_squared = self.closest_centre(x, current_cluster_centres, self.distance_metric)

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

        # Return numpy array of initial cluster centres chosen from the data points according to the kmeans++ heuristic
        return current_cluster_centres

    @staticmethod
    def closest_centre(x, current_cluster_centres, distance_metric):
        """
        The closest_centre method of the KMeans class takes in three input parameters x, current_cluster_centres and distance_metric
        and returns the closest cluster centre to x and the square of the distance between the closest cluster centre
        and x computed according ot the distance metric.

        Inputs:
            'x': x is 1 x d numpy array where d is the number of features. It is a single data point.

            'current_cluster_centres': Numpy array of current cluster centres at a stage of the iteration.

            'distance_metric': It stores the metric which is used to measure the distance between the data points.

            NOTE:   distance_metric other than 'euclidean' is not implemented in this algorithm,
                this parameter is a design choice and has been mentioned for extension purposes only.
                For all purposes in this algorithm distance will be measured as 'euclidean'.

        Outputs:
            'closest_centre': It is 1 x d numpy array where d is the number of features.
                It is a single data point among the current_cluster_centres which is closest to x.

            'distance_squared': The square of the distance between the closest cluster centre and x computed according ot the distance metric.
        """
        # List which stores the distance of x to each centre in current_cluster_centres
        distance_to_each_centre = []

        # For each centre in current_cluster_centres compute the euclidean distance to x and add it to the list
        for centre in current_cluster_centres:
            # Initialize distance to 0
            distance = 0

            # If the distance_metric is euclidean then use np.linalg.norm to compute the distance
            if distance_metric == 'euclidean':
                # calculating Euclidean distance
                # using linalg.norm()
                distance = np.linalg.norm(x - centre)

            # Update the list
            distance_to_each_centre.append(distance)

        # Return the closest cluster centre to x and the square of the distance between the closest cluster centre
        # and x computed according ot the distance metric.
        return np.where(distance_to_each_centre == np.amin(distance_to_each_centre))[0][0], np.square(
            np.amin(distance_to_each_centre))

    def fit(self, X):
        """
        The fit method takes in one input parameters - X
        It iteratively improves the clustering from the initial cluster centres
        which are initialized either by the user(initial_cluster_centres attribute) or according to the algorithm attribute.

        Inputs:
            'X': X is n x d numpy array where n is the number of data points and d is the number of features.
                It is the feature matrix.
        """
        # Initialize current_cluster_centres, inertia, and labels to None
        current_cluster_centres = None
        inertia = None
        labels = None

        # If the initial_cluster_centres attribute is None then use the algorithm parameter to appropriately initialize the centres
        if self.initial_cluster_centres_ is None:
            if self.algorithm == 'kmeans++':
                # Initialize the current_cluster_centres according to the kmeans_plus_plus_cluster_centre_initialization method
                current_cluster_centres = np.array(self.kmeans_plus_plus_cluster_centre_initialization(X), dtype=float)
            elif self.algorithm == 'lloyd':
                # Initialize the current_cluster_centres according to the lloyd_cluster_centre_initialization method
                current_cluster_centres = np.array(self.lloyd_cluster_centre_initialization(X), dtype=float)
        # Else use the user given centres to initialize the current_cluster_centres
        else:
            current_cluster_centres = np.array(self.initial_cluster_centres_, dtype=float)

        # Initialize the iterator for the while loop to 1
        iterator = 1

        # Keep iterating until either there is absolutely no change in cluster centres or maximum number of iterations is exceeded
        while iterator <= self.max_iter:
            # Store the recomputed cluster centres as a dictionary with the cluster indices as the key and the recomputed centre as the value
            recomputed_cluster_centres = {}

            # Store the total number of data points for each cluster as a dictionary with the cluster indices as the key and the number of data points as the value
            number_of_data_points_per_cluster = {}

            # Initialize the inertia of the clustering to 0.0
            inertia = 0.0

            # It stores the list of labels (cluster indices) of the data points after the clustering.
            labels = []

            # For each data point in X compute the closest centre to x among the current cluster centres and label the data point with that cluster index
            for x in X:
                # Call the closest_centre method to get the closest cluster centre to x and the square of the distance to that cluster centre
                label_of_x, distance_squared = self.closest_centre(x, current_cluster_centres, self.distance_metric)

                # Add the label to the list
                labels.append(label_of_x)

                # Update the value of inertia
                inertia += distance_squared

                # If the label_of_x is already a key in the dictionary
                if label_of_x in recomputed_cluster_centres:
                    # Add the data point to the respective recomputed cluster centre
                    recomputed_cluster_centres[label_of_x] = np.add(recomputed_cluster_centres[label_of_x], x)
                    # Update the number of data points in the cluster with index label_of_x
                    number_of_data_points_per_cluster[label_of_x] += 1
                # Else if the label_of_x is not a key in the dictionary
                else:
                    # Initialize recomputed_cluster_centres[label_of_x] with x
                    recomputed_cluster_centres[label_of_x] = x
                    # Initialize the number_data_points_per_cluster[label_of_x] with 1
                    number_of_data_points_per_cluster[label_of_x] = 1

            # For each label in recomputed cluster centres compute the cluster centres
            # by dividing the sum by the number of data points in the respective cluster
            for label in recomputed_cluster_centres:
                recomputed_cluster_centres[label] = np.true_divide(recomputed_cluster_centres[label],
                                                                   number_of_data_points_per_cluster[label])

            # Initialize the flag to be True
            flag = True

            # For each cluster check if the recomputed_cluster_centre is the same as the current cluster centre
            for i in range(0, self.n_clusters):
                # If the cluster index exists in recomputed_cluster_centres and it is not equal to the current_cluster_centre then set the flag as False
                if i in recomputed_cluster_centres and not np.array_equal(current_cluster_centres[i], recomputed_cluster_centres[i]):
                    flag = False
                # If the cluster index exists in recomputed_cluster_centres, then update the corresponding centre
                if i in recomputed_cluster_centres:
                    current_cluster_centres[i] = np.array(copy.deepcopy(recomputed_cluster_centres[i]), dtype=float)

            # If flag is True i.e. all the recomputed cluster centres are the same as the current cluster centres then break the loop
            if flag:
                break

            # Increment the iterator
            iterator += 1

        # Store the final centres of the clusters after iteratively improving the clustering
        self.cluster_centres_ = np.array(copy.deepcopy(current_cluster_centres), dtype=float)

        # Store the sum of square of distances of each data point from its respective cluster centre
        self.inertia_ = inertia

        # Store the final list of labels (cluster indices) of the data points after iteratively improving the clustering.
        self.labels_ = np.array(labels)

    def fit_predict(self, X):
        """
        The fit_predict method takes in one input parameters - X
        It calls the fit method which iteratively improves the clustering from the initial cluster centres
        which are initialized either by the user(initial_cluster_centres attribute) or according to the algorithm attribute
        and return the labels of the data points.

        Inputs:
            'X': X is n x d numpy array where n is the number of data points and d is the number of features.
                It is the feature matrix.

        Outputs:
            'labels_': It stores the final list of labels (cluster indices) of the data points after iteratively improving the clustering.
        """
        # Call the fit method to obtain the labellings
        self.fit(X)

        # Return the labels
        return self.labels_

    def predict(self, X):
        """
        The predict method takes in one input parameters - X and returns the list of labels assigned by the Clustering Model on X.

        Inputs:
            'X': X is n x d numpy array where n is the number of data points and d is the number of features.
                It is the feature matrix.

        Outputs:
            'labels': It stores the list of labels (cluster indices) of the data points.
        """
        # Initialize the list of labels to empty list
        labels = []

        # For each data point in X, compute the label_of_x and add it to the list
        for x in X:
            # Call the closest centre method to get the index of the cluster which is closest to x
            label_of_x, _ = self.closest_centre(x, self.cluster_centres_, self.distance_metric)
            # Append the list with the label_of_x
            labels.append(label_of_x)

        # Return the list of labels
        return labels
