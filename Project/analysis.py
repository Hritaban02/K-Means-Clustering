# MACHINE LEARNING GROUP-40 ASSIGNMENT-2

# Neha Dalmia 19CS30055
# Hritaban Ghosh 19CS30053

# Import the necessary libraries
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn
import copy
from models import KMeans
from utility import wangs_method_of_cross_validation
from Test_A import test_A
from Improved_Test_A import improved_test_A
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import homogeneity_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

# Print Required Information
print("\n###################################################")
print("\n MACHINE LEARNING GROUP-40 ASSIGNMENT-2 \n Neha Dalmia 19CS30055 \n Hritaban Ghosh 19CS30053")
print("\n Dataset Used: https://archive.ics.uci.edu/ml/datasets/liver+disorders")
print("\n###################################################")

# Data Preprocessing Stage for the Liver Disorders Data Set
print("\nDATA PREPROCESSING STAGE\n")

df = pd.read_csv("bupa.data", delimiter=',', header=None)
df.columns = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
final_df = copy.deepcopy(df)

# Add the class label as per the heuristic discussed in Turney, 1995,
# Cost-sensitive classification: Empirical evaluation of a hybrid genetic decision tree induction algorithm),
# who used the 6th field (drinks), after dichotomising, as a dependent variable for classification
final_df['drinks >= 3'] = final_df.apply(lambda x: int(x['drinks'] >= 3), axis=1)

# Dropping the columns 'selector' and 'drinks'
final_df = final_df.drop(columns=['selector'])
final_df = final_df.drop(columns=['drinks'])

# Normalize the Dataset
final_df.loc[:, final_df.columns != 'drinks >= 3'] = (final_df.loc[:, final_df.columns != 'drinks >= 3']-final_df.loc[:, final_df.columns != 'drinks >= 3'].min())/(final_df.loc[:, final_df.columns != 'drinks >= 3'].max()-final_df.loc[:, final_df.columns != 'drinks >= 3'].min())

# Check for Null Values (Missing Data)
print("Checking for null values in the dataframe: ")
print(final_df.isnull().sum())
# No null value found

# Plot the Attributes Correlation Heatmap for the Liver Disorders Dataset
f, ax = plt.subplots(figsize=(10, 6))
corr = final_df.iloc[:, 0:-1].corr()
hm = seaborn.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f', linewidths=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Liver Disorder Attributes Correlation Heatmap', fontsize=14)
f.savefig("Analysis/Liver_Disorder_Attributes_Correlation_Heatmap", bbox_inches='tight')
plt.close(f)

# Split the dataset into feature matrix X and class label y
X = final_df.iloc[:, 0:-1].values
y = final_df.iloc[:, -1].values

# Basic Information for the User
print("\n###################################################")
print("\n INFORMATION FOR THE USER")
print("\n lloyd cluster centre initialization : The initial cluster centres are k randomly selected data points from the dataset.")
print(" kmeans++ cluster centre initialization : The initial cluster centres are selected according to the kmeans++ heuristic.")
print(" Note: The algorithm chosen here will be used for all the analysis in subsequent steps.")
print("\n###################################################")

# Prompt the user to enter the number of clusters k and method of cluster centre initialization (Lloyd or K-Means++)
while 1:
    user_given_k = input("\n Enter the value of k which is the number of clusters (k>1): ")
    user_given_k = int(user_given_k)
    if user_given_k > 1:
        break
    else:
        print("\n The value entered is less than or equal to 1!!! Please try again. \n")
        print("\n--------------------------------------------------------------------")
        continue

print("\n============================================================================")

while 1:
    user_given_algorithm = input("\n Enter the method to be used for cluster centre initialization (lloyd or kmeans++): ")
    user_given_algorithm = user_given_algorithm.strip()
    if user_given_algorithm == "lloyd":
        break
    elif user_given_algorithm == "kmeans++":
        break
    else:
        print("\n The value entered is neither \'lloyd\' nor \'kmeans++\'!!! Please try again. \n")
        print("\n--------------------------------------------------------------------")
        continue

print("\n###################################################")

print(f"\n Training the Kmeans Model with number of clusters {user_given_k} and initialization method {user_given_algorithm} as parameters...")

# Set the Model Parameters
kmeans = KMeans(n_clusters=user_given_k, algorithm=user_given_algorithm)

# Train the model on the Feature Matrix
y_predict = kmeans.fit_predict(X)

# Measure the clustering performance for user given k and initialization method
print("\nMeasuring Clustering Performance using available ground truth\n")

print("homogeneity_score:", homogeneity_score(y, y_predict))
print("adjusted_rand_score:", adjusted_rand_score(y, y_predict))
print("normalized_mutual_info_score:", normalized_mutual_info_score(y, y_predict))
print("fowlkes_mallows_score:", fowlkes_mallows_score(y, y_predict))

print("\n--------------------------------------------------")

print("\nMeasuring Clustering Performance without using ground truth\n")

print("silhouette_score:", silhouette_score(X, y_predict))
print("calinski_harabasz_score:", calinski_harabasz_score(X, y_predict))

# Perform analysis to determine the optimal number of clusters for the Liver Disorders Dataset
print("\n###################################################")

print(" ANALYSIS TO DETERMINE OPTIMAL NUMBER OF CLUSTERS")

silhouette_score_for_each_k = []
calinski_harabasz_score_for_each_k = []
for number_of_clusters in range(2, len(np.unique(np.array(y)))+10):
    kmeans = KMeans(n_clusters=number_of_clusters, algorithm=user_given_algorithm)
    y_predict = kmeans.fit_predict(X)
    silhouette_score_for_each_k.append(silhouette_score(X, y_predict))
    calinski_harabasz_score_for_each_k.append(calinski_harabasz_score(X, y_predict))

# Plot Silhouette Index versus Number Of Clusters
print("\n Plotting and Saving Silhouette Index versus Number Of Clusters graph in Analysis Folder...")
plt.plot(range(2, len(np.unique(np.array(y)))+10), silhouette_score_for_each_k)
plt.xlabel("Number Of Clusters (k)")
plt.ylabel("Silhouette Index")
plt.title("Silhouette Index versus Number Of Clusters")
plt.savefig("Analysis/Silhouette_Index_versus_Number_Of_Clusters", bbox_inches='tight')
plt.close()
print(" Done")

# Plot Calinski Harabasz Index versus Number Of Clusters
print("\n Plotting and Saving Calinski Harabasz Index versus Number Of Clusters graph in Analysis Folder...")
plt.plot(range(2, len(np.unique(np.array(y)))+10), calinski_harabasz_score_for_each_k)
plt.xlabel("Number Of Clusters (k)")
plt.ylabel("Calinski Harabasz Index")
plt.title("Calinski Harabasz Index versus Number Of Clusters")
plt.savefig("Analysis/Calinski_Harabasz_Index_versus_Number_Of_Clusters", bbox_inches='tight')
plt.close()
print(" Done")

print("\n--------------------------------------------------")

print("\n Perform Wang's Method Of Cross Validation")

while 1:
    c = input("\n Enter the value of c which is the number of permutations in Wang's Method Of Cross Validation(c>0): ")
    c = int(c)
    if c > 0:
        break
    else:
        print("\n The value entered is less than or equal to 0!!! Please try again. \n")

average_disagreements_for_each_k = []
for number_of_clusters in range(2, len(np.unique(np.array(y)))+10):
    average_disagreements_for_each_k.append(wangs_method_of_cross_validation(X, n_clusters=number_of_clusters,
                                                                             algorithm=user_given_algorithm, c=c))

# Plot Wang's Method Of Cross Validation on a graph
print("\n Plotting and Saving Wang's Method Of Cross Validation graph in Analysis Folder...")
plt.plot(range(2, len(np.unique(np.array(y)))+10), average_disagreements_for_each_k)
plt.xlabel("Number Of Clusters (k)")
plt.ylabel("Average Disagreements")
plt.title("Wang's Method Of Cross Validation")
plt.savefig("Analysis/Wangs_Method_Of_Cross_Validation", bbox_inches='tight')
plt.close()
print(" Done")

print("\n###################################################")

print("\n OPTIMAL NUMBER OF CLUSTERS FOR THE LIVER DISORDERS DATASET")

# After Analysis Optimal Number of Clusters was found to be 2
optimal_number_of_clusters = 2

print(f"\n Based upon the analysis performed in the previous steps, it is clear that for the Liver Disorders Dataset, optimal number of clusters = {optimal_number_of_clusters}")

print("\n###################################################")

# Perform Test A as described in the assignment
print("\n Performing Test A")
print(" Test A performs random initialization for the initial cluster centres.")

homogeneity_score_avg_list = []
adjusted_rand_score_avg_list = []
normalized_mutual_info_score_avg_list = []
fowlkes_mallows_score_avg_list = []

number_of_iter = 50

for i in range(0, number_of_iter):
    homogeneity_score_avg, adjusted_rand_score_avg, normalized_mutual_info_score_avg, fowlkes_mallows_score_avg = test_A(optimal_number_of_clusters, X, y)

    homogeneity_score_avg_list.append(homogeneity_score_avg)
    adjusted_rand_score_avg_list.append(adjusted_rand_score_avg)
    normalized_mutual_info_score_avg_list.append(normalized_mutual_info_score_avg)
    fowlkes_mallows_score_avg_list.append(fowlkes_mallows_score_avg)

print("\n Plotting and Saving Test A - Homogeneity Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), homogeneity_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Homogeneity Score")
plt.title("Test A - Homogeneity Score Analysis")
ax = plt.gca()
ax.set_ylim([0.0, 1.0])
plt.savefig("Analysis/Test_A_Homogeneity_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" Homogeneity Score in Test A: ")
print(" Mean = ", np.mean(homogeneity_score_avg_list))
print(" Standard Deviation = ", np.std(homogeneity_score_avg_list))

print("\n Plotting and Saving Test A - Adjusted Rand Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), adjusted_rand_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Adjusted Rand Score")
plt.title("Test A - Adjusted Rand Score Analysis")
ax = plt.gca()
ax.set_ylim([-1.0, 1.0])
plt.savefig("Analysis/Test_A_Adjusted_Rand_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" ARI in Test A: ")
print(" Mean = ", np.mean(adjusted_rand_score_avg_list))
print(" Standard Deviation = ", np.std(adjusted_rand_score_avg_list))

print("\n Plotting and Saving Test A - Normalized Mutual Info Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), normalized_mutual_info_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Normalized Mutual Info Score")
plt.title("Test A - Normalized Mutual Info Score Analysis")
ax = plt.gca()
ax.set_ylim([0.0, 1.0])
plt.savefig("Analysis/Test_A_Normalized_Mutual_Info_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" NMI Index in Test A: ")
print(" Mean = ", np.mean(normalized_mutual_info_score_avg_list))
print(" Standard Deviation = ", np.std(normalized_mutual_info_score_avg_list))

print("\n Plotting and Saving Test A - Fowlkes Mallows Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), fowlkes_mallows_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Fowlkes Mallows Score")
plt.title("Test A - Fowlkes Mallows Score Analysis")
ax = plt.gca()
ax.set_ylim([0.0, 1.0])
plt.savefig("Analysis/Test_A_Fowlkes_Mallows_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" Fowlkes Mallows Score  in Test A: ")
print(" Mean = ", np.mean(fowlkes_mallows_score_avg_list))
print(" Standard Deviation = ", np.std(fowlkes_mallows_score_avg_list))

print("---------------------------------------------------")

# Perform Improved Test A based on the kmeans++ initialization heuristic
print("\n Performing Improved Test A")
print(" Improved Test A performs kmeans++ initialization for the initial cluster centres.")

homogeneity_score_avg_list = []
adjusted_rand_score_avg_list = []
normalized_mutual_info_score_avg_list = []
fowlkes_mallows_score_avg_list = []

number_of_iter = 50

for i in range(0, number_of_iter):
    homogeneity_score_avg, adjusted_rand_score_avg, normalized_mutual_info_score_avg, fowlkes_mallows_score_avg = improved_test_A(optimal_number_of_clusters, X, y)
    homogeneity_score_avg_list.append(homogeneity_score_avg)
    adjusted_rand_score_avg_list.append(adjusted_rand_score_avg)
    normalized_mutual_info_score_avg_list.append(normalized_mutual_info_score_avg)
    fowlkes_mallows_score_avg_list.append(fowlkes_mallows_score_avg)

print("\n Plotting and Saving Improved Test A - Homogeneity Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), homogeneity_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Homogeneity Score")
plt.title("Improved Test A - Homogeneity Score Analysis")
ax = plt.gca()
ax.set_ylim([0.0, 1.0])
plt.savefig("Analysis/Improved_Test_A_Homogeneity_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" Homogeneity Score in Improved Test A: ")
print(" Mean = ", np.mean(homogeneity_score_avg_list))
print(" Standard Deviation = ", np.std(homogeneity_score_avg_list))

print("\n Plotting and Saving Improved Test A - Adjusted Rand Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), adjusted_rand_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Adjusted Rand Score")
plt.title("Improved Test A - Adjusted Rand Score Analysis")
ax = plt.gca()
ax.set_ylim([-1.0, 1.0])
plt.savefig("Analysis/Improved_Test_A_Adjusted_Rand_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" ARI in Improved Test A: ")
print(" Mean = ", np.mean(adjusted_rand_score_avg_list))
print(" Standard Deviation = ", np.std(adjusted_rand_score_avg_list))

print("\n Plotting and Saving Improved Test A - Normalized Mutual Info Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), normalized_mutual_info_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Normalized Mutual Info Score")
plt.title("Improved Test A - Normalized Mutual Info Score Analysis")
ax = plt.gca()
ax.set_ylim([0.0, 1.0])
plt.savefig("Analysis/Improved_Test_A_Normalized_Mutual_Info_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" NMI Index in Improved Test A: ")
print(" Mean = ", np.mean(normalized_mutual_info_score_avg_list))
print(" Standard Deviation = ", np.std(normalized_mutual_info_score_avg_list))

print("\n Plotting and Saving Improved Test A - Fowlkes Mallows Score Analysis graph in Analysis Folder...")
plt.plot(range(0, number_of_iter), fowlkes_mallows_score_avg_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Average Fowlkes Mallows Score")
plt.title("Improved Test A - Fowlkes Mallows Score Analysis")
ax = plt.gca()
ax.set_ylim([0.0, 1.0])
plt.savefig("Analysis/Improved_Test_A_Fowlkes_Mallows_Score_Analysis", bbox_inches='tight')
plt.close()
print(" Done")
print(" Fowlkes Mallows Score in Improved Test A: ")
print(" Mean = ", np.mean(fowlkes_mallows_score_avg_list))
print(" Standard Deviation = ", np.std(fowlkes_mallows_score_avg_list))

print("\n###################################################")

print("\n EXTRA: ANALYSIS OF CLUSTERING MODELS")
# Features of the dataset 'mcv', 'alkphos', 'sgpt', 'sgot', and 'gammagt'
colors = ['red', 'blue', 'green', 'yellow', 'cyan']
combinations_object = itertools.combinations([0, 1, 2, 3, 4], 2)
combinations_list = list(combinations_object)

for combination in combinations_list:
    # Split the dataset into feature matrix X and class label y
    sub_X = final_df.iloc[:, list(combination)].values
    sub_y = final_df.iloc[:, -1].values

    # Perform clustering using only these two attributes and plot the graph
    kmeans = KMeans(n_clusters=optimal_number_of_clusters, algorithm=user_given_algorithm)
    label_of_subset_x = kmeans.fit_predict(sub_X)

    print(f"\n Plotting and Saving K-Means Clustering using {final_df.columns[combination[0]]} and {final_df.columns[combination[1]]} graph in Extra_Analysis Folder...")
    plt.scatter(sub_X[label_of_subset_x == 0, 0], sub_X[label_of_subset_x == 0, 1], s=50, color=colors[combination[0]], label='Cluster 1')
    plt.scatter(sub_X[label_of_subset_x == 1, 0], sub_X[label_of_subset_x == 1, 1], s=50, color=colors[combination[1]], label='Cluster 2')
    plt.scatter(kmeans.cluster_centres_[:, 0], kmeans.cluster_centres_[:, 1], s=100, color='black', label='Centroids')
    plt.title(f"K-Means Clustering using {final_df.columns[combination[0]]} and {final_df.columns[combination[1]]}")
    plt.xlabel(f"{final_df.columns[combination[0]]}")
    plt.ylabel(f"{final_df.columns[combination[1]]}")
    plt.legend()
    plt.savefig(f"Extra_Analysis/Clustering_using_{final_df.columns[combination[0]]}_and_{final_df.columns[combination[1]]}", bbox_inches='tight')
    plt.close()
    print(" Done")

    # Measure the clustering performance for user given k and initialization method
    print("\nMeasuring Clustering Performance using available ground truth\n")

    print("homogeneity_score:", homogeneity_score(sub_y, label_of_subset_x))
    print("adjusted_rand_score:", adjusted_rand_score(sub_y, label_of_subset_x))
    print("normalized_mutual_info_score:", normalized_mutual_info_score(sub_y, label_of_subset_x))
    print("fowlkes_mallows_score:", fowlkes_mallows_score(sub_y, label_of_subset_x))

    print("\n--------------------------------------------------")

    print("\nMeasuring Clustering Performance without using ground truth\n")

    print("silhouette_score:", silhouette_score(sub_X, label_of_subset_x))
    print("calinski_harabasz_score:", calinski_harabasz_score(sub_X, label_of_subset_x))

    print("---------------------------------------------------")

print("\n###################################################")
