# Algorithm Summary

# Pros and Cons

# When to use which one

# Evaluation

There are many different clustering algorithms, and each has its own strengths and weaknesses. 

Some algorithms are better at handling high-dimensional data sets, while others are better at handling data sets with many outliers. 




Data preprocessing
Most clustering algorithms require that the data be standardized, meaning that each variable (column) in the data set should be transformed so that it has a mean of zero and a standard deviation of one.
	Exceptions: k-means clustering is not affected by variable scaling, because the distance metric it uses (Euclidean distance) is not affected by variable scaling. Stil better to scale to get better results.


Choice as deciding factor
If you’re working with numerical data, K-Means is a good choice for finding groups of similar items. If you want to find relationships between items, Hierarchical Clustering is a good choice. And if you’re working with a large dataset, Parallel Coordinates are a good choice for speed.

The size of your data: Some algorithms work better with large data sets, while others work better with small data sets.
The dimensionality of your data: Some algorithms are better at handling high-dimensional data, while others work better with low-dimensional data.
The type of data: Some algorithms work better with numeric data, while others work better with categorical data.
The structure of your data: Some algorithms are better at handling data that is well-structured, while others work better with data that is less structured.
The speed of the algorithm: Some algorithms are faster than others. If speed is important to you, then you will want to choose a fast algorithm.
The accuracy of the algorithm: Some algorithms are more accurate than others. If accuracy is important to you, then you will want to choose a more accurate algorithm.



Evaluation
Visualize the clusters using a scatter plot. This will help you to see if the data has been clustered correctly.
Check the Silhouette Coefficient of the clusters. This metric measures how well each data point has been clustered. A high Silhouette Coefficient means that the data points have been clustered correctly.
Check the Calinski-Harabasz Score of the clusters. This metric measures the compactness and separation of the clusters. A high Calinski-Harabasz Score means that the data has been clustered correctly.
Check the Davies-Bouldin Score of the clusters. This metric measures the similarity of the clusters. A low Davies-Bouldin Score means that the data has been clustered correctly.

Techniques to experiment for better results
Different algorithms make different assumptions about the data, so it’s worth trying out a few to see if any give better results.
Another option is to pre-process the data in different ways, such as normalizing it or applying some dimensionality reduction technique. 
 playing around with the various parameters of the algorithm


4 basic types of clustering algos:
connectivity based - how each data point is related to the others in the dataset. Hierarchical relation. Could be agglomorative or divisive. 
the complexity of the algorithm may turn out to be excessive or simply inapplicable for datasets with little to no hierarchy. It also shows poor performance: due to the abundance of iterations, complete processing will take an unreasonable amount of time. On top of that, you won’t get a precise structure using the hierarchical algorithm.

Centroid-based clustering -
a negligent edge of each cluster, because the priorities are set on the center of the cluster, not on its borders;
an inability to create a structure of a dataset with objects that can be classified to multiple clusters in equal measure;
a need to guess the optimal k number, or a need to make preliminary calculations to specify this gauge.

Expectation-maximization algorithm -
it calculates the relation probability of each dataset point to all the clusters we’ve specified
The main “tool” that is used for this clusterization model is Gaussian Mixture Models (GMM) – the assumption that the points of the dataset generally follow the Gaussian distribution.
EM algorithm allows the points to classify for two or more clusters

density-based clustering -
The clusters determined with DBSCAN can have arbitrary shapes, thereby are extremely accurate
 If the dataset consists of variable density clusters, the method shows poor results. It also might not be your choice if the placement of objects is too close

If you’re working with numerical data, K-Means is a good choice for finding groups of similar items. If you want to find relationships between items, Hierarchical Clustering is a good choice. And if you’re working with a large dataset, Parallel Coordinates are a good choice for speed.

Visualize the clusters using a scatter plot. This will help you to see if the data has been clustered correctly.
Check the Silhouette Coefficient of the clusters. This metric measures how well each data point has been clustered. A high Silhouette Coefficient means that the data points have been clustered correctly.
Check the Calinski-Harabasz Score of the clusters. This metric measures the compactness and separation of the clusters. A high Calinski-Harabasz Score means that the data has been clustered correctly.
Check the Davies-Bouldin Score of the clusters. This metric measures the similarity of the clusters. A low Davies-Bouldin Score means that the data has been clustered correctly.


 even such a masterpiece as DBSCAN has a drawback. If the dataset consists of variable density clusters, the method shows poor results. It also might not be your choice if the placement of objects is too close, and the ε parameter can’t be estimated easily.

Unlike the centroid-based models, the EM algorithm allows the points to classify for two or more clusters – it simply presents you the possibility of each event, using which you’re able to conduct further analysis. 

kmeans
a negligent edge of each cluster, because the priorities are set on the center of the cluster, not on its borders;
an inability to create a structure of a dataset with objects that can be classified to multiple clusters in equal measure;
a need to guess the optimal k number, or a need to make preliminary calculations to specify this gauge.
https://www.researchgate.net/figure/Advantage-and-Disadvantage-of-various-Clustering-Techniques_tbl2_258285203

