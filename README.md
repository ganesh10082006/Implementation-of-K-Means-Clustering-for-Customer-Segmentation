# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read Data: Read data from the CSV file Mall_Customers_EX8.csv into a pandas DataFrame.
2. Visualize Data: Plot a scatter plot of the data points using 'Annual Income (k$)' as x-axis and 'Spending Score (1-100)' as y-axis.
3. Apply K-means Clustering: Apply K-means clustering with k=5 clusters on the features 'Annual Income (k$)' and 'Spending Score (1-100)'.
4. Visualize Clusters: Plot the clustered data points along with their centroids and cluster boundaries using different colors for each cluster.


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: GANESH G.
RegisterNumber: 212223230059
*/

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data =pd.read_csv('/content/Mall_Customers_EX8.csv')
data

x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
x

plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(x)

centroids=kmeans.cluster_centers_
labels=kmeans.labels_
print("centroids :\n", centroids)
print("labels :\n",labels)

colors = ['r', 'g', 'b', 'c', 'm']
for i in range(k):
    cluster_points = x[labels == i]
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i + 1}')
    distances = euclidean_distances(cluster_points, [centroids[i]])
    radius = np.max(distances)
    circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
    plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```

## Output:
### 1.Data:
![image](https://github.com/user-attachments/assets/46f73754-4aba-4ac2-966d-b40fb6672f06)
### 2.Scatter Points:
![image](https://github.com/user-attachments/assets/c1989162-b73f-480c-856f-eb64313aa42a)
### 3.Kmeans.fit:
![image](https://github.com/user-attachments/assets/0181151a-e537-43a1-8881-7094aecc2f01)
### 4.Centroids and Labels:
![image](https://github.com/user-attachments/assets/7607b847-3c82-47a0-9d70-d541e9a9b4db)
### 5.Kmeans_clusters:
![image](https://github.com/user-attachments/assets/bb8e0c55-d978-40d3-aaf1-702c36c83034)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
