# Welcome to Clustering!



Solve Business problem in Python and R
---

   There is a big Mall in a specific city that contains information of its clients that subscribed to the membership card. When the customers subscribed to the membership card, they provided info like their gender, age, annual income, spending score ( values between 1 and 100 so that the close to 1 spending score less the client spent, and close to 100 spending score more the client spent, score based to some criteria: income, the amount of dollars spent, number of times per week shown in mall etc.).
   
  **My job is to segment the clients into groups based to annual income and spending score** ( 2 variables for simplicity ). The mall doesn’t know – which are the segments and how many segments are, this is typically clustering problem because we don’t know the answers.
  
  Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters/groupings you would have never thought of otherwise. In the following, you will understand and learn how to implement the following Machine Learning Clustering model:
  
***Hierarchical Clustering***

![whathcdoesforyou](https://user-images.githubusercontent.com/25092397/34782400-3559066e-f631-11e7-9b67-5142858aceda.png)

*Enjoy Machine Learning!*

## Hierarchical Clustering Intuition

  In this section will talk about Hierarchical Clustering Algorithm (HC). It allows you to cluster data, it’s very convenient tool for discovering categories groups of data set and in this section will learn how to understand HC in intuitive levels. Let’s dive into it:  
  Let’s decide we have 2 variables in our data set and we decide to plot those variables on X and Y axes.  
  
---
The question is: 
---
Can we identify certain groups among variables and how would we go about doing it ?!  
Yes, let’s see how!

What is Hierarchical Clustering ?! 
If you have points on your scatterplot or data points, if you apply Hierarchical Clustering (HC for short) what will happen you will get clustering very similar with K-Means. Most of time same as K-Means but different process.  

First thing: 2 types of HC : Agglomerative & Divisive.
Agglomerative is the bottom up approach and you will see in more detail what that means.
I’ll focus on Agglomerative approach.  

**How does it work?!**

## Step by step approach:


![step1](https://user-images.githubusercontent.com/25092397/34783764-154905d2-f635-11e7-80c3-2f51f0b92128.png)


![step2](https://user-images.githubusercontent.com/25092397/34783779-2183e0f6-f635-11e7-954e-94ef3d5e343c.png)


![step3](https://user-images.githubusercontent.com/25092397/34783795-2c046514-f635-11e7-8165-600decaf822a.png)


![step4](https://user-images.githubusercontent.com/25092397/34783802-33292d0c-f635-11e7-8546-330eb94ac071.png)



One thing that stands out here, the word **‘closest cluster‘** ( you can use Euclidian, Manhattan, Chessboard distance etc. ) but here we are actually talking about proximity of clusters not point. How you measure distance between clusters?! 
Euclidian distance (just to get this out of the way) in 2-dimensional space (X and Y axes) is calculated like that. 
![euclidiandistance](https://user-images.githubusercontent.com/25092397/34784104-024462f0-f636-11e7-9e14-d6d440595d74.png)
  

Distance between two clusters:   
***Option 1 : closest points   
Option 2 : furthest points  
Option 3 : average distance   
Option 4 : distance between centroids***    

It’s a very important part of Hierarchical Clustering (HC) what you define as a distance between 2 clusters because that can significantly impact the output of your algorithm. I'm not going deep into this is just something to remember, to note, based on your particular situation (business problem, data science, etc.) you need to define the distance. 

The way that HC agglomerative works is that it maintains memory of how we went to this process and that memory is stored in dendrogram. 


## How do dendrograms work?!  

Dendrogram is kind of memory of the HC algorithm. Is going to remember every single step that we'll perform.
Two points have a certain dissimilarity which is measured by the distance between them, so the distance represent the dissimilarity between these 2 points. The point here is that, the further away 2 points are, the more dissimilarity they have. And that is measured (captured) on in dendrogram by the height of the bar.  
![howdodendogramswork](https://user-images.githubusercontent.com/25092397/34784229-62e861f6-f636-11e7-92c0-2dd578f5c588.png)
  
Our final step is to combine the last 2 clusters, and we'll represent that on dendrogram.
That is how we construct the dendrogram slowly from the bottom to up, and at the end we will get:  
![howdodendogramswork2](https://user-images.githubusercontent.com/25092397/34784286-87aee6cc-f636-11e7-9b44-e82dd4fafa83.png)

The optimal number of clusters: we need to find the largest vertical distance that we can make without  crossing any other horizontal line and then we need to count the number of vertical line of this level. In this example is 2 





## Hierarchical Clustering in R Explanation

We are going to import the mall data set first then we use a dendrogram to find the optimal number of clusters, then we will fit h.c. to our mall dataset and we finally will visualize the results.

Build Dendrogram
	We start by creating a variable dendrogram then we'll use a class hclust. The first argument is a dissimilarity structure as produced by distance, in our case will be the distance metrics of our dataset  X which is matrix that tells for each pair of customers the Euclidian distance between each 2.
The second parameter is method, simply the method used to find the clusters and we'll use `‘ward’` method.
Is a method that tries to minimize the variance within clusters, not like in K-Means where we tried to minimize the WCSS (Within Clusters Sum of Squares).   
  
  
To find the right number of clusters, we need to find the largest vertical distance that we can make without crossing any other horizontal line and then to count them ( on that level )
	Since we have the optimal number of clusters, we will fit hc algorithm to our date X with the right number of clusters.  
  
1.	Create an object of hc class ( but we did it already )  

2.	Use this object to build our vector of clusters (will tell which cluster each customer belongs to and we'll use one of the hc class method which is the cutree (first parameter is our dendrogram(hc), second one is K = number of clusters and we'll leave default for the rest of the  parameters)  

We found the right number of clusters and we fitted correctly to our dataset and now, time for fun: visualizing ! 

### Visualising the clusters and interpretation

![clusters](https://user-images.githubusercontent.com/25092397/34784460-fa7073ba-f636-11e7-9aea-a6fb7a991465.png)




---

**Code for Python** 
---  
		
```# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 20, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 20, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 20, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 20, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 20, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()  
```
---

**Code for R** 
---  

```# Hierarchical Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting Hierarchical Clustering to the dataset
hc = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(dataset,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')  
```


