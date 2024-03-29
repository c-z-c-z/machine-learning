
# Dota hero clustering with k-means

## What is k-means clustering?
K-means clustering is an unsupervised machine learning algorithm which attempts to group a set of observations into a specified number of clusters based on similarity metrics. When an instance of the algorithm is created, cluster centroids are randomly initialized based on the desired number of clusters. Each observation is assigned a cluster based on how similar its values are to those of the centroid.  At each iteration, the location of each centroid is recomputed and observations are reallocated if necessary. This continues until the locations of the centroids and cluster allocations no longer change. 
The algorithm is often applied to unlabeled data in order to uncover structure, but in this specific task we do have the true category labels, so they were used to evaluate the model's performance.

## Implementation and libraries
This task was carried out using pandas, matplotlib and seaborn, and scikit-learn.

## The task
The task at hand was to use k-means clustering to group Dota heroes into three clusters based on their base attributes. Given that the true category membership of the heroes is known, the performance of the k-means algorithm could be evaluated. High accuracy was achieved after exploratory analysis and subsequent feature selection. 

## Data set
The data set is publicly available from https://www.kaggle.com/datasets/prokid1911/dota-2-all-hero-data-727d. Many thanks to Mridul Gupta for putting this data set together.

## Repository contents
This repository consists of a .ipynb file (chosen to display visualizations handily) and the dataset (also linked in the previous section).

## Status
This project is complete. I have some interest in pursuing further Dota-related machine learning projects, namely clustering based on the many hero roles instead of the three hero categories, which is a much more complex task.
