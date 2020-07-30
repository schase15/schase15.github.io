---
layout: post
title: K Nearest Neighbors Algorithm with Basic Python
---

## K Nearest Neighbors Algorithm
#### *De-mystifying the black box by building a working algorithm with nothing but basic Python*
  
*By Steven Chase*

As a beginer Data Scientist or as an outsider looking in, Machine Learning can seem like a mystical arena. KNN, XGBoost, DBSCAN Clustering, Random Forest... For many, these names are enough to intimidate them from even peaking under the hood. Besides, they are magical black boxes. You don't need to know what they do as long as you know how to use them. You simply put in the data that you have, and poof, the black box predicts the unknown. However, I am here to pop the hood and explain how at least one of these, K Nearest Neighbors, black boxes work. Because an informed operator of these models will produce more informed and subsquently far superior predictions. 

## K Nearest Neighbors

K Nearest Neighbors (KNN) is a supervised machine learning algorithm. While it is most commonly used for classification it can also be used for regression problems. The basic intuition behind KNN can be understood simply by looking at its name, K Nearest Neighbors. While trying to decide how to classify an observation, the KNN model will look for the most similiar (nearest neighbor) observations in the training dataset. The K is just an input from the user that tells the model how many nearest neighbors to look for. For example, if I had a vehicle and was trying to determine which type it was, I could look in the parking lot and find the three (my k value) most similar vehicles to mine. Knowing what type of vehicles the ones from the parking lot are, I can determine which type my car is. This is all there is to KNN.

Now that you have a grasp of the concept, lets dive in to the algorithm and the mathmatics that make it work.

## Theory Behind KNN

K Nearest Neighbors is a simple machine learning model that makes predictions based off of the most similar observations in its traing data. While it can be very powerful, its predictive capability is limited to observations that are similar to what the training data it has in memory.

Unlike most other models, KNN does not 'learn' from its training dataset. Instead it holds the entire training set in memory and then compares the new observation to its stored data. KNN performs no work until a prediction is required.

When a prediction is required it does exactly what it name says. The model examines the new observation and finds the most similar records (nearest neighbors) that it holds in its training set. The number of neighbors (k) the model selects from its training data is defined by the user. 

A prediction can be made by either returning the most common outcome (classification) or by taking the average (regression).

**Some Important Notes on Using KNN**

- KNN is a simple model to implement, but as a result it is limited in the types of data it can take as input. When working with KNN the phrase "garbage in, garbage out" is never more accurate. KNN does not handle categorical variables so everything must be pre-processed to include numerical values only. Additionally, as you will see in the next section, the nearest neighbors are found by calculating the distances between the new observation and the records held in memory. Those with the smallest distances are considered most similar. Intuatively, you should understand the importance of scaling your data (so that they are all being measured on the same metric) before running a KNN model. 

## Algorithm Implementation
Now that we understand the theory behind KNN, we can implement our own algorithm from scratch in three setps.

Step 1. Calculate Euclidean Distance

Step 2. Get Nearest Neighbors

Step 3. Make Predicitons

### Step 1: Calculate Euclidean Distance

The formula for Euclidean Distance is:

\sum_{i=1}^n (x_{i}-y_{i})^2

The Euclidean Distance may sound complicated, and the formula may look intimitating. But the concept is very simple. The Euclidean Distance is the ordinary straight line distance between two data points. The formula can be simply derived from the Pythagorean formula: 

Pythagorean Theorem:

$c^2 = a^2 + b^2 $

Where c is the Euclidean distance between datapoints a and b.

For simplicity, let's first say that data point a and b are 2-Dimensial and described by their x and y coordinates.

a= ($a_{1}, a_{2}$) and b= ($b_{1}, b_{2}$).

To help with understanding, visually we can view this on a graph. On the graph below data points a and b have been ploted (represented by the large arrowheads). The Euclidean distance we are trying to calculate is the vector drawn in yellow.

By drawing in the vectors representing the datapoints (in blue and red) we can clearly see that the yellow Euclidean distance is simply the hypotenuse of the triangle. 

<img src="/img/pyth_tri.png">


We know that the length of the vectors for point a and b can be calculated by |$a_{1} - b_{1}$| and |$a_{2} - b_{2}$|

So it follows that,

$c^2 = (a_{1}-b_{1})^2 + (a_{2}-b_{2})^2 $

$c = \sqrt{(a_{1}-b_{1})^2 + (a_{2}-b_{2})^2} $

This is the basic formula for Euclidean Distance for 2-D datapoints.

However, this can be expanded to 3-D and beyond leaving us with the finalized formula of Euclidean Distance we saw above.

$c = \sqrt{(a_{1}-b_{1})^2 + (a_{2}-b_{2})^2 + (a_{3}-b_{3})^2 + ... + (a_{n}-b_{n})^2} $

More succintly written as,

$ \sum_{i=1}^n (a_{i}-b_{i})^2 $

**Given our understanding of the mathmatics behind calculating the Euclidean distance, how can we write that calculation in python?**

When working with datasets, each row is a datapoint. Each column represents another dimension of the datapoint (but that leads us off track into the subject of dimensionality and furthermore the curse of dimensionality). If you don't know what I am talking about, it is a crucial concept to understand when building your own machine learning models. For the purposes of this article, I will leave further research of that topic to you.

To calculate the Euclidean distance between two points we can use the following function:

<img src="/img/euc_dis.png">

The function above assumes that the output target is the last column of the datapoint and is therefore not included in the distance calculations. In our final KNN class we will have a fit method that saves the X values and the target separately.  

### Step 2: Get Nearest Neighbors

