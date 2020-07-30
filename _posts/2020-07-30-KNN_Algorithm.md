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

<img src="/img/euc_formula.png">

The Euclidean Distance may sound complicated, and the formula may look intimitating. But the concept is very simple. The Euclidean Distance is the ordinary straight line distance between two data points. The formula can be simply derived from the Pythagorean formula: 

<img src="/img/a_b_corr.png">

To help with understanding, visually we can view this on a graph. On the graph below data points a and b have been ploted (represented by the large arrowheads). The Euclidean distance we are trying to calculate is the vector drawn in yellow.

By drawing in the vectors representing the datapoints (in blue and red) we can clearly see that the yellow Euclidean distance is simply the hypotenuse of the triangle. 

<img src="/img/pyth_tri.png">


<img src="/img/formula_proof.png">

**Given our understanding of the mathmatics behind calculating the Euclidean distance, how can we write that calculation in python?**

When working with datasets, each row is a datapoint. Each column represents another dimension of the datapoint (but that leads us off track into the subject of dimensionality and furthermore the curse of dimensionality). If you don't know what I am talking about, it is a crucial concept to understand when building your own machine learning models. For the purposes of this article, I will leave further research of that topic to you.

To calculate the Euclidean distance between two points we can use the following function:

<img src="/img/euc_dis.png">

The function above assumes that the output target is the last column of the datapoint and is therefore not included in the distance calculations. In our final KNN class we will have a fit method that saves the X values and the target separately.  

### Step 2: Get Nearest Neighbors

Now that we know how to calculate the distance betweeen two datapoints, we can find the k nearest neighbors (closest instances in the training data) to our new datapoint. 

First we can use the above function to calculate the distances between our new observation and each datapoint in our training set. Once calculated, we can sort these distances and return the instances with the smallest calculated distances.

The below function get_KNN() will implement this idea in python.

<img src="/img/get_KNN_code_1.png">

### Step 3: Make Predictions

We have used our knowledge of Euclidean Distance to find the k nearest neighbors to our test datapoint. Now we can make predictions, the whole point of the model.

We have the most similar instances from the dataset to our test observation. Intuatively, by looking at the target outputs of our nearest neighbors, we should be able to predict an output for our test case.

**Classification**:
For a classification problem, that is as simple as counting up the instances of each output across the k nearest neighbors. Our prediction for our test datapoint will be whichever output occured most frequnetly in the nearest neighbors. 

The function below utilizes the output from the get_KNN() function to implement the idea of classification prediction in python:

<img src="/img/class_code.png">

**Regression**
For a regression problem, we use the same logic of looking at the output values of the K nearest neighbors. Instead of returning the most common occurance, we will return the mean value of the output values as the regression prediction.
The function below utilizes the output from the get_KNN() function to make a regression prediction in python.

<img src="/img/reg_code.png">

The two prediction functions created above are for making a prediction for one new data point. That was primarily for ease of understanding. Generally, we are not looking for a single prediction, but a prediction for each point in a large dataset. To adapt the above functions to handle multiple predictions, just iterate through your new dataset, calling the predict function on each point

The code below will accomplish that for classification.

<img src="/img/multiple_class_code.png">

The above can be similarly modified to handle regression predictions.

### Step 3B: Determine Accuracy of Predictions
We may have predictions but what use are they if we do not know how accurate they are? 

**Classification error metric**:
For classification we will use accuracy to determine the strength of our predictive model. 

This can simply be calculated by counting the number of correct predictions the model made divided by the amount of predictions it made.

**accuracy = correct_predictions / total_predictions**

In python this can be implemented as follows:

<img src="/img/model_accuracy.png">

**Regression error metric**: For regression there are many appropriate error metrics to evaluate your model's ability. Mean squared error, Root mean squared error, mean absolute error, and $R^2$ are a few ooptions. Explaining them all is outside the scope of this article but I suggest you spend some time learning the pros and cons for each one. For our example, we will use mean squared error. 

Calculating the MSE is the average of the squared differences between the actual output and the predicted output. 

Mathmatically, this formula can be written as

MSE = <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{n} \sum(actual - predicted)^2">

In python, we can implement an MSE calculation as follows:

<img src="/img/model_mse.png">

## Put the pieces together in a KNN class


Testing code blocks:

```python
# Return the mean square error by comparing the predicted outputs with the known outputs

def model_mse(predicted, actual):
    # Start a mse variable at 0
    mse = 0

    # For each predicted value - square the difference bewteen the actual and predicted output
    # Sum them all and divide by the number of predicted outputs
    for i in range(len(predicted)):
      mse += (actual[i] - predicted[i])**2
      mse = mse / len(predicted)

    # Return the calculated mean square root
    return mse
```
without python highlight

```
# Return the mean square error by comparing the predicted outputs with the known outputs

def model_mse(predicted, actual):
    # Start a mse variable at 0
    mse = 0

    # For each predicted value - square the difference bewteen the actual and predicted output
    # Sum them all and divide by the number of predicted outputs
    for i in range(len(predicted)):
      mse += (actual[i] - predicted[i])**2
      mse = mse / len(predicted)

    # Return the calculated mean square root
    return mse
```
