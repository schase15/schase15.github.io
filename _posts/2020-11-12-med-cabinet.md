---
layout: post
title: Med Cabinet; Find Your Strain of Cannabis!
---

## A Web Application to Pair Users with the Best Cannabis Strain for Their Needs
  
*By Steven Chase*

<img src="/img/mj_map.jpg">

While Americans may be as politically divided as ever, there is one thing people seem to agree on at the polls; and that is marijuana. During the 2020 election cycle 5 states included legalizing marijuana measures on their ballots and all 5 states passed them with significant support. Currently, 15 states and the District of Columbia have legalized recreational marijuana use while 21 states have legalized its medical use. That leaves only 14 states where marijuana use is still illegal. Many of those states, such as Texas, have crafted bills that aim to legalize marijuana in the upcoming years. Additionally, American marijuana businesses have been projected to contribute $130 billion on an annual basis to the U.S economy by 2024. As the marijuana industry steps out of the shadows and into the spotlight, it will attract millions of new users and billions of dollars in research. The result will be a confluence of a mostly uneducated new user base with the development of a plethora of new strains of cannabis. 

In order to help the public stay up to date and educate themselves as to their options and to obtain the best strains of marijuana for their needs, I set out to create an application that will recommend cannabis strains to users based on their desired effects. The result is Med Cabinet. Med Cabinet allows users to either chose preferences from a series of drop-down menus or type out a description of what they are looking for in a text box. Either way, they will be returned five profiles including the name, average ranking, strain type, effects and flavor of the strains best matching their preferences. 

To use the application for yourself, visit my [website]( https://sc-med-cabinet.herokuapp.com/)! 

### How it works:
(*View the complete notebook [here]( https://github.com/schase15/Med_Cabinet/blob/main/notebooks/Med_Cabinet_Final.ipynb).*)

To accomplish this, data was sourced from Kaggle to form a PostgreSQl database that holds information on over 2,000 strains. After cleaning the data, NLP techniques were used to create tokens and vectors to represent the words. Treating the data this way allows us to train a KNearestNeighbors (KNN) model to be able to return the five closest instances to the input from the database.
```python
# Define and fit model
nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nn.fit(dtm)
```
Both of these models were then ‘pickled’ using Sklearn’s Joblib package so that they could be inserted into the Flask app that powers this application.
```python
# Create pickles of the model and the transformer for web deployment
import pickle
from sklearn.externals import joblib 

# Save the model as a pickle file 
joblib.dump(nn, 'nn02_model.pkl') 

# Save the transformer as a pickle file
joblib.dump(tf, 'tf_01.pkl')
```

### Deployment:
In order to allow users to access Med Cabinet it needs to be deployed to a website. This was accomplished by using a Flask app and hosting the deployment on Heroku. 

**Flask App:** 
Flask allows you to develop your own API in Python even if you are not a web engineer. This is a crucial skill to have for any data scientist or machine learning engineer as it allows others, whether clients or other members on a cross-functional team, to access the hard work you have been doing. *A detailed explanation of exactly how to implement a Flask app is outside the scope of this article but there are plenty of useful resources already written.* For Med Cabinet, I was able to leverage Flask to handle both the front-end user interface as well as the data science backend. I used HTML to create a landing page where users can choose to visit either the drop-down menu or text input page. Their input from either of these pages will then be routed through the two pickled models we created to vectorize their preferences and run KNN to pull the five closest strains from our database. The information on these five strains will be displayed back to the user on the results page which again uses HTML to neatly create profiles of each strain. The results will educate the user as to the best strains for their needs hopefully leading to a better and more beneficial marijuana experience!

**Heroku:**
Flask app packages our work so that users can successfully navigate our website, however it does not actually put it on the internet. For that we turn to Heroku. Heroku offers a free platform for users to host their projects online. As a best practice, Med Cabinet was created in a GitHub repo so hosting it on Heroku was as simple as creating a project on Heroku and pushing the GitHub code to it. After adding a couple config variables to allow Heroku to access our PostgreSQL database, Med Cabinet was [live]( https://sc-med-cabinet.herokuapp.com/)! *Again, a full discussion of implementing Heroku is outside the scope of this article.* 

As the marijuana industry expands its reach both in users and applications, it will be crucial for consumers to stay on the cutting edge of marijuana advancements. Med Cabinet will do the research for them, matching users with the best stains to have the positive impact they are looking for. 

##### For access to the code base, please visit my [Github]( https://github.com/schase15/Med_Cabinet)
