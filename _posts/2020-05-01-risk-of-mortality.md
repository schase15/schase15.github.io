---
layout: post
title: Modeling to Predict Patient's Risk of Mortality
---

## A Model to Help Medical Practitioners Predict a Patient's Risk of Mortality
#### *For more effective triaging of patients*
  
*By Steven Chase*

<img src="/img/hospital-room.jpg">

While working in a hospital, there is always a lot of pressure to make prudent decisions. Afterall, the decision you make could be the difference between life and death for your patient. Ideally each patient would receive the full resources of the hospital immediately upon being admitted. However, realistically this is not the case. Hospitals have limited resources and doctors can only be at one place at a time. This requires that a level of urgency is assigned so that more patients can be successfully treated. Those with critical conditions must be seen and treated immediately, while those who can afford to, must wait. In common practice this ranking of urgency is called triage. 

So how do you even begin to triage your patients? Armed with a 2.3 million row dataset sourced from the New York State Health Department, I set out to create an automated model to help practitioners quickly and effectively triage their incoming patients. Utilizing patient information that is available early in the admittance process, my model will return a calculated level of risk of mortality (minor, moderate, major or extreme). Being able to quickly determine their patient's risk of mortality will allow practitioners to more effectively allocate both the hospital's resources, as well as their own.

*If you would rather explore the application itself, feel free to visit the website [now](https://risk-of-mortality.herokuapp.com/)!*
*Otherwise, continue reading to learn about the process that went into creating the model.*

### Feature Selection:

The hospital inpatient dataset I used had 34 columns and 2.3 million rows. In my final model I used only 7 of these as features. Through exploratory data analysis, I eliminated unique identifiers and redundant information. For example, for each medical code there was an accompanying description. While useful for interpreting results, this was redundant information. Additionally, ‘CCS Diagnosis Code’ and ‘APR MDC Code’ contained information already captured by ‘CCS Procedure Code’ and ‘APR DRG Code’ respectfully, so they were also removed as being redundant. Considering that I wanted this model to be implemented early in the treatment process, I also eliminated columns such as ‘Length of Stay’ and ‘Total Costs’ as this data is not available until the patient has already completed treatment. I also eliminated the column ‘Severity of Illness’ as this had the potential for data leakage. Data leakage is when a feature you include in your model is directly related to the target. This allows the model to ‘cheat’ by learning about the target from this feature. In this case, the severity of the illness is directly related to their risk of mortality; as the severity of illness increases, so does the patients risk of mortality.

After cleaning the data, I ran a model using the remaining features and viewed the permutation feature importance. In short, permutation feature importance shows you the weight that a specific feature has on the predictive ability of the model. Features with higher weights are more helpful in making accurate predictions, while those with lower weights are less effective. Features can even have negative weights, meaning they are hurting the predictive capabilities of the model.

<img src="/img/all_permutations.png">

Based on the above permutation feature importance, I decided to use the following 7 features:
-	Age Group
-	Type of Admission
-	APR DRG Code
-	CCS Procedure Code
-	APR Medical Surgical Description
-	Payment Typology 1
-	Emergency Department Indicator

### Evaluation Metrics:

As this is a classification model, I chose two metrics to evaluate its performance: accuracy and weighted average f1 score.
- Accuracy tells you, out of the total number of predictions, how many of them were predicted correctly.
- The f1 score evaluates the level of precision and recall for your model. Precision tells you, out of the total predictions for a specific class, how many of them were correct. Recall tells you, out of the total number of actual values for a specific target class, how many were correctly identified as such. Since this is a multi-class target, we use the average to evaluate how the model works across all categories. Additionally, because the target class is right skewed, the weighted score is used as it helps account for label imbalance.

### Comparing Models:





Visit the [application](https://risk-of-mortality.herokuapp.com/) to get predictions of your patients' risk of mortality.

*For more detailed information about the process, evaluation and insights of the model, please explore the cooresponding pages on the Dash web app.*

##### For access to the raw code (which includes a link to the raw data) please visit my [GitHub](https://github.com/schase15/risk_of_mortality)
