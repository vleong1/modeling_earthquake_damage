# modeling_earthquake_damage

Researchers: Chris Martin, Eric Cheng, Veronica Leong

## Problem Statement

**Is it possible to predict the level of damage sustained by a building in the 2015 Gorkha earthquake in Nepal using classification modeling against the building’s features?**


## Executive Summary

On April 25, 2015, an earthquake measuring 7.8 on the Richter scale struck a region in Nepal. As a result of this temblor and its aftershocks, about 8,900 people were killed and over a million homes were damaged or destroyed. So we decided to ask the question: could we have predicted how much damage a building sustained by examining some of the structure’s features?

Our team decided to construct several models in an attempt to find answers. We downloaded a 260,000-observation dataset from DrivenData.org, a site that is hosting a data science competition pertaining to this problem. We decided to prepare a report for an audience of data scientists, technical advisors, disaster relief specialists, and officials from global organizations. We were hopeful that any results we uncovered could have ramifications for other countries that are located in earthquake-ravaged areas of the world.

After inspecting the data for null values, we used label encoding to change the categorical variables into numeric ones for easier modeling. Then we took the first 10% of rows from the dataset to use for initial modeling.

The subsetted data was inspected and certain aspects of the data visualized. We broke down the data by top-level geographical location, superstructure, foundation type, and damage grade and also noted six slightly correlative variables. Charts and graphs were created to highlight this information.

For the modeling step, we scaled the data and selected eight different classifiers for fitting against our data. We also grid searched hyperparameters in pipelines containing these classifiers. The accuracy scores were recorded for each pipeline along with the corresponding parameters.

Surprisingly, the neural networks did not perform very well no matter how the layers and parameters were changed. Similarly, K-Nearest Neighbors and Logistic Regression models barely surpassed the baseline model; however, more favorable accuracy scores were achieved using Bagging, Adaboost, Decision Tree, and Extra Trees classifiers.

Our best-performing model was a Random Forest classifier with a maximum features parameter of 35 and a maximum depth of 11. The model posted accuracy scores of 68.9% on the training data and an impressive 70.5% on the testing dataset. Upon closer examination, geographical variables were among the most important features in our model as were building age, foundation type, roof type, number of families occupying the building, and a superstructure of mud mortar and stone. 

Our project also included the construction of an online app using the Streamlit framework. We picked out five permutation feature importances and gave app users the ability to toggle these features and determine what feature values would predict different levels of damage grade. This app was instrumental in communicating the results of our modeling to any future non-technical audiences.

We did achieve our goal of coming up with recommendations for other countries based on our modeling results. We suggested that builders phase out mud mortar and stone foundations in new construction and incorporate roof and foundation types that are less prone to serious quake damage. Moreover, older buildings and multi-family housing structures should be reinforced, and additional resources should be allocated to the regions that suffer the worst earthquake damage. 

To be sure, our experiment was hindered by certain limitations, the most obvious of which involved the categorical data. For purposes of the DrivenData competition, several features’ categories were deliberately obfuscated with letters so as to minimize any real-world information about those variables. As a result, we were restricted in our efforts to glean real-world insights from our models and transfer them into future recommendations.

Should we decide to extend our experiment further, we would likely attempt to embrace some imbalanced data approaches such as using ADASYN, SMOTE, and random oversampling to address this issue. We might also choose to integrate other boosters (such as GradientBoost and HistGradientBoost) into our pipelines that contain our base estimators to see if we can improve our models’ accuracy.

Though still an inexact science, our data science methods showed that we could predict earthquake damage on buildings to some extent based on the structures; characteristics. And until science comes up with a less error-prone way to predict earthquakes, tools like our models are probably the next best approach to minimizing the damage these quakes can cause.

## Table of Contents
- [Background](#Background)
- [Data Collection](#Data-Collection)
- [Data Inspection/Cleaning](#Data-Inspection/Cleaning)
- [Data Compilation](#Data-Compilation)
- [Data Visualization](#Data-Visualization)
- [Imported Libraries](#Imported-Libraries)
- [Data Dictionary](#Data-Dictionary)
- [Data Modeling & Analysis](#Data-Modeling-&-Analysis)
- [Conclusions and Recommendations](#Conclusions-and-Recommendations)
- [Limitations and Future Project Refinement](#Limitations-and-Future-Project-Refinement)


## Background
On April 25, 2015, an earthquake measuring 7.8 on the Richter scale struck a region of the Asian nation of Nepal less than 50 miles northwest of the capital Kathmandu. Hundreds of aftershocks followed in the ensuing weeks, including a 7.3 magnitude quake 18 days later and two others measuring 6.5 and 6.6 on the Richter scale. About 8,900 people were killed and some 22,000 others injured by the temblor, which also damaged or destroyed over a million homes. Following the disaster, Nepal completed a [comprehensive household survey] ('https://www.mercycorps.org/blog/quick-facts-nepal-earthquake#:~:text=Strength%3A%207.8%20on%20the%20Richter,strength%20and%20the%20resulting%20damage') to assess building damage in the quake-affected districts. The main goal was to identify beneficiaries eligible for government housing reconstruction assistance.

Earlier this year, the data science competition site DrivenData.org opened a contest to the public in an effort to discover the best classification model for predicting building damage in the Nepal earthquake. As of this project date, over 4,000 entrants had submitted models to the website.

### Data Collection

We began our experiment by downloading the datasets from the DrivenData.org [competition page]('https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/') containing information about hundreds of thousands of buildings that were examined in the earthquake zone. This consisted of a training set of data (with more than 260,000 observations), a test set of data (consisting of almost 89,000 observations), and a training label data (which had a target variable value associated with each observation in the training data).

The training and test datasets each contained 38 columns of data which included seven integer variables, eight categorical variables, and 22 binary variables. The target variable in the training and train label datasets was a three-value categorical variable representing the level of damage sustained by each structure.

### Data Inspection/Cleaning

After checking the datasets for null values (and finding none), we visually inspected the data tables to get a sense of the information we were preparing to model. We noticed that several categorical variables consisted of single letter values which had no relevance to any of the data points. These values were intentionally obfuscated by the competition organizers so as to reduce the amount of information provided to the entrants.

Because of these lettered values, we proceeded to label encode the values in the training and test datasets so that the data would be in numerical form, which would allow us to model these categories along with the other features. Once encoding was complete, we then subsetted the training dataset by pulling out the first ten percent of observations. This subsetted dataset was the one we used for modeling purposes in an effort to save computer runtime during the machine learning process.

### Data Compilation 

Comprehensive datasets:
- [Proj5_test_values](/Proj5_test_values.csv))

- [Proj5_train_labels](/Proj5_train_labels.csv)

- [Proj5_Proj5_train_values](/Proj5_train_values.csv)

- [Proj5_train_10pct](/train_10pct.csv)

- [Proj5_train_10pct_labels](/train_10pct_labels.csv)

### Data Visualization

Next, we engaged in some data visualization with the hopes of gaining some initial insights before we entered the modeling phase of our experiment. First, we leveraged the seaborn package to create a heatmap of all of the variables.  Unfortunately, no substantial correlative information could be gleaned from this busy illustration, so we did not include it in our presentation.

In an effort to better understand the geographic scale of the earthquake damage, we constructed a bar chart of all of the buildings separated out by top-level geographic id (geo_level_1_id). The discrete values in this column ranged from 1 to 30, and each value can best be compared to a U.S. state (with geo_level_2_id and geo_level_3_id corresponding to counties and neighborhoods, respectively). The chart revealed that the majority of the buildings affected by the earthquake were situated in just seven geographic areas.

Since there were eleven binary features which alluded to the "superstructure" of the buildings, we constructed bar charts that illustrated whether or not a given building was constructed with a particular superstructure makeup. What we discovered was that about 20,000 of the 26,000 buildings in our working dataset contained a superstructure of mud mortar and stone - which we felt was not the most ideal choice for remaining unscathed by an earthquake.

We also checked out the distribution of buildings that were classified in each of the three damage grade categories. About 56.7% of the buildings sustained what was referred to as a "medium amount of damage" from the quake. Approximately 33.6% of them were recorded as being "almost completely destroyed", while the remaining 9.6% suffered only "low damage." We used this information to come up with our baseline (null) model of 56.7% for our upcoming experiment.

Finally, we leveraged two Python data visualization packages Pandas Profiling and SweetViz, to generate a pair of formal reports which contained numerous graphs, charts, and data. In the latter report, we noted six variables which were deemed to be somewhat correlative with our target variable (which we depicted in a bar graph): 

+ Area_percentage (normalized area of the building footprint):  0.14
+ Geo_level_1_id (top level geographic designator):  0.13
+ Height_percentage (normalized height of the building footprint):  0.05
+ Geo_level_2_id  (mid level geographic designator): 0.05
+ Age of structure:  0.04
+ Geo_level_3_id  (lowest level geographic designator): 0.02

### Imported Libraries
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

### Data Dictionary
[link]('https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/')

|Feature|Data Type|Description|
|---|---|---|
|**geo_level_1_id, geo_level_2_id, geo_level_3_id**|*integer*|geographic region in which building exists, from largest (level 1) to most specific sub-region (level 3). Possible values: level 1: 0-30, level 2: 0-1427, level 3: 0-12567.|
|**count_floors_pre_eq**|*integer*|number of floors in the building before the earthquake.|
|**age**|*integer*|age of the building in years.|
|**area_percentage**|*integer*|normalized area of the building footprint.|
|**height_percentage**|*integer*|normalized height of the building footprint.|
|**land_surface_condition**|*categorical*|surface condition of the land where the building was built. Possible values: n, o, t.|
|**foundation_type**|*categorical*|type of foundation used while building. Possible values: h, i, r, u, w.|
|**roof_type**|*categorical*|type of roof used while building. Possible values: n, q, x.|
|**ground_floor_type**|*categorical*|type of the ground floor. Possible values: f, m, v, x, z.|
|**other_floor_type**|*categorical*|type of constructions used in higher than the ground floors (except of roof). Possible values: j, q, s, x.|
|**position**|*categorical*|position of the building. Possible values: j, o, s, t.|
|**plan_configuration**|*categorical*|building plan configuration. Possible values: a, c, d, f, m, n, o, q, s, u.|
|**has_superstructure_adobe_mud**|*binary*|flag variable that indicates if the superstructure was made of Adobe/Mud.|
|**has_superstructure_mud_mortar_stone**|*binary*|flag variable that indicates if the superstructure was made of Mud Mortar - Stone.|
|**has_superstructure_stone_flag**|*binary*|flag variable that indicates if the superstructure was made of Stone.|
|**has_superstructure_cement_mortar_stone**|*binary*|flag variable that indicates if the superstructure was made of Cement Mortar - Stone.|
|**has_superstructure_mud_mortar_brick**|*binary*|flag variable that indicates if the superstructure was made of Mud Mortar - Brick.|
|**has_superstructure_cement_mortar_brick**|*binary*|flag variable that indicates if the superstructure was made of Cement Mortar - Brick.|
|**has_superstructure_timber**|*binary*|flag variable that indicates if the superstructure was made of Timber.|
|**has_superstructure_bamboo**|*binary*|flag variable that indicates if the superstructure was made of Bamboo.|
|**has_superstructure_rc_non_engineered**|*binary*|flag variable that indicates if the superstructure was made of non-engineered reinforced concrete.|
|**has_superstructure_rc_engineered**|*binary*|flag variable that indicates if the superstructure was made of engineered reinforced concrete.|
|**has_superstructure_other**|*binary*|flag variable that indicates if the superstructure was made of any other material.|
|**legal_ownership_status**|*categorical*|legal ownership status of the land where building was built. Possible values: a, r, v, w.|
|**count_families**|*integer*|number of families that live in the building.|
|**has_secondary_use**|*binary*|flag variable that indicates if the building was used for any secondary purpose.|
|**has_secondary_use_agriculture**|*binary*|flag variable that indicates if the building was used for agricultural purposes.|
|**has_secondary_use_hotel**|*binary*|flag variable that indicates if the building was used as a hotel.|
|**has_secondary_use_rental**|*binary*|flag variable that indicates if the building was used for rental purposes.|
|**has_secondary_use_institution**|*binary*|flag variable that indicates if the building was used as a location of any institution.|
|**has_secondary_use_school**|*binary*|flag variable that indicates if the building was used as a school.|
|**has_secondary_use_industry**|*binary*|flag variable that indicates if the building was used for industrial purposes.|
|**has_secondary_use_health_post**|*binary*|flag variable that indicates if the building was used as a health post.|
|**has_secondary_use_gov_office**|*binary*|flag variable that indicates if the building was used as a government office.|
|**has_secondary_use_use_police**|*binary*|flag variable that indicates if the building was used as a police station.|
|**has_secondary_use_other**|*binary*|flag variable that indicates if the building was secondarily used for other purposes.|
|**damage_grade**|*categorical/ordinal*|target variable that represents a level of damage to the building that was hit by the earthquake. Possible values: 1 (low damage), 2 (medium damage), 3 (almost complete destruction).|

*NOTE: Categorical variables have been obfuscated by random lowercase letters. The appearance of the same letter in multiple distinct columns does* not *imply the same original value.*

### Data Modeling & Analysis

For the modeling process, we selected eight different estimators that are designed to address classification problems of this nature. For each model, we utilized StandardScaler, as well as the estimator within a Pipeline. We also noted a randon_state to keep the reproduction of the code and results consistent. Afterwards, we GridSearched over several different hyperparameters and noted the best parameters and best accuracy scores (after different models tested) for each model below:

|Model|Best Parameters|Accuracy|
|---|---|---|
|**Baseline Accuracy Score**|---|**56.7%**|
|Logistic Regression|C = 10, solver = ‘lbfgs’|58.3%|
|KNeighborsClassifier|n_neighbors = 11|57.1%|
|RandomForestClassifier|max_depth = 11, max_features = 35|68.9%|
|ExtraTreesClassifier|max_depth = 11, max_features = 35|67.9%|
|DecisionTreeClassifier|max_depth = 7, min_sample_split = 4|66.5%|
|BaggingClassifier|max_features = 9|66.8%|
|AdaboostClassifier|learning_rate = 1.3, n_estimators = 80|66.0%|
|Neural Network|Dense layers = 3, with nodes of 12, 80, and 30; Dropout layers = 2, both with a value of 0.3; Epochs = 50; Early Stopping: yes, at 45 epochs|56.8%|

### Streamlit App

For demo purposes, we build a Streamlit app based on our best model estimator (RandomForestClassifier) and top 5 permutation feature importances ('age', 'count_families', 'foundation_type', 'roof_type', 'has_superstructure_mud_mortar_stone') so that users of the app would be able to toggle these features and determine what feature values would predict different levels of damage grade. Since we're only using 5 features, we had to re-GridSearchCV to find the best hyperparameters for the subset of features. After grid searching, we utilized the best parameters (max_depth = 4, max_features = 2) within the RandomForestClassifier for Streamlit. Against the test data, the model score continued to perform slightly better than the baseline accuracy score (0.5695 vs. 0.5693), so it was still ok to utilize for demo purposes. This app provides value if a user wants to know what feature value combinations predict a high damage grade, or what feature value combinations predict a low damage grade to take into consideration when constructing a building.

### Conclusions/Recommendations

Though the original dataset was lacking in specific information (due to the ongoing competition on DrivenData.org), we did manage to reach some conclusions from the results of our modeling. First, we were somewhat surprised at the lack of performance we received from our neural networks and our K-Nearest Neighbors models, some of which didn't even surpass our baseline model. In contrast, our other classifiers - namely our boosting, extra trees, and decision tree models - did perform significantly better, posting accuracy scores of above 66%.

But it was our random forest classification model which posted the most impressive accuracy score (above 70% on the test data) in our modeling efforts. This model revealed that the damage grades assigned to these earthquake-ravaged structures were most related to foundation type, building age, roof type, the family occupancy count, and the presence of a mud mortar/stone superstructure.

Given these results, we feel confident enough to make the following recommendations to our audience:

1. Encourage builders to incorporate the materials of the roofs and foundation types that our models show to be less prone to serious damage.
2. Phase out or eliminate the use of mud mortar and stone superstructures in new construction because their presence is positively correlated with the amount of damage sustained in the 2015 earthquake.
3. Since the building’s age has a positive correlation with more serious damage, efforts should be undertaken to reinforce older buildings against future quakes.
4. A similar initiative is recommended for multi-family housing, since our models indicate these structures are more likely to be destroyed by an earthquake than are single-family homes.
5. Identify the major geographic areas which sustained the most serious damage and focus on allocating more earthquake damage prevention resources to those locations.

### Limitations and Future Project Refinements

It's also important to note the limitations that are present in our modeling results and experiment conclusions due to the nature of the data we were provided. The most obvious shortcoming was the obfuscation of key information relating to geographic location, land surface condition, foundation type, roof type, description of ground and upper floors, structure position, building plan configuration, and legal ownership status. To be sure, we could have provided more targeted and specific recommendations to reduce future quake damage had we been privy to these details.

As for our modeling, we have considered other tactics which could be employed should we choose to continue our research into this project. Since the original dataset consisted of significantly unbalanced data, the use of imbalanced learning strategies such as SMOTE, ADASYN, and random oversampling might help us uncover trends or patterns that weren't apparent to us in our modeling. Furthermore, we could incorporate additional boosting strategies (such as AdaBoost, gradient boosting, or histogram-based gradient boosting) into our base models to see if aggregating these tools produced more favorable accuracy scores.

Until we discover methods or models to accurately predict future earthquakes, our strongest tools to reduce the physical destruction caused by these temblors can be found in structural damage mitigation strategies. And the more accurate our models can be in predicting potential quake damage, the better off that the world's buildings - and the people who work and live in them - will be.

May 21, 2021