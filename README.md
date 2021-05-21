# modeling_earthquake_damage

Researchers: Chris Martin, Eric Cheng, Veronica Leong

## Problem Statement

**Is it possible to predict the level of damage sustained by a building in the 2015 Gorkha earthquake in Nepal using classification modeling against the building’s features?**


## Executive Summary

## Table of Contents
- Background
- Data Collection
- Data Inspection/Cleaning
- NLP/Feature Engineering
- Data Compilation
- Data Visualization
- Imported Libraries
- Data Dictionary
- Data Modeling
- Analysis
- Conclusions and Recommendations
- Limitations and Future Project Refinements

## Background
On April 25, 2015, an earthquake measuring 7.8 on the Richter scale struck a region of the Asian nation of Nepal less than 50 miles northwest of the capital Kathmandu. Hundreds of aftershocks followed in the ensuing weeks, including a 7.3 magnitude quake 18 days later and two others measuring 6.5 and 6.6 on the Richter scale. About 8,900 people were killed and some 22,000 others injured by the temblor, which also damaged or destroyed over a million homes. Following the disaster, Nepal completed out a comprehensive household survey to assess building damage in the quake-affected districts. The main goal was to identify beneficiaries eligible for government housing reconstruction assistance. [link]('https://www.mercycorps.org/blog/quick-facts-nepal-earthquake#:~:text=Strength%3A%207.8%20on%20the%20Richter,strength%20and%20the%20resulting%20damage')

Earlier this year, the data science competition site DrivenData.org opened a contest to the public in an effort to discover the best classification model for predicting building damage in the Nepal earthquake. As of this project date, over 4,000 entrants had submitted models to the website.

### Data Collection

We began our experiment by downloading the datasets from the DrivenData.org [competition page]('https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/') containing information about hundreds of thousands of buildings that were examined in the earthquake zone. This consisted of a training set of data (with more than 260,000 observations), a test set of data (consisting of almost 89,000 observations), and a training label data (which had a target variable value associated with each observation in the training data).

The training and test datasets each contained 38 columns of data which included seven integer variables, eight categorical variables, and 22 binary variables. The target variable in the training and train label datasets was a three-value categorical variable representing the level of damage sustained by each structure.

### Data Inspection/Cleaning

After checking the datasets for null values (and finding none), we visually inspected the data tables to get a sense of the information we were preparing to model. We noticed that several categorical variables consisted of single letter values which had no relevance to any of the data points. These values were intentionally obfuscated by the competition organizers so as to reduce the amount of information provided to the entrants.

Because of these lettered values, we proceeded to label encode the values in the training and test datasets so that the data would be in numerical form, which would allow us to model these categories along with the other features. Once encoding was complete, we then subsetted the training dataset by pulling out the fisrt ten percent of observations. This subsetted dataset was the one we used for modeling purposes in an effort to save computer runtime during the machine learning process.

### Data Compilation 

Comprehensive datasets:
- [Proj5_test_values](/Proj5_test_values.csv))

- [Proj5_train_labels](/Proj5_train_labels.csv)

- [Proj5_Proj5_train_values](/Proj5_train_values.csv)

- [Proj5_train_10pct](/train_10pct.csv)

- [Proj5_train_10pct_labels](/train_10pct_labels.csv)

### Data Visualization

Next, we engaged in some data visualization with the hopes of gaining some initial insights before we entered the modeling phase of our experiment. First, we leveraged the seaborn package to create a heatmap of all of the variables.  Unfortunately, no substantial correlative information could be gleaned from this busy illustration, so we did not include it in our presentation.

In an effort to better understand the geographic scale of the earthquake damage, we constructed a bar chart of all of the buildings separated out by top-level geographic id (geo_level_1_id). The discrete values in this column ranged from 1 to 30, and each value can best be compared to a U.S. state (with geo_level_2_id and geo_level_3_id corresponding to counties and neighborhoods, respectively). The chart revealed that the majority of the  buildings affected by the earthquake were situated in just seven geographic areas.

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
|**has_secondary_use_gov_office**|*binary*|flag variable that indicates if the building was used fas a government office.|
|**has_secondary_use_use_police**|*binary*|flag variable that indicates if the building was used as a police station.|
|**has_secondary_use_other**|*binary*|flag variable that indicates if the building was secondarily used for other purposes.|
|**damage_grade**|*categorical/ordinal*|target variable that represents a level of damage to the building that was hit by the earthquake. Possible values: 1 (low damage), 2 (medium damage), 3 (almost complete destruction).|

*NOTE: Categorical variables have been obfuscated by random lowercase letters. The appearance of the same letter in multiple distinct columns does* not *imply the same original value.*

### Data Modeling


Parameters used: {template}
+ Logistic Regression: Penalty['l1', 'l2', 'elasticnet', 'none'], C[.01,.1,1,10,100]
+ Naive Bayes: Alpha[.01,.1,1,10,100]

## Analysis

Models with higher accuracy scores were:

|Classifier|Pipeline|Accuracy|
|---|---|---|
|**Baseline Accuracy Score**|---|**56.74%**|
|RandomForestClassifier|LabelEncoding(), StandardScaler()|68.87%|
|ExtraTreesClassifier|LabelEncoding(), StandardScaler()|66.43%|
|GradientBoostingClassifier|No pipeline|58.06%|

### Conclusions/Recommendations

Though the original dataset was lacking in specific information (due to the ongoing competition on DrivenData.org), we did manage to reach some conclusions from the results of our modeling. First, we were somewhat surprised at the lack of performance we received from our neural networks and our K-Nearest Neighbors models, some of which didn't even surpass our baseline model. In contrast, our other classifiers - namely our boosting, extra trees, and decision tree models - did perform significantly better, posting accuracy scores of above 66%.

But it was our random forest classification model which posted the most impressive accuracy score (above 70% on the test data) in our modeling efforts. This model revealed that the damage grade assigned to these earthquake-ravaged structures were most related to foundation type, building age, roof type, the family occupancy count, and the presence of a mud mortar/stone superstructure.

Given these results, we feel confident enough to make the follolwing recommendations to our audience:

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


