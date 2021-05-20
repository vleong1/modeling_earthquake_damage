# modeling_earthquake_damage

## Problem Statement

**Using data modeling, is it possible to predict the level of damage sustained by a building in the 2015 Gorkha earthquake in Nepal using descriptive features of the building iteslf?**


## Executive Summary

## Table of Contents
- Background
- Data Collection
- Data Inspection/Cleaning
- NLP/Feature Engineering
- Imported Libraries
- Data Compilation
- Data Dictionary
- Data Modeling
- Analysis
- Conclusion

## Background
On April 25, 2015, an earthquake measuring 7.8 on the Richter scale struck a region of the Asian nation of Nepal less than 50 miles northwest of the capital Kathmandu. Hundreds of aftershocks followed in the ensuing weeks, including a 7.3 magnitude quake 18 days later and two others measuring 6.5 and 6.6 on the Richter scale. About 8,900 people were killed and some 22,000 others injured by the temblor, which also damaged or destroyed over a million homes. Following the disaster, Nepal completed out a comprehensive household survey to assess building damage in the quake-affected districts. The main goal was to identify beneficiaries eligible for government housing reconstruction assistance. [link]('https://www.mercycorps.org/blog/quick-facts-nepal-earthquake#:~:text=Strength%3A%207.8%20on%20the%20Richter,strength%20and%20the%20resulting%20damage')

Earlier this year, the data science competition site DrivenData.org opened a contest to the public in an effort to discover the best classification model for predicting building damage in the Nepal earthquake. As of this project date, over 4,000 entrants had submitted models to the website.

### Data Collection

### Data Inspection/Cleaning

### Imported Libraries
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
```

### Data Compilation 

Comprehensive datasets:
- [Proj5_submission_format](/Proj5_submission_format.csv)

- [Proj5_test_values](/Proj5_test_values.csv))

- [Proj5_train_labels](/Proj5_train_labels.csv)

- [Proj5_Proj5_train_values](/Proj5_train_values.csv)

- [Proj5_train_10pct](/train_10pct.csv)

- [Proj5_train_10pct_encoded](/train_10pct_encoded)

- [Proj5_train_10pct_labels](/train_10pct_labels.csv)

### Data Dictionary
[link]('https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/')

<<<<<<< HEAD
geo_level_1_id, geo_level_2_id, geo_level_3_id (type: int): geographic region in which building exists, from largest (level 1) to most specific sub-region (level 3). Possible values: level 1: 0-30, level 2: 0-1427, level 3: 0-12567.
count_floors_pre_eq (type: int): number of floors in the building before the earthquake.
age (type: int): age of the building in years.
area_percentage (type: int): normalized area of the building footprint.
height_percentage (type: int): normalized height of the building footprint.
land_surface_condition (type: categorical): surface condition of the land where the building was built. Possible values: n, o, t.
foundation_type (type: categorical): type of foundation used while building. Possible values: h, i, r, u, w.
roof_type (type: categorical): type of roof used while building. Possible values: n, q, x.
ground_floor_type (type: categorical): type of the ground floor. Possible values: f, m, v, x, z.
other_floor_type (type: categorical): type of constructions used in higher than the ground floors (except of roof). Possible values: j, q, s, x.
position (type: categorical): position of the building. Possible values: j, o, s, t.
plan_configuration (type: categorical): building plan configuration. Possible values: a, c, d, f, m, n, o, q, s, u.
has_superstructure_adobe_mud (type: binary): flag variable that indicates if the superstructure was made of Adobe/Mud.
has_superstructure_mud_mortar_stone (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Stone.
has_superstructure_stone_flag (type: binary): flag variable that indicates if the superstructure was made of Stone.
has_superstructure_cement_mortar_stone (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Stone.
has_superstructure_mud_mortar_brick (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Brick.
has_superstructure_cement_mortar_brick (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Brick.
has_superstructure_timber (type: binary): flag variable that indicates if the superstructure was made of Timber.
has_superstructure_bamboo (type: binary): flag variable that indicates if the superstructure was made of Bamboo.
has_superstructure_rc_non_engineered (type: binary): flag variable that indicates if the superstructure was made of non-engineered reinforced concrete.
has_superstructure_rc_engineered (type: binary): flag variable that indicates if the superstructure was made of engineered reinforced concrete.
has_superstructure_other (type: binary): flag variable that indicates if the superstructure was made of any other material.
legal_ownership_status (type: categorical): legal ownership status of the land where building was built. Possible values: a, r, v, w.
count_families (type: int): number of families that live in the building.
has_secondary_use (type: binary): flag variable that indicates if the building was used for any secondary purpose.
has_secondary_use_agriculture (type: binary): flag variable that indicates if the building was used for agricultural purposes.
has_secondary_use_hotel (type: binary): flag variable that indicates if the building was used as a hotel.
has_secondary_use_rental (type: binary): flag variable that indicates if the building was used for rental purposes.
has_secondary_use_institution (type: binary): flag variable that indicates if the building was used as a location of any institution.
has_secondary_use_school (type: binary): flag variable that indicates if the building was used as a school.
has_secondary_use_industry (type: binary): flag variable that indicates if the building was used for industrial purposes.
has_secondary_use_health_post (type: binary): flag variable that indicates if the building was used as a health post.
has_secondary_use_gov_office (type: binary): flag variable that indicates if the building was used fas a government office.
has_secondary_use_use_police (type: binary): flag variable that indicates if the building was used as a police station.
has_secondary_use_other (type: binary): flag variable that indicates if the building was secondarily used for other purposes.
damage_grade (type: categorical/ordinal): target variable that represents a level of damage to the building that was hit by the earthquake. Possible values: 1 (low damage), 2 (medium damage), 3 (almost complete destruction).
=======
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
|**count_families**|*categorical*|number of families that live in the building.|
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
>>>>>>> main

*NOTE: Categorical variables have been obfuscated by random lowercase letters. The appearance of the same letter in multiple distinct columns does* not *imply the same original value.*

### Data Modeling

Parameters used: {template}
+ Logistic Regression: Penalty['l1', 'l2', 'elasticnet', 'none'], C[.01,.1,1,10,100]
+ Naive Bayes: Alpha[.01,.1,1,10,100]

## Analysis

Models with higher accuracy scores were:
{template}
|Classifier|Pipeline|Accuracy|
|---|---|---|
|RandomForestClassifier|CountVectorizer(max_features=200), DecisionTreeClassifier^(max_depth=11)|58.87%|
|GradientBoostingClassifier|No pipeline|58.06%|

### Conclusions/Recommendations

