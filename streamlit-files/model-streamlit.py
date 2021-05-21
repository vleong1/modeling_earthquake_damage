from os import lseek
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# load in data
train_values = pd.read_csv('data/Proj5_train_values.csv')
train_labels = pd.read_csv('data/Proj5_train_labels.csv')

# Label Encode categorical features

le = LabelEncoder()
train_enc = train_values.apply(le.fit_transform)
train_enc

# merge train values + labels

earthquake_encoded = pd.merge(train_enc, train_labels, on = 'building_id')

# set X + y
X = earthquake_encoded[['age', 'count_families', 'foundation_type', 'roof_type', 'has_superstructure_mud_mortar_stone']]
y = earthquake_encoded['damage_grade']

# tts
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 123)

# model

# scale
sscaler = StandardScaler()
X_train_scaled = sscaler.fit_transform(X_train)
X_test_scaled = sscaler.transform(X_test)

# estimator
forest = RandomForestClassifier(max_depth = 4, max_features = 2)
forest.fit(X_train_scaled, y_train)

# save model
with open('saved-earthquake-model.pkl', 'wb') as model_file:
    pickle.dump(forest, model_file)