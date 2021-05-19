from os import lseek
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
import pickle

# load in data
train_values = pd.read_csv('Proj5_train_values.csv')
train_labels = pd.read_csv('Proj5_train_labels.csv')

# merge train values + labels

earthquake_encoded = pd.merge(train_values, train_labels, on = 'building_id')


# set X + y
X = earthquake_encoded.drop(columns = ['building_id', 'damage_grade'])
y = earthquake_encoded['damage_grade']

# tts
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 123)

# model

pipe_forest = make_pipeline(OneHotEncoder(handle_unknown='ignore'), StandardScaler(with_mean=False), RandomForestClassifier(n_jobs = -1, max_depth = 11, max_features = 30))
pipe_forest.fit(X_train, y_train)

# ohe
# ohe = OneHotEncoder(use_cat_names = True)
# X_train_encoded = ohe.fit_transform(X_train)
# X_test_encoded = ohe.transform(X_test)

# # scale
# sscaler = StandardScaler()
# X_train_scaled = sscaler.fit_transform(X_train_encoded)
# X_test_scaled = sscaler.transform(X_test_encoded)

# # estimator
# forest = RandomForestClassifier(max_depth = 11, max_features = 30)
# forest.fit(X_train_scaled, y_train)

# save model
with open('saved-earthquake-model.pkl', 'wb') as model_file:
    pickle.dump(pipe_forest, model_file)