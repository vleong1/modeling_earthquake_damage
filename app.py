from numpy.core.fromnumeric import reshape
import streamlit as st
import pickle
import numpy as np

st.title("Modeling Earthquake Damage")
st.header("How much damage did the building incur?")

# load model

with open('saved-earthquake-model.pkl', 'rb') as file:
    model = pickle.load(file)

# geo_level
geo_level_1_id = np.random.randint(0, 30)
geo_level_2_id = np.random.randint(0, 1427)
geo_level_3_id = np.random.randint(0, 12567)

# count_floors_pre_eq
count_floors = st.slider('# of Floors Pre-Earthquake', min_value = 1, max_value = 9, step = 1)

# age_of_building
age = np.random.randint(0, 995)

# area_percentage
area_percentage = np.random.randint(1, 100)

# height percentage
height_percentage = np.random.randint(2, 32)

# foundation_ type
foundation_type = np.random.choice(['h', 'i', 'r', 'u', 'w'])

# land_surface_condition
land_surface_condition = np.random.choice(['n', 'o', 't'])

# roof_type
roof_type = np.random.choice(['n', 'q', 'x'])

# ground_floor_type
ground_floor_type = np.random.choice(['f', 'm', 'v', 'x', 'z'])

# other_floor_type
other_floor_type = np.random.choice(['j', 'q', 's', 'x'])

# position
position = np.random.choice(['j', 'o', 's', 't'])

# plan_configuration
plan_configuration = np.random.choice(['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'])

# has_superstructure_adobe_mud
has_superstructure_adobe_mud = np.random.choice([0,1])

# has_superstructure_mud_mortar_stone 
has_superstructure_mud_mortar_stone = np.random.choice([0,1])

# has_superstructure_stone_flag 
has_superstructure_stone_flag = np.random.choice([0,1])

# has_superstructure_cement_mortar_stone 
has_superstructure_cement_mortar_stone = np.random.choice([0,1])

# has_superstructure_mud_mortar_brick 
has_superstructure_mud_mortar_brick = np.random.choice([0,1])

# has_superstructure_cement_mortar_brick 
has_superstructure_cement_mortar_brick = np.random.choice([0,1])

# has_superstructure_timber 
has_superstructure_timber = np.random.choice([0,1])

# has_superstructure_bamboo 
has_superstructure_bamboo = np.random.choice([0,1])

# has_superstructure_rc_non_engineered 
has_superstructure_rc_non_engineered = np.random.choice([0,1])

# has_superstructure_rc_engineered 
has_superstructure_rc_engineered = np.random.choice([0,1])

# has_superstructure_other 
has_superstructure_other = np.random.choice([0,1])

# legal_ownership_status 
legal_ownership_status = np.random.choice(['a', 'r', 'v', 'w'])

# count_families
count_families = st.slider('# of Families that Live in the building', min_value = 0, max_value = 6, step = 1)

# has_secondary_use 
has_secondary_use = np.random.choice([0,1])

# has_secondary_use_agriculture 
has_secondary_use_agriculture = np.random.choice([0,1])

# has_secondary_use_hotel 
has_secondary_use_hotel = np.random.choice([0,1])

# has_secondary_use_rental 
has_secondary_use_rental = np.random.choice([0,1])

# has_secondary_use_institution 
has_secondary_use_institution = np.random.choice([0,1])

# has_secondary_use_school 
has_secondary_use_school = np.random.choice([0,1])

# has_secondary_use_industry 
has_secondary_use_industry = np.random.choice([0,1])

# has_secondary_use_health_post 
has_secondary_use_health_post = np.random.choice([0,1])

# has_secondary_use_gov_office 
has_secondary_use_gov_office = np.random.choice([0,1])

# has_secondary_use_use_police 
has_secondary_use_use_police = np.random.choice([0,1])

# has_secondary_use_other 
has_secondary_use_other = np.random.choice([0,1])

all_features = np.array([geo_level_1_id, geo_level_2_id, geo_level_3_id, count_floors, age, area_percentage, height_percentage, foundation_type, land_surface_condition, roof_type, ground_floor_type, 
other_floor_type, position, plan_configuration, has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone, has_superstructure_stone_flag, has_superstructure_cement_mortar_stone,
has_superstructure_mud_mortar_brick, has_superstructure_cement_mortar_brick, has_superstructure_timber, has_superstructure_bamboo, has_superstructure_rc_engineered, has_superstructure_rc_non_engineered,
has_superstructure_other, legal_ownership_status, count_families, has_secondary_use, has_secondary_use_agriculture, has_secondary_use_hotel, has_secondary_use_rental, has_secondary_use_institution, 
has_secondary_use_school, has_secondary_use_industry, has_secondary_use_health_post, has_secondary_use_gov_office, has_secondary_use_use_police, has_secondary_use_other])

#model.predict()
prediction = model.predict(all_features.reshape(1, -1))

st.header(f'The model predicts: {prediction[0]}')