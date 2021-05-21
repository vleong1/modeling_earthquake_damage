from numpy.core.fromnumeric import reshape
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Modeling Earthquake Damage")
st.header("How much damage did the building incur?")
st.markdown("----------------------------------------")

# load model

with open('saved-earthquake-model.pkl', 'rb') as file:
    model = pickle.load(file)


# user input features

# count_floors_pre_eq
st.markdown("Number of Floors Pre-Earthquake:")
count_floors = st.slider('', min_value = 1, max_value = 9, step = 1)

# count_families
st.markdown("Number of Families that Live in the building:")
count_families = st.slider('', min_value = 0, max_value = 6, step = 1)

# foundation_ type
st.markdown("Type of foundation used while building:")
foundation_type_choice = st.radio('', ['H', 'I', 'R', 'U', 'W'])

if foundation_type_choice == 'H':
    foundation_type = 0
elif foundation_type_choice == 'I':
    foundation_type = 1
elif foundation_type_choice == 'R':
    foundation_type = 2
elif foundation_type_choice == 'U':
    foundation_type = 3
else:
    foundation_type = 4

# land_surface_condition
st.markdown("Surface condition of the land where the building was built:")
land_surface_condition_choice = st.radio('', ['N', 'O', 'T'])

if land_surface_condition_choice == 'N':
    land_surface_condition = 0
elif land_surface_condition_choice == 'O':
    land_surface_condition = 1
else:
    land_surface_condition = 2

# roof_type
st.markdown("Type of roof used while building:")
roof_type_choice = st.radio('', ['N', 'Q', 'X'])

if roof_type_choice == 'N':
    roof_type = 0
elif roof_type_choice == 'Q':
    roof_type = 1
else:
    roof_type = 2


# generate random features on button click
st.markdown("### Generate random feature values to make a prediction! 🔮")
click = st.button("Generate Values & Make Prediction")

if click:

    # randomly generated features

    # geo_level
    geo_level_1_id = np.random.randint(0, 30)
    geo_level_2_id = np.random.randint(0, 1427)
    geo_level_3_id = np.random.randint(0, 12567)

    # age_of_building
    age = np.random.randint(0, 995)

    # area_percentage
    area_percentage = np.random.randint(1, 100)

    # height percentage
    height_percentage = np.random.randint(2, 32)

    # ground_floor_type
    ground_floor_type_choice = np.random.choice(['F', 'M', 'V', 'X', 'Z'])

    if ground_floor_type_choice == 'F':
        ground_floor_type = 0
    elif ground_floor_type_choice == 'M':
        ground_floor_type = 1
    elif ground_floor_type_choice == 'V':
        ground_floor_type = 2
    elif ground_floor_type_choice == 'X':
        ground_floor_type = 3
    elif ground_floor_type_choice == 'Z':
        ground_floor_type = 4

    # other_floor_type
    other_floor_type_choice = np.random.choice(['J', 'Q', 'S', 'X'])

    if other_floor_type_choice == 'J':
        other_floor_type = 0
    elif other_floor_type_choice == 'Q':
        other_floor_type = 1
    elif other_floor_type_choice == 'S':
        other_floor_type = 2
    elif other_floor_type_choice == 'X':
        other_floor_type = 3

    # position
    position_choice = np.random.choice(['J', 'O', 'S', 'T'])

    if position_choice == 'J':
        position = 0
    elif position_choice == 'O':
        position = 1
    elif position_choice == 'S':
        position = 2
    elif position_choice == 'T':
        position = 3

    # plan_configuration
    plan_configuration_choice = np.random.choice(['A', 'C', 'D', 'F', 'M', 'N', 'O', 'Q', 'S', 'U'])

    if plan_configuration_choice == 'A':
        plan_configuration = 0
    elif plan_configuration_choice == 'C':
        plan_configuration = 1
    elif plan_configuration_choice == 'D':
        plan_configuration = 2
    elif plan_configuration_choice == 'F':
        plan_configuration = 3
    elif plan_configuration_choice == 'M':
        plan_configuration = 4
    elif plan_configuration_choice == 'N':
        plan_configuration = 5
    elif plan_configuration_choice == 'O':
        plan_configuration = 6
    elif plan_configuration_choice == 'Q':
        plan_configuration = 7
    elif plan_configuration_choice == 'S':
        plan_configuration = 8
    elif plan_configuration_choice == 'U':
        plan_configuration = 9

    # legal_ownership_status 
    legal_ownership_status_choice = np.random.choice(['A', 'R', 'V', 'W'])

    if legal_ownership_status_choice == 'A':
        legal_ownership_status = 0
    elif legal_ownership_status_choice == 'R':
        legal_ownership_status = 1
    elif legal_ownership_status_choice == 'V':
        legal_ownership_status = 2
    elif legal_ownership_status_choice == 'W':
        legal_ownership_status = 3

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

    # write other randomly generated features to a chart
    st.subheader("Randomly Generated Feature Values")

    st.write(pd.DataFrame({'Feature' : ['Geo Level 1 ID (0-30)', 'Geo Level 2 ID (0-1427)', 'Geo Level 3 ID (0-12567)', 'Age of Building', 'Area Percentage', 'Height Percentage', 'Ground Floor Type', 
    'Other Floor Type', 'Position', 'Plan Configuration', 'Legal Ownership Status', 'Adobe/Mud Superstructure? (0 - No, 1 - Yes)', 'Mud Mortar - Stone Superstructure? (0 - No, 1 - Yes)', 
    'Stone Superstructure? (0 - No, 1 - Yes)', 'Cement Mortar - Stone Superstructure? (0 - No, 1 - Yes)', 'Mud Mortar - Brick Superstructure? (0 - No, 1 - Yes)', 
    'Cement Mortar - Brick Superstructure? (0 - No, 1 - Yes)', 'Timber Superstructure? (0 - No, 1 - Yes)', 'Bamboo Superstructure? (0 - No, 1 - Yes)', 
    'Non-Engineered Reinforced Concrete Superstructure? (0 - No, 1 - Yes)', 'Engineered Reinforced Concrete Superstructure? (0 - No, 1 - Yes)', 'Other Material Superstructure? (0 - No, 1 - Yes)', 
    'Has Secondary Use? (0 - No, 1 - Yes)', 'Agricultural Secondary Use? (0 - No, 1 - Yes)', 'Hotel Secondary Use? (0 - No, 1 - Yes)', ' Rental Secondary Use? (0 - No, 1 - Yes)', 
    'Institution Secondary Use? (0 - No, 1 - Yes)', 'School Secondary Use? (0 - No, 1 - Yes)', 'Industrial Secondary Use? (0 - No, 1 - Yes)', 'Health Post Secondary Use? (0 - No, 1 - Yes)', 
    'Government Office Secondary Use? (0 - No, 1 - Yes)', 'Police Station Secondary Use? (0 - No, 1 - Yes)', 'Other Secondary Use Purposes? (0 - No, 1 - Yes)'],
        'Value' : [geo_level_1_id, geo_level_2_id, geo_level_3_id, age, area_percentage, height_percentage, ground_floor_type_choice, other_floor_type_choice, position_choice, plan_configuration_choice, 
        legal_ownership_status_choice,
        has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone, has_superstructure_stone_flag, has_superstructure_cement_mortar_stone, has_superstructure_mud_mortar_brick, 
        has_superstructure_cement_mortar_brick, has_superstructure_timber, has_superstructure_bamboo, has_superstructure_rc_non_engineered, has_superstructure_rc_engineered, has_superstructure_other, 
        has_secondary_use, has_secondary_use_agriculture, has_secondary_use_hotel, has_secondary_use_rental, has_secondary_use_institution, has_secondary_use_school, 
        has_secondary_use_industry, has_secondary_use_health_post, has_secondary_use_gov_office, has_secondary_use_use_police, has_secondary_use_other]
    }))

    # model predictions
    
    all_features = np.array([count_floors, count_families, foundation_type, land_surface_condition, roof_type, geo_level_1_id, geo_level_2_id, geo_level_3_id, age, area_percentage, height_percentage, 
    ground_floor_type, other_floor_type, position, plan_configuration, legal_ownership_status, has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone, has_superstructure_stone_flag, 
    has_superstructure_cement_mortar_stone, has_superstructure_mud_mortar_brick, has_superstructure_cement_mortar_brick, has_superstructure_timber, has_superstructure_bamboo, has_superstructure_rc_engineered, 
    has_superstructure_rc_non_engineered, has_superstructure_other, has_secondary_use, has_secondary_use_agriculture, has_secondary_use_hotel, has_secondary_use_rental, has_secondary_use_institution, 
    has_secondary_use_school, has_secondary_use_industry, has_secondary_use_health_post, has_secondary_use_gov_office, has_secondary_use_use_police, has_secondary_use_other])
    prediction = model.predict(all_features.reshape(1, -1))

    if prediction == 1:
        st.header(f'The model predicts a damage grade of {prediction[0]} - low building damage 🏠 ✔️')
    elif prediction == 2:
        st.header(f'The model predicts a damage grade of {prediction[0]} - medium amount of building damage 🏠 🔨')
    else:
        st.header(f'The model predicts a damage grade of {prediction[0]} - almost complete building destruction 🏚 ❌')

# data dictionary source
st.write("##")
st.markdown("<a href='https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/'>Data Dictionary / Data Source</a>", unsafe_allow_html=True)
