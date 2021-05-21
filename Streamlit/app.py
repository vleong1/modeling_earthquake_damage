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

# age
st.markdown("Age of building:")
age = st.slider('', min_value = 0, max_value = 995, step = 25)

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

# roof_type
st.markdown("Type of roof used while building:")
roof_type_choice = st.radio('', ['N', 'Q', 'X'])

if roof_type_choice == 'N':
    roof_type = 0
elif roof_type_choice == 'Q':
    roof_type = 1
else:
    roof_type = 2

# has_superstructure_mud_mortar_stone
st.markdown("Is the building made out of Mud Mortar Stone?")
has_superstructure_mud_mortar_stone_choice = st.radio('', ['Yes', 'No'])

if has_superstructure_mud_mortar_stone_choice == 'Yes':
    has_superstructure_mud_mortar_stone = 1
else:
    has_superstructure_mud_mortar_stone = 0

# button click prediction
st.markdown("### Make a prediction! 🔮")
click = st.button("Click Here")

if click:

    # model predictions

    all_features = np.array([age, count_families, foundation_type, roof_type, has_superstructure_mud_mortar_stone])
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
