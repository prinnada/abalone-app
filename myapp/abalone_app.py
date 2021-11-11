import streamlit as st
import pandas as pd

st.write("""
# My First Web Application
Let's enjoy **data science** project!
""")
st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

v_Sex = st.sidebar.radio('Sex', ['Male','Female','Infant'])
v_Length = st.sidebar.slider('Length', min_value=0.0, max_value=1.0, value = 0.5)
v_Diameter = st.sidebar.slider('Diameter', min_value=0.0, max_value=1.0, value = 0.5)
v_Height = st.sidebar.slider('Height', min_value=0.0, max_value=1.0, value = 0.5)
v_Whole_weight = st.sidebar.slider('Whole Weight', min_value=0.0, max_value=1.0, value = 0.5)
v_Shucked_weight = st.sidebar.slider('Shucked Weight', min_value=0.0, max_value=1.0, value = 0.5)
v_Viscera_weight = st.sidebar.slider('Viscera Weight', min_value=0.0, max_value=1.0, value = 0.5)
v_Shell_weight = st.sidebar.slider('Shell Weight', min_value=0.0, max_value=1.0, value = 0.5)

if v_Sex == 'Male': v_Sex = 'M'
elif v_Sex == 'Female': v_Sex = 'F'
else: v_Sex = 'I'

data = {
    'Sex': v_Sex,
    'Length': v_Length,
    'Diameter': v_Diameter,
    'Height': v_Height,
    'Whole_weight': v_Whole_weight,
    'Shucked_weight': v_Shucked_weight,
    'Viscera_weight': v_Viscera_weight,
    'Shell_weight': v_Shell_weight,
    }

df = pd.DataFrame(data, index=[0])

st.header('Application of Abalone\'s Age Prediction:')
st.subheader('User Input:')

st.write(df)

data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)
#st.write(df)

cat_data = pd.get_dummies(df[['Sex']])

X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1]

X_new = X_new.drop(columns=['Sex'])

st.subheader('Pre-Processed Input:')
st.write(X_new)

import pickle
load_nor = pickle.load(open('normalization.pkl', 'rb'))
X_new = load_nor.transform(X_new)
st.subheader('Normalized Input:')
st.write(X_new)

load_knn = pickle.load(open('best_knn.pkl', 'rb'))
prediction = load_knn.predict(X_new)
st.subheader('Prediction:')
st.write(prediction)