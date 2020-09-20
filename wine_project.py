import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Wine Prediction App

The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators.
There are thirteen different measurements taken for different constituents found in the three types of wine.
   """)

st.sidebar.header('User Input Parameters')

def user_input_features():
	Alcohol = st.sidebar.slider('Alcohol', 11.0, 14.8, 13.0)
	Malic_Acid = st.sidebar.slider('Malic Acid', 0.74, 5.80, 2.34)
	Ash = st.sidebar.slider('Ash', 1.36, 3.23, 2.36)
	Alcalinity_of_Ash = st.sidebar.slider('Alcalinity of Ash', 10.6, 30.0, 19.5)
	Magnesium = st.sidebar.slider('Magnesium', 70.0, 162.0, 99.7)
	Total_Phenols = st.sidebar.slider('Total Phenols', 0.98, 3.88, 2.29)
	Flavanoids = st.sidebar.slider('Flavanoids', 0.34, 5.08, 2.03)
	Nonflavanoid_Phenols = st.sidebar.slider('Nonflavanoid Phenols', 0.13, 0.66, 0.36)
	Proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.41, 3.58, 1.59)
	Colour_Intensity = st.sidebar.slider('Colour Intensity', 1.3, 13.0, 5.1)
	Hue = st.sidebar.slider('Hue', 0.48, 1.71, 0.96)
	diluted_wines = st.sidebar.slider('OD280/OD315 of diluted wines', 1.27, 4.00, 2.61)
	Proline = st.sidebar.slider('Proline', 278, 1680, 746)
	data = {'Alcohol': Alcohol,
			'Malic Acid' : Malic_Acid,
			'Ash': Ash,
			'Alcalinity of Ash' : Alcalinity_of_Ash,
			'Magnesium' : Magnesium,
			'Total Phenols' : Total_Phenols,
			'Flavanoids' : Flavanoids,
			'Nonflavanoid Phenols' : Nonflavanoid_Phenols,
			'Proanthocyanins' : Proanthocyanins,
			'Colour Intensity' : Colour_Intensity,
			'Hue' : Hue,
			'OD280/OD315 of diluted wines' : diluted_wines,
			'Proline' : Proline}
	features = pd.DataFrame(data, index=[0])
	return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

wine = datasets.load_wine()
X = wine.data
Y = wine.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels, Distribution and their corresponding index number')
st.write("""
class_0 (59), class_1 (71), class_2 (48)
   """)
st.write(wine.target_names)

st.subheader('Prediction')
st.write(wine.target_names[prediction])

st.subheader('Prediction Probabilty')
st.write(prediction_proba)