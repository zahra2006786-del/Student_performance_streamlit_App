import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline


Pipeline = joblib.load('LinearRegression.joblib')

st.markdown('# ***Know your performance***')
df = pd.read_csv("Student_Performance.csv")
hour = st.selectbox('how many Hours you studied ?',df['Hours Studied'].unique())
prev_score = st.number_input('What is your previous score ?')

extra_act = st.selectbox('Are you involved in Extracurricular Activities ?', df['Extracurricular Activities'].unique())
no = 0 if extra_act == 'False' else 1

sleep = st.selectbox('How many hours do you sleep ?', df['Sleep Hours'].unique())
sample = st.selectbox('Number of Sample question papers you practiced ?', df['Sample Question Papers Practiced'].unique())

pred = pd.DataFrame({'Hours Studied': [hour], 'Previous Scores': [prev_score], 'Extracurricular Activities': [no], 'Sleep Hours': [sleep], 'Sample Question Papers Practiced':[sample]})

if st.button('Predict'):
    index = Pipeline.predict(pred)
    st.success(f"Your performance index is: {int(np.round(index, 0))}")