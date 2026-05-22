# STREAMLIT APP

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_performance.csv")

# Train model
X = df.drop("final_score", axis=1)
y = df["final_score"]

model = LinearRegression()
model.fit(X, y)

# UI
st.title("🎓 Student Performance Predictor")

study_hours = st.number_input("Study Hours")
attendance = st.number_input("Attendance")
sleep_hours = st.number_input("Sleep Hours")
previous_score = st.number_input("Previous Score")

if st.button("Predict"):
    input_data = np.array([[study_hours, attendance, sleep_hours, previous_score]])
    result = model.predict(input_data)
    st.success(f"Predicted Score: {result[0]:.2f}")