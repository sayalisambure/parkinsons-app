import pickle
import numpy as np
import streamlit as st
import pandas as pd

st.title("Parkinson's Disease Prediction")
st.caption("Predicts from voice measurements in a research dataset. Not a medical diagnosis.")

model = pickle.load(open("trained_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

features = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]

defaults = [197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166,
            0.00168, 0.00498, 0.01098, 0.09700, 0.00563, 0.00680,
            0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367,
            -7.348300, 0.177551, 1.743867, 0.085569]

values = []
cols = st.columns(3)
for i, name in enumerate(features):
    values.append(cols[i % 3].number_input(name, value=defaults[i], format="%.6f"))

if st.button("Predict"):
    data = pd.DataFrame([values], columns=features)
    data = scaler.transform(data)
    result = model.predict(data)[0]
    if result == 1:
        st.error("The model predicts Parkinson's disease.")
    else:
        st.success("The model predicts a healthy person.")