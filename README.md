# Parkinson's Disease Prediction App

Predicts whether a voice recording suggests Parkinson's disease, using
voice measurements from the UCI Parkinson's dataset.

**Live app:** https://parkinsons-app-4oq5epoubwjzed6gxgmapp.streamlit.app

## Use Case
- Early screening support: flags voice patterns linked to Parkinson's so a person can be referred for a proper clinical check.
- Remote and low-cost screening: voice recordings need no special equipment.
- Learning tool for healthcare machine learning: shows how a trained model is turned into a working web app.

## Model
- Support Vector Machine (linear kernel) with StandardScaler
- Test accuracy: 87.1% 
## Tech Stack
Python, scikit-learn, pandas, NumPy, Streamlit

## Future Scope
- Upload a voice recording and extract the features automatically, so users don't type 22 values.
- Compare more models (Random Forest, XGBoost) and add precision, recall, and F1.
- Use a patient-wise train/test split for a fairer evaluation.
- Add explainability (SHAP) to show which voice features drive each prediction.
- Train on larger, more varied datasets to improve reliability.
- Add a mobile-friendly interface.

## Dataset
UCI Machine Learning Repository: Parkinson's Dataset

*For educational and screening-support purposes only; not a medical diagnosis.*
