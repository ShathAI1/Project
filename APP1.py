import streamlit as st
import numpy as np
import joblib
import pickle as p

# Load the trained Random Forest model and scaler
with open('Ourfinalisedmodel.pickle', 'rb') as f:
    pipe = p.load(f)

rf_model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')


def main():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title('Energy Prediction App')

    # Input fields
    purchases = st.number_input('Purchases', value=0.0)
    sb_total = st.number_input('SB Total', value=0.0)
    sb_solar = st.number_input('SB Solar', value=0.0)
    sb_non_solar = st.number_input('SB Non-Solar', value=0.0)
    sb_hydro = st.number_input('SB Hydro', value=0.0)
    mcv_total = st.number_input('MCV Total', value=0.0)
    mcv_solar = st.number_input('MCV Solar', value=0.0)
    mcv_non_solar = st.number_input('MCV Non-Solar', value=0.0)
    mcv_hydro = st.number_input('MCV Hydro', value=0.0)
    fsv_total = st.number_input('FSV Total', value=0.0)
    fsv_solar = st.number_input('FSV Solar', value=0.0)
    fsv_non_solar = st.number_input('FSV Non-Solar', value=0.0)
    fsv_hydro = st.number_input('FSV Hydro', value=0.0)

    if st.button('Predict'):
        # Collect input features into an array
        features = np.array([[purchases, sb_total, sb_solar, sb_non_solar, sb_hydro,
                              mcv_total, mcv_solar, mcv_non_solar, mcv_hydro,
                              fsv_total, fsv_solar, fsv_non_solar, fsv_hydro]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = rf_model.predict(features_scaled)[0]

        # Display the prediction
        st.success(f'The predicted energy output is: {prediction}')


if __name__ == '__main__':
    main()
