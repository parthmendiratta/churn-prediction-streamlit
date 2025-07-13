import streamlit as st
import pandas as pd
import joblib

model=joblib.load("churn_model_tuned.pkl")

st.title("Customer churn prediction app")
st.write("Upload your customer data to predict who might churn")

uploaded_file=st.file_uploader("Choose a csv file",type="csv")

if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.subheader("Input Data Preview")
    st.write(data.head())

    if st.button("Predict Churn"):
        try:
            predictions=model.predict(data)
            data['Churn Predictions']=predictions
            st.success("Prediction Compelete")
            st.write(data)

            csv=data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error in prediction: {e} ")
