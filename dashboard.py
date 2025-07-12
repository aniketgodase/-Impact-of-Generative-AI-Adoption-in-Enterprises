import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
st.set_page_config(page_title='GenAI Dashboard', layout='centered')
st.title('GenAI Workforce Impact Dashboard')
try:
    df = pd.read_csv('outputs/final_dataset.csv')
    model = joblib.load('outputs/model.pkl')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
st.subheader("1. Sample Data")
st.dataframe(df.head())
st.subheader("2. Sentiment Summary")
if 'Sentiment_Label' in df.columns:
    sentiment_counts = df['Sentiment_Label'].value_counts()
    st.bar_chart(sentiment_counts)
else:
    st.warning("Sentiment_Label column not found in data.")
st.subheader("3. Predict Productivity Change (%)")
col1, col2, col3 = st.columns(3)
with col1:
    impacted = st.number_input("Employees Impacted", 0, 5000, 100)
with col2:
    roles = st.number_input("New Roles Created", 0, 500, 5)
with col3:
    hours = st.number_input("Training Hours", 0, 10000, 1000)
if st.button("Predict"):
    try:
        prediction = model.predict([[impacted, roles, hours]])[0]
        st.success(f" Predicted Productivity Change: {prediction:.2f}%")
    except Exception as e:
        st.error(f"Prediction error: {e}")
st.subheader("4. Cluster Visualization")
try:
    st.image("outputs/visuals/clusters.png", caption="Enterprise Clusters", use_column_width=True)
except:
    st.warning("Cluster image not found.")
