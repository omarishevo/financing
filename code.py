import streamlit as st
import pandas as pd

# Load data function (no machine learning involved)
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Streamlit Layout
st.title("📊 Data Visualization Dashboard")
st.sidebar.header("Upload Your Dataset")

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.success("✅ Dataset uploaded!")

        # Show raw data
        st.subheader("🗃 Raw Data")
        st.dataframe(df.head())

        # Basic data visualization
        st.subheader("📈 Basic Data Plot")
        
        # Ensure 'Date' is a datetime column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Plot using Streamlit's built-in line chart
        st.line_chart(df.set_index('Date')['Close Price'])

    except Exception as e:
        st.error(f"❌ Error loading the file: {e}")
else:
    st.info("📤 Please upload a CSV file to begin.")
