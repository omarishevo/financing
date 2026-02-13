import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction App")
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Select target column
        target_column = st.selectbox("Select Target Column (House Price)", df.columns)

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=np.number)

        if target_column not in numeric_df.columns:
            st.error("Target column must be numeric.")
        else:
            X = numeric_df.drop(columns=[target_column])
            y = numeric_df[target_column]

            if X.shape[1] == 0:
                st.error("No numeric feature columns found.")
            else:
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Evaluation
                st.subheader("Model Performance")

                col1, col2, col3 = st.columns(3)

                col1.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
                col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
                col3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

                st.subheader("Make a Prediction")

                input_data = {}
                for column in X.columns:
                    input_data[column] = st.number_input(
                        f"Enter {column}", 
                        float(X[column].min()), 
                        float(X[column].max()), 
                        float(X[column].mean())
                    )

                if st.button("Predict Price"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    st.success(f"Predicted House Price: {prediction:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV dataset to begin.")
