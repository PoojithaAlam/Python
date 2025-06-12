import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    st.title("🏠 California Housing Price Predictor")
    st.write("Using the California Housing dataset from Scikit-learn")

    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    feature_names = housing.feature_names
    target_name = housing.target_names[0]

    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    # User feature selection (from sidebar)
    st.sidebar.title("🛠 Feature Inputs")
    input_features = {}
    for feature in feature_names:
        val = st.sidebar.slider(
            f"{feature}", 
            float(df[feature].min()), 
            float(df[feature].max()), 
            float(df[feature].mean())
        )
        input_features[feature] = val

    # Train model
    X = df[feature_names]
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make prediction
    input_df = pd.DataFrame([input_features])
    prediction = model.predict(input_df)[0]

    st.subheader("🏷️ Prediction Result")
    st.success(f"Estimated Median House Value: ${prediction * 100000:,.2f}")

    # Model performance
    st.subheader("📉 Model Performance")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error on test data: {mse:.4f}")

# Run app
if __name__ == "__main__":
    main()
