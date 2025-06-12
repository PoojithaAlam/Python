import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

def main():
    st.title("🏠 Custom House Price Predictor")
    st.write("Upload your own dataset and predict house prices using Linear Regression.")

    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        # Let user pick the target column
        target_col = st.selectbox("🎯 Select the target (price) column", df.columns)

        # Let user pick features
        features = st.multiselect(
            "📌 Select feature columns to use for prediction",
            [col for col in df.columns if col != target_col]
        )

        if features and target_col:
            # Split and train
            X = df[features]
            y = df[target_col]

            model = LinearRegression()
            model.fit(X, y)

            # User input form
            st.subheader("✏️ Enter feature values to predict price")
            input_data = []
            for feature in features:
                val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
                input_data.append(val)

            if st.button("🔮 Predict Price"):
                prediction = model.predict([input_data])[0]
                st.success(f"Estimated Price: ${prediction:,.2f}")

            # Chart
            st.subheader("📈 Scatter Plot (First feature vs Price)")
            fig = px.scatter(df, x=features[0], y=target_col, title=f"{features[0]} vs {target_col}")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
