import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# =====================================
# GEMINI CONFIGURATION
# =====================================

API_KEY = "AQ.Ab8RN6KG-NvQDTfp3jJhjUKVc8YyroSSgDcaAv5IaBhYfR-J6A"

genai.configure(api_key=API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="EV Insight AI",
    page_icon="🚗",
    layout="centered"
)

# =====================================
# LOAD MODEL
# =====================================

with open("ev_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# =====================================
# SESSION STATE
# =====================================

if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = None

# =====================================
# TITLE
# =====================================

st.title("🚗 EV Insight AI")

st.subheader(
    "Electric Vehicle Resale Value Prediction & Advisory System"
)

st.write(
    "Predict EV resale value using Machine Learning and receive AI-powered insights."
)

st.divider()

# =====================================
# USER INPUT
# =====================================

st.header("Vehicle Information")

battery = st.number_input(
    "Battery Capacity (kWh)",
    min_value=10.0,
    max_value=200.0,
    value=60.0
)

range_km = st.number_input(
    "Remaining Range (km)",
    min_value=50.0,
    max_value=1000.0,
    value=350.0
)

age = st.number_input(
    "Car Age (Years)",
    min_value=0,
    max_value=20,
    value=3
)

# =====================================
# PREDICTION BUTTON
# =====================================

if st.button("Predict Resale Price"):

    features = np.array([
        [battery, range_km, age]
    ])

    prediction = model.predict(features)

    st.session_state.predicted_price = float(prediction[0])

    st.success(
        f"Predicted Resale Price: INR {st.session_state.predicted_price:,.2f}"
    )

    if st.session_state.predicted_price > 1000000:

        st.info(
            "Excellent resale value. Strong market potential."
        )

    elif st.session_state.predicted_price > 500000:

        st.info(
            "Moderate resale value. Competitive in the used EV market."
        )

    else:

        st.warning(
            "Lower resale value. Age and battery condition may affect demand."
        )

# =====================================
# DISPLAY CURRENT PREDICTION
# =====================================

if st.session_state.predicted_price is not None:

    st.divider()

    st.subheader("Current Prediction")

    st.success(
        f"INR {st.session_state.predicted_price:,.2f}"
    )

# =====================================
# AI ASSISTANT
# =====================================

st.divider()

st.header("🤖 EV AI Assistant")

question = st.text_area(
    "Ask a question about EV resale value, battery health, depreciation, or future market trends"
)

if st.button("Ask AI"):

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        prompt = f"""
You are an Electric Vehicle Resale Value Expert.

Machine Learning Prediction:
Predicted Resale Price = INR {st.session_state.predicted_price}

Vehicle Information:
Battery Capacity = {battery} kWh
Remaining Range = {range_km} km
Vehicle Age = {age} years

User Question:
{question}

Instructions:
1. Use the machine learning prediction in your answer.
2. Explain how battery capacity, remaining range, and vehicle age affect resale value.
3. Keep the response professional.
4. Limit the response to approximately 150 words.
"""

        try:

            response = gemini_model.generate_content(
                prompt
            )

            st.subheader("AI Response")

            st.write(response.text)

        except Exception as e:

            st.error(
                "Gemini AI is temporarily unavailable."
            )

            st.write(str(e))

            fallback = f"""
Based on the available vehicle information:

• Battery Capacity: {battery} kWh

• Remaining Range: {range_km} km

• Vehicle Age: {age} years

Predicted Resale Price:
INR {st.session_state.predicted_price}

Analysis:

A higher battery capacity and longer remaining range generally improve resale value because buyers prefer vehicles with stronger battery performance.

Vehicle age contributes to depreciation. As EVs get older, battery wear and technological advancements may reduce market value.

Maintaining battery health and keeping service records can help preserve resale value.
"""

            st.subheader("Local AI Analysis")

            st.write(fallback)

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Powered by Linear Regression, Streamlit, and Google Gemini AI"
)