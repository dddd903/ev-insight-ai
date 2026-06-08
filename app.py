import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# =========================
# CONFIG
# =========================

API_KEY = "AQ.Ab8RN6K9Etv4Bt48rowfxgKjzYmY_ExcUbAvdaE3xJSPIeLzaw"

genai.configure(api_key=API_KEY)
model_ai = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open("ev_price_model.pkl", "rb"))

# =========================
# SESSION STATE INIT
# =========================

if "price" not in st.session_state:
    st.session_state.price = None

# =========================
# UI
# =========================

st.title("🚗 EV Intelligent Resale System")
st.write("Predict + Analyze + AI Explanation")

st.divider()

# =========================
# INPUT
# =========================

battery = st.number_input("Battery (kWh)", 10.0, 200.0, 60.0)
range_km = st.number_input("Remaining Range", 50.0, 1000.0, 350.0)
age = st.number_input("Car Age", 0, 20, 3)

# =========================
# PREDICTION
# =========================

if st.button("Predict Price"):

    X = np.array([[battery, range_km, age]])
    price = model.predict(X)[0]

    st.session_state.price = float(price)

    st.success(f"Predicted Price: INR {price:,.2f}")

    if price > 1000000:
        st.info("High Value EV")
    elif price > 500000:
        st.info("Medium Value EV")
    else:
        st.warning("Low Value EV")

# =========================
# AI CHAT
# =========================

st.divider()
st.subheader("AI Assistant")

question = st.text_area("Ask question")

if st.button("Ask AI"):

    if st.session_state.price is None:
        st.warning("Please run prediction first")

    elif question.strip() == "":
        st.warning("Please enter a question")

    else:

        prompt = f"""
You are an EV resale expert.

Rules:
- Only use given data
- Do NOT hallucinate
- Use prediction value only
- Keep answer under 100 words

Battery: {battery}
Range: {range_km}
Age: {age}
Predicted Price: {st.session_state.price}

Question: {question}
"""

        try:
            response = model_ai.generate_content(prompt)
            st.subheader("AI Response")
            st.write(response.text)

        except Exception as e:
            st.error("AI unavailable. Using fallback response.")

            st.write(f"""
EV Analysis:

Battery: {battery} kWh  
Range: {range_km} km  
Age: {age} years  

Predicted Price: INR {st.session_state.price}

Key Insight:
Battery, range, and age are main factors affecting EV resale value.
Older vehicles generally have lower resale value due to depreciation.
""")