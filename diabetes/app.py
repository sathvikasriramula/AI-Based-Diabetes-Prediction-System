import streamlit as st
import pandas as pd
import joblib
import os
import base64



st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)



background_image = "imagee.jpg"

if os.path.exists(background_image):

    with open(background_image, "rb") as image_file:

        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    page_background = f"""
        background:
            linear-gradient(
                rgba(240, 249, 252, 0.88),
                rgba(230, 246, 250, 0.88)
            ),
            url("data:image/jpeg;base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    """

else:

    page_background = """
        background:
            linear-gradient(
                135deg,
            );
    """



st.markdown(
    f"""
    <style>

    


    .stApp {{
        {page_background}
    }}


    


    .block-container {{
        max-width: 1050px;

        padding-top: 35px;
        padding-bottom: 50px;
    }}


    


    h1 {{
        color: #123b5d !important;

        font-weight: 800 !important;

        font-size: 42px !important;
    }}

    h2 {{
        color: #123b5d !important;

        font-weight: 750 !important;
    }}

    h3 {{
        color: #123b5d !important;
    }}


    


    p {{
        color: #344b5a !important;
    }}


    


    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {{

        color: #123b5d !important;

        font-weight: 700 !important;

        font-size: 15px !important;
    }}


    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p {{

        color: #123b5d !important;

        font-weight: 700 !important;
    }}


    


    div[data-testid="stNumberInput"] > div {{

        background: #ffffff !important;

        border: 2px solid #8ecddd !important;

        border-radius: 12px !important;

        box-shadow:
            0 3px 10px rgba(20, 100, 130, 0.12) !important;

        overflow: hidden !important;
    }}


    


    div[data-testid="stNumberInput"] input {{

        background: #ffffff !important;

        color: #111827 !important;

        -webkit-text-fill-color: #111827 !important;

        font-size: 16px !important;

        font-weight: 700 !important;

        border: none !important;

        outline: none !important;
    }}


    


    div[data-testid="stNumberInput"] input::placeholder {{

        color: #555555 !important;

        opacity: 1 !important;
    }}


    


    div[data-testid="stNumberInput"] > div:focus-within {{

        border: 2px solid #087fce !important;

        box-shadow:
            0 0 0 3px rgba(8, 127, 206, 0.15) !important;
    }}


    


    div[data-testid="stNumberInput"] button {{

        background: #087fce !important;

        color: #ffffff !important;

        border: none !important;

        font-weight: 900 !important;
    }}


    div[data-testid="stNumberInput"] button:hover {{

        background: #056aa8 !important;

        color: #ffffff !important;
    }}


    


    div[data-testid="stSelectbox"] > div > div {{

        background: #ffffff !important;

        border: 2px solid #8ecddd !important;

        border-radius: 12px !important;

        box-shadow:
            0 3px 10px rgba(20, 100, 130, 0.12) !important;
    }}


    


    div[data-baseweb="select"] span {{

        color: #111827 !important;

        font-weight: 700 !important;
    }}


    div[data-baseweb="select"] input {{

        color: #111827 !important;

        -webkit-text-fill-color: #111827 !important;
    }}


    


    div[data-baseweb="select"] svg {{

        fill: #087fce !important;
    }}


    


    div[role="listbox"] {{

        background: #ffffff !important;

        border-radius: 10px !important;

        border: 1px solid #c6dce5 !important;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }}


    div[role="option"] {{

        background: #ffffff !important;

        color: #111827 !important;

        font-weight: 600 !important;
    }}


    div[role="option"]:hover {{

        background: #e7f6fa !important;

        color: #087fce !important;
    }}


    


    .stButton > button {{

        width: 100%;

        height: 58px;

        border-radius: 13px;

        border: none;

        background:
            linear-gradient(
                90deg,
            ) !important;

        color: #ffffff !important;

        font-size: 17px !important;

        font-weight: 750 !important;

        box-shadow:
            0 7px 18px rgba(0, 120, 170, 0.25);
    }}


    .stButton > button:hover {{

        background:
            linear-gradient(
                90deg,
            ) !important;

        color: #ffffff !important;
    }}


    


    div[data-testid="stMetric"] {{

        background: #ffffff !important;

        border: 1px solid #cce5ed !important;

        border-radius: 13px !important;

        padding: 15px !important;

        box-shadow:
            0 4px 12px rgba(0, 70, 100, 0.08);
    }}


    div[data-testid="stMetricLabel"] {{

        color: #607583 !important;
    }}


    div[data-testid="stMetricValue"] {{

        color: #123b5d !important;

        font-weight: 800 !important;
    }}


    


    div[data-testid="stProgress"] {{

        margin-top: 10px;

        margin-bottom: 15px;
    }}


    


    div[data-testid="stAlert"] {{

        border-radius: 12px !important;
    }}


    


    hr {{

        border-color: #b9dce6 !important;
    }}


    



        visibility: hidden;
    }}

    footer {{

        visibility: hidden;
    }}

    [data-testid="stToolbar"] {{

        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True
)



try:

    model = joblib.load(
        "diabetes_model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    imputer = joblib.load(
        "imputer.pkl"
    )

    feature_columns = joblib.load(
        "feature_columns.pkl"
    )

except Exception as error:

    st.error(
        "❌ Machine Learning model files could not be loaded."
    )

    st.write(
        "Please make sure these files are in the same "
        "folder as app.py:"
    )

    st.write(
        "• diabetes_model.pkl"
    )

    st.write(
        "• scaler.pkl"
    )

    st.write(
        "• imputer.pkl"
    )

    st.write(
        "• feature_columns.pkl"
    )

    st.code(
        str(error)
    )

    st.stop()



st.title(
    "🩺 Diabetes Risk Prediction"
)

st.subheader(
    "AI Healthcare"
)

st.write(
    "Intelligent health risk assessment powered by "
    "Machine Learning."
)

st.info(
    "Enter the patient's health information below "
    "to estimate diabetes risk."
)



st.divider()

st.header(
    "👤 Patient Details"
)

st.write(
    "Please provide the patient's basic health information "
    "for risk assessment."
)



col1, col2 = st.columns(
    2,
    gap="large"
)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )


with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )



col1, col2 = st.columns(
    2,
    gap="large"
)


with col1:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=24.6,
        step=0.1,
        format="%.1f"
    )


with col2:

    physical_activity = st.number_input(
        "Physical Activity (minutes per week)",
        min_value=0,
        max_value=1500,
        value=150,
        step=10
    )



col1, col2 = st.columns(
    2,
    gap="large"
)


with col1:

    systolic_bp = st.number_input(
        "Systolic Blood Pressure",
        min_value=80,
        max_value=250,
        value=120,
        step=1
    )


with col2:

    diastolic_bp = st.number_input(
        "Diastolic Blood Pressure",
        min_value=40,
        max_value=150,
        value=80,
        step=1
    )



col1, col2 = st.columns(
    2,
    gap="large"
)


with col1:

    cholesterol = st.number_input(
        "Cholesterol Level",
        min_value=50,
        max_value=500,
        value=190,
        step=1
    )


with col2:

    glucose = st.number_input(
        "Glucose Level",
        min_value=40,
        max_value=400,
        value=100,
        step=1
    )



st.write("")

predict_button = st.button(
    "🔍  Predict Diabetes Risk",
    use_container_width=True
)



if predict_button:

    try:


        if physical_activity < 100:

            activity_value = 0

            activity_level = "Low"

        elif physical_activity < 300:

            activity_value = 1

            activity_level = "Moderate"

        else:

            activity_value = 2

            activity_level = "High"



        blood_pressure = (
            systolic_bp + diastolic_bp
        ) / 2



        input_data = pd.DataFrame(
            {
                "Age": [age],

                "BMI": [bmi],

                "Glucose": [glucose],

                "BloodPressure": [
                    blood_pressure
                ],

                "Cholesterol": [
                    cholesterol
                ],

                "PhysicalActivity": [
                    activity_value
                ]
            }
        )



        for column in feature_columns:

            if column not in input_data.columns:

                input_data[column] = 0


        input_data = input_data[
            feature_columns
        ]



        input_imputed = imputer.transform(
            input_data
        )



        input_scaled = scaler.transform(
            input_imputed
        )



        prediction = model.predict(
            input_scaled
        )[0]



        if hasattr(
            model,
            "predict_proba"
        ):

            probability = model.predict_proba(
                input_scaled
            )[0][1]

        else:

            probability = float(
                prediction
            )


        risk_percentage = (
            probability * 100
        )



        st.divider()

        st.header(
            "📊 Prediction Result"
        )


        if prediction == 1:

            st.error(
                "⚠️ Prediction: Higher likelihood of diabetes"
            )

        else:

            st.success(
                "✅ Prediction: Lower likelihood of diabetes"
            )



        st.metric(
            "Estimated Diabetes Probability",
            f"{risk_percentage:.2f}%"
        )


        st.progress(
            min(
                int(risk_percentage),
                100
            )
        )



        if risk_percentage < 30:

            st.info(
                "The model estimates a relatively lower "
                "probability based on the provided information."
            )

        elif risk_percentage < 70:

            st.warning(
                "The model estimates a moderate probability "
                "based on the provided information."
            )

        else:

            st.warning(
                "The model estimates a higher probability "
                "based on the provided information. "
                "Consider discussing the result with a "
                "qualified healthcare professional."
            )



        st.divider()

        st.header(
            "📋 Patient Summary"
        )

        st.write(
            "Patient information used for this prediction:"
        )



        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Age",
                age
            )


        with c2:

            st.metric(
                "BMI",
                f"{bmi:.1f}"
            )


        with c3:

            st.metric(
                "Glucose",
                glucose
            )


        with c4:

            st.metric(
                "Cholesterol",
                cholesterol
            )



        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Blood Pressure",
                f"{systolic_bp}/{diastolic_bp}"
            )


        with c2:

            st.metric(
                "Activity",
                f"{physical_activity} min"
            )


        with c3:

            st.metric(
                "Gender",
                gender
            )


        with c4:

            st.metric(
                "Activity Level",
                activity_level
            )



        st.divider()

        st.info(
            "ℹ️ This AI system is developed for academic "
            "and educational purposes as a Minor Project. "
            "The prediction generated by this system should "
            "not be considered a medical diagnosis."
        )


    except Exception as error:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.code(
            str(error)
        )



st.divider()

st.caption(
    "🩺 AI Healthcare  |  Diabetes Risk Prediction  |  "
    "Machine Learning Minor Project"
)
