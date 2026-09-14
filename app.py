import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
)


# Load the pre-trained XGBoost model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_predictive_model.pkl")


model = load_model()

# App Title & Description
st.title("⚙️ Industrial IoT Predictive Maintenance System")
st.write(
    "Enter real-time IoT sensor telemetry data below to predict potential machine failures."
)

st.markdown("---")

# Sidebar for Machine Type Input
st.sidebar.header("🔧 Machine Configuration")
type_input = st.sidebar.selectbox(
    "Machine Type",
    [
        "Select Machine Type...",
        "L (Low Quality Variant)",
        "M (Medium Quality Variant)",
        "H (High Quality Variant)",
    ],
    index=0,
)

# Input Layout for Sensor Data (Initially Empty)
st.subheader("📊 Sensor Telemetry Data")
col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=290.0,
        max_value=310.0,
        value=None,
        placeholder="e.g. 300.0",
    )
    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=300.0,
        max_value=320.0,
        value=None,
        placeholder="e.g. 310.0",
    )
    rpm = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=None,
        placeholder="e.g. 1500",
    )

with col2:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=None,
        placeholder="e.g. 40.0",
    )
    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=None,
        placeholder="e.g. 108",
    )

st.markdown("---")

# Real-time Prediction Action
if st.button("🚀 Predict Machine Health", type="primary"):
    # Validation 1: Check Machine Type
    if type_input == "Select Machine Type...":
        st.warning(
            "⚠️ Please select a valid **Machine Type** from the sidebar!"
        )
    # Validation 2: Check Empty Inputs
    elif None in [air_temp, process_temp, rpm, torque, tool_wear]:
        st.error(
            "⚠️ Please fill in all the sensor telemetry values before predicting!"
        )
    else:
        # One-Hot Encoding mapping
        type_M = 1 if "M (Medium" in type_input else 0
        type_L = 1 if "L (Low" in type_input else 0

        # Construct Input DataFrame matching model exact feature order
        input_data = pd.DataFrame(
            [
                {
                    "Air temperature K": air_temp,
                    "Process temperature K": process_temp,
                    "Rotational speed rpm": rpm,
                    "Torque Nm": torque,
                    "Tool wear min": tool_wear,
                    "Type_L": type_L,
                    "Type_M": type_M,
                }
            ]
        )

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        st.subheader("💡 Prediction Result")

        if prediction == 1:
            st.error(
                f"⚠️ **WARNING: Potential Machine Failure Detected!**\n\n"
                f"**Failure Risk Probability:** `{prediction_proba:.2%}`"
            )
        else:
            st.success(
                f"✅ **Machine Status: Normal / Healthy**\n\n"
                f"**Failure Risk Probability:** `{prediction_proba:.2%}`"
            )