from __future__ import annotations

import streamlit as st

from fertilizer_recommender.pipeline import get_feature_options, load_artifacts, predict_recommendation

FERTILIZER_INFO = {
    "Urea": "High nitrogen fertilizer that supports leafy growth and fast greening.",
    "DAP": "Di-ammonium phosphate, useful when crops need phosphorus support and early root development.",
    "MOP": "Muriate of potash, a potassium-rich fertilizer that helps crop quality and disease resistance.",
    "NPK": "A balanced fertilizer that supplies nitrogen, phosphorus, and potassium together.",
    "SSP": "Single super phosphate, often used to add phosphorus and sulfur.",
    "Compost": "Organic matter that improves soil structure and releases nutrients gradually.",
    "Zinc Sulphate": "A micronutrient fertilizer used to correct zinc deficiency and improve yield.",
}


st.set_page_config(
    page_title="Fertilizer Recommendation",
    page_icon="🌱",
    layout="wide",
)


@st.cache_data
def load_options() -> dict[str, list[str]]:
    return get_feature_options()


@st.cache_resource
def load_prediction_artifacts():
    return load_artifacts()


def build_inputs(options: dict[str, list[str]]) -> dict[str, object]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Soil details")
        soil_type = st.selectbox("Soil type", options["Soil_Type"])
        soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5, step=0.1)
        soil_moisture = st.slider("Soil moisture (%)", 0.0, 100.0, 35.0, step=0.5)
        organic_carbon = st.slider("Organic carbon (%)", 0.0, 5.0, 1.2, step=0.1)
        electrical_conductivity = st.slider("Electrical conductivity", 0.0, 3.0, 0.5, step=0.1)
        nitrogen_level = st.number_input("Nitrogen level", min_value=0, max_value=300, value=80)
        phosphorus_level = st.number_input("Phosphorus level", min_value=0, max_value=300, value=40)
        potassium_level = st.number_input("Potassium level", min_value=0, max_value=300, value=60)
        fertilizer_used_last_season = st.slider(
            "Fertilizer used last season (kg/ha)", 0.0, 400.0, 175.0, step=5.0
        )
        yield_last_season = st.slider("Yield last season (tons/ha)", 0.0, 10.0, 4.5, step=0.1)

    with col2:
        st.subheader("Crop and environment")
        crop_type = st.selectbox("Crop type", options["Crop_Type"])
        crop_growth_stage = st.selectbox("Crop growth stage", options["Crop_Growth_Stage"])
        season = st.selectbox("Season", options["Season"])
        irrigation_type = st.selectbox("Irrigation type", options["Irrigation_Type"])
        previous_crop = st.selectbox("Previous crop", options["Previous_Crop"])
        region = st.selectbox("Region", options["Region"])
        temperature = st.slider("Temperature (C)", 0.0, 50.0, 25.0, step=0.5)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0, step=1.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 2000.0, 800.0, step=10.0)

    return {
        "Soil_Type": soil_type,
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": electrical_conductivity,
        "Nitrogen_Level": int(nitrogen_level),
        "Phosphorus_Level": int(phosphorus_level),
        "Potassium_Level": int(potassium_level),
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": crop_growth_stage,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Previous_Crop": previous_crop,
        "Region": region,
        "Fertilizer_Used_Last_Season": fertilizer_used_last_season,
        "Yield_Last_Season": yield_last_season,
    }


def main() -> None:
    st.title("Fertilizer Recommendation System")
    st.caption("A Streamlit app that predicts a fertilizer recommendation from soil, crop, and weather inputs.")

    try:
        artifacts = load_prediction_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the project first with `python train_model.py`, then rerun `streamlit run app.py`.")
        st.stop()

    options = load_options()
    raw_inputs = build_inputs(options)

    if st.button("Get recommendation", type="primary", use_container_width=True):
        prediction = predict_recommendation(raw_inputs, artifacts=artifacts)
        st.success(f"Recommended fertilizer: {prediction}")
        details = FERTILIZER_INFO.get(prediction)
        if details:
            st.info(details)


if __name__ == "__main__":
    main()
