from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "fertilizer_recommendation.csv"
MODEL_DIR = PROJECT_ROOT / "model"

MODEL_PATH = MODEL_DIR / "fertilizer_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
TARGET_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
CATEGORICAL_ENCODERS_PATH = MODEL_DIR / "cat_encoders.pkl"

TARGET_COLUMN = "Recommended_Fertilizer"
FEATURE_ORDER = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
    "Region",
    "Fertilizer_Used_Last_Season",
    "Yield_Last_Season",
]

