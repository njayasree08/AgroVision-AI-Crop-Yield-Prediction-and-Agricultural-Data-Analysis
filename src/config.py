"""Central configuration: paths, column names, seeds and valid value ranges."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"
DATA_PATH = DATA_DIR / "agricultural_data.csv"

RANDOM_SEED = 42
N_RECORDS = 700
TEST_SIZE = 0.2

# ----------------------------------------------------------------- columns
ID_COLUMN = "Record_ID"
OPTIONAL_COLUMNS = ["Record_ID", "State", "District"]  # kept for display only

NUMERIC_FEATURES = [
    "Area_Hectares",
    "Annual_Rainfall_mm",
    "Average_Temperature_C",
    "Soil_pH",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Fertilizer_Usage",
    "Previous_Year_Yield",
]
CATEGORICAL_FEATURES = ["Crop_Type", "Season", "Soil_Type", "Irrigation_Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REG_TARGET = "Crop_Yield"
CLS_TARGET = "Yield_Category"
CLASS_ORDER = ["Low", "Medium", "High"]
REQUIRED_COLUMNS = FEATURES + [REG_TARGET]

# Clustering ignores Previous_Year_Yield so that groups are discovered from
# conditions (climate, soil, inputs, crop) rather than from yield information.
CLUSTER_DEFAULT_FEATURES = [f for f in FEATURES if f != "Previous_Year_Yield"]

# ------------------------------------------------------- validity ranges
# Values outside these ranges are physically implausible and are treated as
# data-entry errors (set to missing, then imputed inside the model pipeline).
DOMAIN_BOUNDS = {
    "Area_Hectares": (0.1, 500.0),
    "Annual_Rainfall_mm": (200.0, 3000.0),
    "Average_Temperature_C": (5.0, 45.0),
    "Soil_pH": (3.5, 9.5),
    "Nitrogen": (0.0, 600.0),
    "Phosphorus": (0.0, 200.0),
    "Potassium": (0.0, 600.0),
    "Fertilizer_Usage": (0.0, 500.0),
    "Previous_Year_Yield": (0.0, 15.0),
    "Crop_Yield": (0.0, 15.0),
}

CATEGORY_SYNONYMS = {
    "tubewell": "Tube Well",
    "tube-well": "Tube Well",
    "tube well": "Tube Well",
    "rain fed": "Rainfed",
    "rain-fed": "Rainfed",
}

# ------------------------------------------------------------- upload limits
MIN_ROWS = 50
MAX_UPLOAD_MB = 10
MAX_UPLOAD_ROWS = 50_000
