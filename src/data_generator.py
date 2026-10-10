"""Synthetic agricultural dataset generator (700 records, fixed random seed).

IMPORTANT: every value produced here is SYNTHETIC. The relationships are
plausible, simplified agronomic rules written for teaching purposes; they are
not field measurements and must not be used for real farming decisions.

Run from the project root:   python generate_data.py
"""
import numpy as np
import pandas as pd

from .config import DATA_DIR, DATA_PATH, N_RECORDS, RANDOM_SEED

# ----------------------------------------------------------------- crops
# base = yield potential in tonnes/hectare under good conditions.
CROPS = {
    "Rice":      dict(base=4.4, rain_opt=1300, rain_w=600, temp_opt=27, temp_w=6, ph_opt=6.2, fert_ref=120,
                      seasons={"Kharif": 0.70, "Rabi": 0.10, "Summer": 0.20}),
    "Wheat":     dict(base=4.2, rain_opt=700, rain_w=450, temp_opt=20, temp_w=5, ph_opt=6.8, fert_ref=130,
                      seasons={"Rabi": 1.0}),
    "Maize":     dict(base=4.8, rain_opt=900, rain_w=450, temp_opt=25, temp_w=6, ph_opt=6.5, fert_ref=140,
                      seasons={"Kharif": 0.60, "Rabi": 0.25, "Summer": 0.15}),
    "Cotton":    dict(base=2.3, rain_opt=850, rain_w=450, temp_opt=28, temp_w=6, ph_opt=7.0, fert_ref=110,
                      seasons={"Kharif": 1.0}),
    "Soybean":   dict(base=2.0, rain_opt=950, rain_w=450, temp_opt=26, temp_w=5, ph_opt=6.5, fert_ref=50,
                      seasons={"Kharif": 1.0}),
    "Groundnut": dict(base=2.4, rain_opt=800, rain_w=450, temp_opt=27, temp_w=6, ph_opt=6.3, fert_ref=45,
                      seasons={"Kharif": 0.55, "Rabi": 0.25, "Summer": 0.20}),
    "Bajra":     dict(base=2.1, rain_opt=600, rain_w=400, temp_opt=29, temp_w=6, ph_opt=7.2, fert_ref=60,
                      seasons={"Kharif": 0.80, "Summer": 0.20}),
    "Chickpea":  dict(base=1.7, rain_opt=650, rain_w=400, temp_opt=21, temp_w=5, ph_opt=7.0, fert_ref=40,
                      seasons={"Rabi": 1.0}),
}
CROP_WEIGHTS = [0.18, 0.17, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08]

STATES = {
    "Punjab":         dict(districts=["Ludhiana", "Amritsar", "Patiala"], rain=650, temp_off=-1.5, soils=["Alluvial", "Loamy", "Sandy"]),
    "Haryana":        dict(districts=["Karnal", "Hisar", "Rohtak"], rain=600, temp_off=-1.0, soils=["Alluvial", "Sandy", "Loamy"]),
    "Uttar Pradesh":  dict(districts=["Lucknow", "Meerut", "Varanasi"], rain=950, temp_off=0.0, soils=["Alluvial", "Loamy", "Clay"]),
    "Maharashtra":    dict(districts=["Pune", "Nagpur", "Nashik"], rain=900, temp_off=0.5, soils=["Black", "Red", "Clay"]),
    "Gujarat":        dict(districts=["Rajkot", "Surat", "Anand"], rain=800, temp_off=1.5, soils=["Black", "Sandy", "Alluvial"]),
    "Karnataka":      dict(districts=["Mysuru", "Belagavi", "Dharwad"], rain=1150, temp_off=0.0, soils=["Red", "Black", "Loamy"]),
    "Tamil Nadu":     dict(districts=["Thanjavur", "Madurai", "Coimbatore"], rain=950, temp_off=1.5, soils=["Red", "Alluvial", "Clay"]),
    "Madhya Pradesh": dict(districts=["Indore", "Bhopal", "Jabalpur"], rain=1050, temp_off=0.5, soils=["Black", "Loamy", "Red"]),
    "West Bengal":    dict(districts=["Bardhaman", "Nadia", "Hooghly"], rain=1500, temp_off=0.5, soils=["Alluvial", "Clay", "Loamy"]),
    "Andhra Pradesh": dict(districts=["Guntur", "Kurnool", "Krishna"], rain=950, temp_off=1.5, soils=["Red", "Black", "Alluvial"]),
}

SEASON_TEMP = {"Kharif": 27.5, "Rabi": 20.5, "Summer": 31.5}

# soil: (N, P, K means, mean pH, suitability multiplier)
SOILS = {
    "Alluvial": dict(n=220, p=32, k=190, ph=7.0, mult=1.03),
    "Black":    dict(n=200, p=24, k=280, ph=7.8, mult=1.00),
    "Red":      dict(n=160, p=20, k=140, ph=5.9, mult=0.95),
    "Loamy":    dict(n=240, p=36, k=210, ph=6.8, mult=1.05),
    "Sandy":    dict(n=130, p=16, k=110, ph=6.5, mult=0.90),
    "Clay":     dict(n=210, p=28, k=240, ph=7.2, mult=0.97),
}
SOIL_CROP_BONUS = {
    ("Cotton", "Black"): 0.06, ("Groundnut", "Sandy"): 0.05, ("Groundnut", "Red"): 0.04,
    ("Rice", "Clay"): 0.04, ("Rice", "Alluvial"): 0.04, ("Wheat", "Loamy"): 0.03,
    ("Wheat", "Alluvial"): 0.03, ("Soybean", "Black"): 0.04, ("Chickpea", "Black"): 0.04,
}

# extra water supplied by irrigation (mm) and an efficiency multiplier
IRRIGATION = {
    "Canal":      dict(water=200, eff=1.00),
    "Drip":       dict(water=250, eff=1.04),
    "Sprinkler":  dict(water=200, eff=1.02),
    "Tube Well":  dict(water=220, eff=1.00),
    "Rainfed":    dict(water=0,   eff=0.97),
}
IRRIGATION_NAMES = list(IRRIGATION)
STATE_IRRIGATION_P = {  # canal, drip, sprinkler, tube well, rainfed
    "Punjab": [0.35, 0.05, 0.05, 0.50, 0.05], "Haryana": [0.35, 0.05, 0.10, 0.45, 0.05],
    "Uttar Pradesh": [0.25, 0.05, 0.05, 0.50, 0.15], "Maharashtra": [0.10, 0.25, 0.10, 0.20, 0.35],
    "Gujarat": [0.15, 0.25, 0.15, 0.25, 0.20], "Karnataka": [0.15, 0.20, 0.10, 0.20, 0.35],
    "Tamil Nadu": [0.30, 0.15, 0.05, 0.30, 0.20], "Madhya Pradesh": [0.15, 0.10, 0.10, 0.30, 0.35],
    "West Bengal": [0.20, 0.05, 0.05, 0.30, 0.40], "Andhra Pradesh": [0.30, 0.15, 0.05, 0.25, 0.25],
}


def _gauss(x: float, mu: float, width: float) -> float:
    return float(np.exp(-(((x - mu) / width) ** 2)))


def _yield_potential(crop, soil, irrigation, rain, temp, ph, n, p, k, fert, area) -> float:
    """Deterministic (noise-free) yield in tonnes/hectare from growing conditions."""
    spec = CROPS[crop]
    water = rain + IRRIGATION[irrigation]["water"]
    f_rain = 0.65 + 0.40 * _gauss(water, spec["rain_opt"], spec["rain_w"])
    f_temp = 0.70 + 0.32 * _gauss(temp, spec["temp_opt"], spec["temp_w"])
    f_ph = max(0.80, 1.0 - 0.07 * abs(ph - spec["ph_opt"]))
    f_soil = SOILS[soil]["mult"] + SOIL_CROP_BONUS.get((crop, soil), 0.0)
    f_nutr = (
        (0.85 + 0.15 * (1 - np.exp(-n / 120)))
        * (0.92 + 0.08 * (1 - np.exp(-p / 18)))
        * (0.93 + 0.07 * (1 - np.exp(-k / 100)))
    )
    ratio = fert / spec["fert_ref"]
    f_fert = 0.80 + 0.28 * (1 - np.exp(-ratio))
    if ratio > 2.0:  # over-application penalty
        f_fert -= min(0.12, 0.05 * (ratio - 2.0))
    f_irr = IRRIGATION[irrigation]["eff"]
    f_area = 1.02 - 0.01 * np.log1p(area)
    return float(spec["base"] * f_rain * f_temp * f_ph * f_soil * f_nutr * f_fert * f_irr * f_area)


def generate_clean_dataset(n: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate n synthetic records without any data-quality problems."""
    rng = np.random.default_rng(seed)
    crop_names = list(CROPS)
    state_names = list(STATES)
    district_factor = {
        (s, d): float(rng.normal(1.0, 0.12)) for s in state_names for d in STATES[s]["districts"]
    }

    rows = []
    for i in range(n):
        crop = str(rng.choice(crop_names, p=CROP_WEIGHTS))
        spec = CROPS[crop]
        state = state_names[int(rng.integers(len(state_names)))]
        st_spec = STATES[state]
        district = st_spec["districts"][int(rng.integers(len(st_spec["districts"])))]
        season = str(rng.choice(list(spec["seasons"]), p=list(spec["seasons"].values())))
        soil = str(rng.choice(st_spec["soils"], p=[0.5, 0.3, 0.2]))
        irrigation = str(rng.choice(IRRIGATION_NAMES, p=STATE_IRRIGATION_P[state]))

        rain = float(np.clip(st_spec["rain"] * district_factor[(state, district)] * rng.lognormal(0, 0.12), 250, 2600))
        temp = float(SEASON_TEMP[season] + st_spec["temp_off"] + rng.normal(0, 1.5))
        ph = float(np.clip(rng.normal(SOILS[soil]["ph"], 0.45), 4.5, 8.8))
        n_val = float(SOILS[soil]["n"] * rng.lognormal(0, 0.18))
        p_val = float(SOILS[soil]["p"] * rng.lognormal(0, 0.25))
        k_val = float(SOILS[soil]["k"] * rng.lognormal(0, 0.20))
        fert = float(np.clip(spec["fert_ref"] * rng.lognormal(0, 0.30), 10, 300))
        area = float(np.clip(rng.lognormal(np.log(3.0), 0.8), 0.3, 60))

        potential = _yield_potential(crop, soil, irrigation, rain, temp, ph, n_val, p_val, k_val, fert, area)
        crop_yield = potential * rng.lognormal(0, 0.09)
        # last year's yield: same farm conditions, independent weather/management noise
        previous = potential * rng.lognormal(0, 0.13)

        rows.append({
            "Record_ID": f"AGV-{i + 1:04d}",
            "State": state,
            "District": district,
            "Crop_Type": crop,
            "Season": season,
            "Area_Hectares": round(area, 2),
            "Annual_Rainfall_mm": round(rain, 1),
            "Average_Temperature_C": round(temp, 1),
            "Soil_Type": soil,
            "Soil_pH": round(ph, 2),
            "Nitrogen": round(n_val, 1),
            "Phosphorus": round(p_val, 1),
            "Potassium": round(k_val, 1),
            "Fertilizer_Usage": round(fert, 1),
            "Irrigation_Type": irrigation,
            "Previous_Year_Yield": round(previous, 2),
            "Crop_Yield": round(crop_yield, 2),
        })
    return pd.DataFrame(rows)


def inject_quality_issues(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Add realistic data-quality problems so the cleaning step has work to do.

    * ~1-2 % missing values in several feature columns (never in Crop_Yield)
    * inconsistent category spelling (case / extra spaces)
    * a few impossible values (data-entry errors) outside valid ranges
    """
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    n = len(out)

    missing_rates = {"Soil_pH": 0.02, "Potassium": 0.02, "Nitrogen": 0.015,
                     "Fertilizer_Usage": 0.02, "Previous_Year_Yield": 0.025, "Irrigation_Type": 0.01}
    for col, rate in missing_rates.items():
        idx = rng.choice(n, size=int(round(rate * n)), replace=False)
        out.loc[out.index[idx], col] = np.nan

    def respell(col, count, funcs):
        idx = rng.choice(n, size=count, replace=False)
        for j in idx:
            val = out.at[out.index[j], col]
            if isinstance(val, str):
                out.at[out.index[j], col] = funcs[int(rng.integers(len(funcs)))](val)

    respell("Crop_Type", 18, [str.lower, str.upper, lambda s: f" {s} ", lambda s: s + " "])
    respell("Season", 10, [str.lower, str.upper])
    respell("Soil_Type", 6, [str.upper, str.lower])
    respell("Irrigation_Type", 5, [str.lower, lambda s: s.replace(" ", "").lower()])

    bad = rng.choice(n, size=9, replace=False)
    for j in bad[:4]:
        out.at[out.index[j], "Annual_Rainfall_mm"] = float(rng.uniform(3600, 7000))
    for j in bad[4:7]:
        out.at[out.index[j], "Fertilizer_Usage"] = float(rng.uniform(900, 1500))
    for j in bad[7:]:
        out.at[out.index[j], "Nitrogen"] = float(rng.uniform(900, 1500))
    return out


def generate_dataset(n: int = N_RECORDS, seed: int = RANDOM_SEED, inject_issues: bool = True) -> pd.DataFrame:
    df = generate_clean_dataset(n, seed)
    return inject_quality_issues(df, seed) if inject_issues else df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} SYNTHETIC records to {DATA_PATH}")


if __name__ == "__main__":
    main()
