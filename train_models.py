"""Train, evaluate and save all models.   Usage: python train_models.py

Reads data/agricultural_data.csv (generating it if missing), prints the three
comparison tables, saves pipelines to models/ and result tables to outputs/.
"""
import pandas as pd

from src.data_loader import load_dataset
from src.pipeline import export_tables, run_full_analysis, train_and_save_all
from src.preprocessing import prepare_dataset

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main() -> None:
    df, thresholds, report = prepare_dataset(load_dataset())
    print(f"Records after cleaning: {report['rows_out']}  |  yield thresholds (t/ha): {thresholds}")
    analysis = run_full_analysis(df)
    saved = train_and_save_all(df, thresholds, analysis=analysis)
    export_tables(analysis, df)

    print("\n=== Classification (test set) ===")
    print(analysis["classification"]["table"].round(4).to_string(index=False))
    print("\n=== Regression (test set) ===")
    print(analysis["regression"]["table"].round(4).to_string(index=False))
    print("\n=== Agglomerative clustering (ward) ===")
    print(analysis["clustering"]["table"].round(4).to_string(index=False))
    print("\nSummary:")
    for line in analysis["summary"]:
        print(" -", line.replace("**", ""))
    print("\nSaved models:", saved, "-> models/   |  tables -> outputs/")


if __name__ == "__main__":
    main()
