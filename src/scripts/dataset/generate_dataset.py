import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Path adjustments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import config and pipeline
from src.config import Config
from src.pipelines.pipes import preprocessing_pipeline

# ===== Load configuration =====
config = Config("./config.yaml")

raw_dataset_file = Path(config["dataset"]["base_file"])
processed_dataset_file = Path(config["dataset"]["processed_file"])
processed_dataset_file.parent.mkdir(parents=True, exist_ok=True)

pipeline_output_file = Path(config["dataset"]["pipeline_file"])
pipeline_output_file.parent.mkdir(parents=True, exist_ok=True)

metadata_output_file = Path(config["dataset"]["metadata_file"])
metadata_output_file.parent.mkdir(parents=True, exist_ok=True)

# ===== Load dataset =====
df = pd.read_csv(raw_dataset_file, parse_dates=[0], index_col=0)
print(f"Loaded dataset with shape: {df.shape}")

# ===== Filter only operational turbine =====
df = df[df["TurbineStatus"] == 2]
df = df[["ActivePower"]]
print(f"Filtered dataset shape: {df.shape}")

# ===== Apply preprocessing pipeline =====
clean_series = preprocessing_pipeline.fit_transform(df)
print(f"Preprocessed dataset shape: {clean_series.shape}")

# ===== Split dataset (before windowing) =====
n = len(clean_series)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

clean_series = clean_series.copy()
clean_series["split"] = "train"
clean_series.iloc[train_end:val_end, clean_series.columns.get_loc("split")] = "val"
clean_series.iloc[val_end:, clean_series.columns.get_loc("split")] = "test"

print(clean_series["split"].value_counts())

# ===== Save dataset as a single file =====
clean_series.to_parquet(processed_dataset_file)
print(f"Saved processed dataset to: {processed_dataset_file}")

# ===== Save pipeline (including scaler) =====
joblib.dump(preprocessing_pipeline, pipeline_output_file)
print(f"Saved preprocessing pipeline to: {pipeline_output_file}")

# ===== Save metadata =====
metadata = {
    "input_steps": 672,
    "output_steps": 48,
    "train_size": int(train_end),
    "val_size": int(val_end - train_end),
    "test_size": int(n - val_end),
    "dataset_file": str(processed_dataset_file),
    "pipeline_file": str(pipeline_output_file),
}
with open(metadata_output_file, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Saved metadata to: {metadata_output_file}")

print("\n🎉 Dataset creation COMPLETE!")
