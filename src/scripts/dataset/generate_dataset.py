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

# ===== FORECASTING PARAMETERS (optimized for wind turbine data) =====
CONTEXT_LENGTH = 672   # 14 días * 48 timesteps/day @ 30min (2 semanas de contexto)
PREDICTION_LENGTH = 48  # 1 día * 48 timesteps/day @ 30min (horizonte de 1 día)

print("\n" + "="*80)
print(" GENERATING AUTOGLUON-OPTIMIZED DATASET")
print("="*80)
print(f"\n Forecasting Configuration:")
print(f"   Context Length:     {CONTEXT_LENGTH} timesteps (14 days @ 30min freq)")
print(f"   Prediction Length:  {PREDICTION_LENGTH} timesteps (1 day @ 30min freq)")
print(f"   Frequency:          Every 30 minutes (30T)")

# ===== Load dataset =====
print(f"\n Loading raw data from: {raw_dataset_file}")
df = pd.read_csv(raw_dataset_file, parse_dates=[0], index_col=0)
print(f"    Loaded {len(df):,} rows × {len(df.columns)} columns")
print(f"    Date range: {df.index.min()} to {df.index.max()}")

# ===== Filter only operational turbine =====
print(f"\n Filtering operational data (TurbineStatus == 2)...")
df_filtered = df[df["TurbineStatus"] == 2]
df_filtered = df_filtered[["ActivePower"]]
print(f"    Kept {len(df_filtered):,} operational rows ({len(df_filtered)/len(df)*100:.1f}%)")

# ===== Apply preprocessing pipeline =====
print(f"\n Applying preprocessing pipeline...")
print(f"   Pipeline steps: {[step[0] for step in preprocessing_pipeline.steps]}")
clean_series = preprocessing_pipeline.fit_transform(df_filtered)
print(f"    Preprocessed to {len(clean_series):,} rows")

# ===== Fix NaN issues and ensure proper frequency =====
print(f"\n Ensuring regular 30min frequency...")

# Check current frequency
inferred_freq = pd.infer_freq(clean_series.index)
print(f"    Current frequency: {inferred_freq}")
# If frequency is not 30min or has gaps, resample explicitly
if inferred_freq != '30min' or clean_series['ActivePower'].isna().any():
    print(f"   Resampling to 30min frequency...")
    
    # Resample to 30min (this creates NaNs where data is missing)
    clean_series = clean_series.resample('30min').mean()
    
    # Count NaNs after resampling
    nans_after_resample = clean_series['ActivePower'].isna().sum()
    print(f"   After resample: {nans_after_resample} NaN values ({nans_after_resample/len(clean_series)*100:.1f}%)")
    
    if nans_after_resample > 0:
        # Interpolate using time-based method
        clean_series['ActivePower'] = clean_series['ActivePower'].interpolate(
            method='time',
            limit_direction='both'
        )
        
        # If still NaNs, forward fill then backward fill
        if clean_series['ActivePower'].isna().any():
            clean_series['ActivePower'] = clean_series['ActivePower'].ffill().bfill()
        
        nans_final = clean_series['ActivePower'].isna().sum()
        print(f"   After interpolation: {nans_final} NaN values ({nans_final/len(clean_series)*100:.1f}%)")
        
        if nans_final > 0:
            print(f"   ⚠️  Dropping {nans_final} remaining NaN rows")
            clean_series = clean_series.dropna()
else:
    print(f"    Frequency already correct: 30min")

print(f"    Final dataset: {len(clean_series):,} rows with freq=30min")

# ===== Validate data quality =====
print(f"\n Data Quality Check:")
print(f"   Missing values:  {clean_series['ActivePower'].isna().sum()} ({clean_series['ActivePower'].isna().sum()/len(clean_series)*100:.2f}%)")
print(f"   Min value:       {clean_series['ActivePower'].min():.2f}")
print(f"   Max value:       {clean_series['ActivePower'].max():.2f}")
print(f"   Mean:            {clean_series['ActivePower'].mean():.2f}")
print(f"   Std:             {clean_series['ActivePower'].std():.2f}")

# ===== Calculate optimal split for time series =====
n = len(clean_series)

# Minimum data requirement
min_required = CONTEXT_LENGTH + PREDICTION_LENGTH * 2  # Need at least 2 prediction windows
if n < min_required:
    raise ValueError(
        f" Dataset too small!\n"
        f"   Need at least: {min_required:,} rows (context + 2×prediction)\n"
        f"   Got:           {n:,} rows\n"
        f"   Missing:       {min_required - n:,} rows"
    )

# Split ratios optimized for time series
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Calculate split points
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

# Adjust for overlapping context (critical for AutoGluon!)
val_start = max(0, train_end - CONTEXT_LENGTH)
test_start = max(val_start, val_end - CONTEXT_LENGTH)

print(f"\n{'='*80}")
print(" TIME SERIES SPLIT STRATEGY (AutoGluon Optimized)")
print(f"{'='*80}")
print(f"\n📏 Split Ratios:")
print(f"   Train:      {train_ratio*100:.0f}%")
print(f"   Validation: {val_ratio*100:.0f}%")
print(f"   Test:       {test_ratio*100:.0f}%")

print(f"\n Data Splits (with overlapping context):")
print(f"   {'Split':<12} {'Range':<25} {'Total Rows':<12} {'Context Overlap'}")
print(f"   {'-'*12} {'-'*25} {'-'*12} {'-'*20}")
print(f"   {'Train':<12} [0:{train_end}]{' ':<10} {train_end:>10,}  {'-'}")
print(f"   {'Validation':<12} [{val_start}:{val_end}]{' ':<7} {val_end - val_start:>10,}  {train_end - val_start:,} from train")
print(f"   {'Test':<12} [{test_start}:{n}]{' ':<9} {n - test_start:>10,}  {val_end - test_start:,} from val")

# ===== Calculate training samples (what AutoGluon will actually use) =====
train_samples = max(0, (train_end - CONTEXT_LENGTH) // PREDICTION_LENGTH)
val_samples = max(0, (val_end - val_start - CONTEXT_LENGTH) // PREDICTION_LENGTH)
test_samples = max(0, (n - test_start - CONTEXT_LENGTH) // PREDICTION_LENGTH)

print(f"\n Training Batches (AutoGluon sliding windows):")
print(f"   Train:      {train_samples:>5} batches")
print(f"   Validation: {val_samples:>5} batches")
print(f"   Test:       {test_samples:>5} batches")
print(f"   Total:      {train_samples + val_samples + test_samples:>5} batches")

if train_samples < 5:
    print(f"\n  WARNING: Only {train_samples} training batches!")
    print(f"   Consider increasing data or reducing prediction_length")

# ===== Save full dataset as Parquet (for fast local analysis) =====
parquet_file = processed_dataset_file
clean_series_with_split = clean_series.copy()
clean_series_with_split["split"] = "train"
clean_series_with_split.iloc[val_start:val_end, clean_series_with_split.columns.get_loc("split")] = "val"
clean_series_with_split.iloc[test_start:, clean_series_with_split.columns.get_loc("split")] = "test"
clean_series_with_split.to_parquet(parquet_file)
print(f"\n💾 Saved full dataset (Parquet): {parquet_file}")

# ===== Save CSV splits (compatible with train_entrypoint.py) =====
split_dir = processed_dataset_file.parent / "split"
split_dir.mkdir(exist_ok=True)

base_name = processed_dataset_file.stem
train_csv = split_dir / f"{base_name}_train.csv"
val_csv = split_dir / f"{base_name}_val.csv"
test_csv = split_dir / f"{base_name}_test.csv"

print(f"\n Saving CSV splits (compatible with SageMaker training)...")

# IMPORTANT: Format must match train_entrypoint.py expectations:
# - Index column (unnamed) → will be renamed to "timestamp"
# - "ActivePower" column → target variable
# train_entrypoint.py expects: df.rename(columns={"Unnamed: 0": "timestamp"})

# Train split (pure training data)
train_data = clean_series.iloc[:train_end].copy()
train_data.to_csv(train_csv, index=True)  # Keep index as "Unnamed: 0"
print(f"    Train:      {len(train_data):>8,} rows → {train_csv.name}")

# Validation split (includes context from train)
val_data = clean_series.iloc[val_start:val_end].copy()
val_data.to_csv(val_csv, index=True)
print(f"    Validation: {len(val_data):>8,} rows → {val_csv.name}")
print(f"      (includes {train_end - val_start:,} rows of context from train)")

# Test split (includes context from val)
test_data = clean_series.iloc[test_start:].copy()
test_data.to_csv(test_csv, index=True)
print(f"    Test:       {len(test_data):>8,} rows → {test_csv.name}")
print(f"      (includes {val_end - test_start:,} rows of context from val)")

# ===== Save preprocessing pipeline =====
joblib.dump(preprocessing_pipeline, pipeline_output_file)
print(f"\n Saved preprocessing pipeline: {pipeline_output_file}")

# ===== Generate comprehensive metadata =====
metadata = {
    # Forecasting parameters
    "context_length": CONTEXT_LENGTH,
    "prediction_length": PREDICTION_LENGTH,
    "frequency": "30T",
    
    # Dataset statistics
    "total_rows": n,
    "train_rows": len(train_data),
    "val_rows": len(val_data),
    "test_rows": len(test_data),
    
    # Training batches
    "train_batches": train_samples,
    "val_batches": val_samples,
    "test_batches": test_samples,
    
    # Split boundaries
    "split_indices": {
        "train_start": 0,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": n
    },
    
    # Date ranges
    "date_ranges": {
        "train_start": str(clean_series.index[0]),
        "train_end": str(clean_series.index[train_end-1]),
        "val_start": str(clean_series.index[val_start]),
        "val_end": str(clean_series.index[val_end-1]),
        "test_start": str(clean_series.index[test_start]),
        "test_end": str(clean_series.index[-1])
    },
    
    # Data quality
    "data_quality": {
        "missing_values": int(clean_series['ActivePower'].isna().sum()),
        "min_value": float(clean_series['ActivePower'].min()),
        "max_value": float(clean_series['ActivePower'].max()),
        "mean_value": float(clean_series['ActivePower'].mean()),
        "std_value": float(clean_series['ActivePower'].std())
    },
    
    # File paths
    "files": {
        "raw_data": str(raw_dataset_file),
        "processed_parquet": str(parquet_file),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "pipeline": str(pipeline_output_file)
    },
    
    # CSV format (for reference)
    "csv_format": {
        "description": "Compatible with train_entrypoint.py",
        "index_column": "Unnamed: 0 (timestamp)",
        "target_column": "ActivePower",
        "note": "train_entrypoint.py adds 'item_id' and renames index to 'timestamp'"
    },
    
    # AutoGluon configuration hints
    "autogluon_config": {
        "prediction_length": PREDICTION_LENGTH,
        "eval_metric": "MASE",
        "recommended_time_limit": 300  # 5 minutes minimum
    }
}

with open(metadata_output_file, "w") as f:
    json.dump(metadata, f, indent=2)
print(f" Saved metadata: {metadata_output_file}")

# ===== Final summary =====
print("\n" + "="*80)
print(" DATASET GENERATION COMPLETE!")
print("="*80)

print("\n Generated Files:")
print(f"   1. Full dataset:  {parquet_file}")
print(f"   2. Train data:    {train_csv}")
print(f"   3. Val data:      {val_csv}")
print(f"   4. Test data:     {test_csv}")
print(f"   5. Pipeline:      {pipeline_output_file}")
print(f"   6. Metadata:      {metadata_output_file}")

print("\n Dataset Summary:")
print(f"   Total data:       {n:,} rows")
print(f"   Training:         {len(train_data):,} rows → {train_samples} batches")
print(f"   Validation:       {len(val_data):,} rows → {val_samples} batches")
print(f"   Test:             {len(test_data):,} rows → {test_samples} batches")

print("\n Next Steps:")
print("\n   1  Upload training data to S3:")
print(f"      aws s3 cp {train_csv} s3://chronos-presmanes/data/")

print("\n   2  Update config.yaml with training time:")
print(f"      training:")
print(f"        limit_time: 300  # minimum 5 minutes for {train_samples} batches")

print("\n   3  Launch SageMaker training:")
print(f"      python src/scripts/sagemaker/launch_training_job.py")
print(f"      → Select the uploaded training CSV interactively")
print(f"      → Model will train on {train_samples} batches")

print("\n   4  Compare models after training:")
print(f"      python src/scripts/compare_models.py \\")
print(f"        --test-data {test_csv} \\")
print(f"        --context-length {CONTEXT_LENGTH} \\")
print(f"        --prediction-length {PREDICTION_LENGTH}")

print("\n Training Optimization:")
print(f"   • Context: {CONTEXT_LENGTH} points → Model sees 14 days history @ 30min freq")
print(f"   • Horizon: {PREDICTION_LENGTH} points → Predicts next 1 day @ 30min freq")
print(f"   • Batches: {train_samples} → More batches = better learning")
print(f"   • Time:    Recommend {max(300, train_samples * 10)}s for {train_samples} batches")

if train_samples < 10:
    print("\n WARNING: Low number of training batches!")
    print("   Consider:")
    print(f"   • Collecting more data (current: {n:,} rows)")
    print(f"   • Reducing prediction_length (current: {PREDICTION_LENGTH})")
    print(f"   • Reducing context_length (current: {CONTEXT_LENGTH})")

print("\n" + "="*80 + "\n")
