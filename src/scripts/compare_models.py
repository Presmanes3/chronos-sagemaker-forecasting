"""
Compare SageMaker endpoint model vs local baseline Chronos model.

This script:
1. Loads test data from local CSV
2. Makes predictions using SageMaker endpoint (fine-tuned model)
3. Makes predictions using local Chronos baseline model
4. Compares both with professional metrics and plots

Usage:
    python src/scripts/compare_models.py --test-data data/wind-power-forecasting/processed/split/Turbine_data_processed_test.csv
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import boto3  # ✅ Changed from requests to boto3
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from chronos import BaseChronosPipeline
from autogluon.timeseries import TimeSeriesDataFrame
from src.config import Config

# Load configuration
config = Config("./config.yaml")


def load_test_data(csv_path: str, item_id_col: str = None, timestamp_col: str = None):
    """Load and prepare test data."""
    print(f"\n📥 Loading test data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Auto-detect columns
    if item_id_col is None:
        for col in ['item_id', 'ItemId', 'id']:
            if col in df.columns:
                item_id_col = col
                break
        if item_id_col is None:
            # ✅ Use 'Turbine_1' to match training data (see train_entrypoint.py)
            df['item_id'] = 'Turbine_1'
            item_id_col = 'item_id'
            print(f"   ⚠️ No item_id column found, using default: 'Turbine_1'")
    
    if timestamp_col is None:
        for col in ['timestamp', 'Timestamp', 'date', 'Unnamed: 0']:
            if col in df.columns:
                timestamp_col = col
                break
    
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
    
    return df, item_id_col, timestamp_col


def predict_with_sagemaker(data: list, endpoint_name: str, region: str = "eu-west-1", profile: str = None):
    """Make predictions using SageMaker endpoint with proper AWS authentication."""
    print(f"\n🔮 Predicting with SageMaker endpoint: {endpoint_name}")
    
    try:
        # Create boto3 session with authentication
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        
        # Use SageMaker Runtime client (handles authentication automatically)
        runtime = session.client('sagemaker-runtime')
        
        # ✅ The endpoint expects a plain JSON array of objects (see deployment/sample_input.json)
        # Each object must have: item_id, timestamp, and the target column (e.g., ActivePower)
        # NO wrapping needed - just send the list directly
        
        # ⚠️ DIAGNOSTIC: Try sending fewer data points first
        # AutoGluon might have issues with too many context points
        max_context = 100  # Start with 100 points for testing
        if len(data) > max_context:
            print(f"⚠️ Reducing context from {len(data)} to {max_context} points for testing")
            data_to_send = data[-max_context:]  # Use most recent points
        else:
            data_to_send = data
        
        payload = json.dumps(data_to_send)
        
        print(f"📤 Sending {len(data_to_send)} data points")
        print(f"   Sample (first): {data_to_send[0] if data_to_send else 'N/A'}")
        print(f"   Sample (last): {data_to_send[-1] if data_to_send else 'N/A'}")
        
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        print(f"✅ SageMaker prediction successful")
        return result
            
    except Exception as e:
        print(f"❌ SageMaker request failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_with_baseline(
    values: np.ndarray,
    model_path: str,
    prediction_length: int = 24
):
    """Make predictions using local baseline Chronos model."""
    print(f"\n🔮 Predicting with baseline Chronos model: {model_path}")
    
    try:
        # Load pipeline
        pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        # Predict
        context = torch.tensor(values, dtype=torch.float32)
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        
        # Extract median prediction
        median_pred = quantiles[0, :, 1].numpy()
        
        print(f"✅ Baseline prediction successful")
        return median_pred
        
    except Exception as e:
        print(f"❌ Baseline prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate regression metrics."""
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }


def plot_comparison(
    historical_data,
    sagemaker_pred,
    baseline_pred,
    actual_future,
    metrics_df,
    output_path="model_comparison.png"
):
    """Create professional comparison plots."""
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main comparison plot
    ax1 = fig.add_subplot(gs[0, :])
    
    n_hist = len(historical_data)
    n_pred = len(sagemaker_pred) if sagemaker_pred is not None else len(baseline_pred) if baseline_pred is not None else 0
    
    x_hist = range(n_hist)
    x_pred = range(n_hist, n_hist + n_pred)
    
    ax1.plot(x_hist, historical_data, label='Historical Data', linewidth=2, alpha=0.7)
    
    if actual_future is not None and len(actual_future) > 0:
        x_actual = range(n_hist, n_hist + len(actual_future))
        ax1.plot(x_actual, actual_future, label='Actual Future', linewidth=2, 
                color='black', linestyle='-', marker='o', markersize=3)
    
    if sagemaker_pred is not None and len(sagemaker_pred) > 0:
        ax1.plot(x_pred, sagemaker_pred, label='SageMaker (Fine-tuned)', 
                linewidth=2, linestyle='--', marker='s', markersize=4)
    
    if baseline_pred is not None and len(baseline_pred) > 0:
        ax1.plot(x_pred, baseline_pred, label='Baseline (Chronos)', 
                linewidth=2, linestyle='-.', marker='^', markersize=4)
    
    ax1.axvline(x=n_hist, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.set_title('Model Predictions Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Metrics table
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = []
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['model'],
            f"{row['RMSE']:.2f}",
            f"{row['MAE']:.2f}",
            f"{row['MAPE']:.2f}%",
            f"{row['R²']:.3f}"
        ])
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Model', 'RMSE', 'MAE', 'MAPE', 'R²'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.175, 0.175, 0.175, 0.175]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    colors = ['#E8F5E9', '#FFF3E0']
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[(i-1) % 2])
    
    ax2.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # Metrics bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    
    x_pos = np.arange(len(metrics_df))
    width = 0.25
    
    rmse_norm = metrics_df['RMSE'] / metrics_df['RMSE'].max()
    mae_norm = metrics_df['MAE'] / metrics_df['MAE'].max()
    mape_norm = metrics_df['MAPE'] / metrics_df['MAPE'].max()
    
    ax3.bar(x_pos - width, rmse_norm, width, label='RMSE (norm)', alpha=0.8)
    ax3.bar(x_pos, mae_norm, width, label='MAE (norm)', alpha=0.8)
    ax3.bar(x_pos + width, mape_norm, width, label='MAPE (norm)', alpha=0.8)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Normalized Error')
    ax3.set_title('Error Metrics Comparison (Normalized)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics_df['model'], rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Residuals plot (if actual data available)
    if actual_future is not None and len(actual_future) > 0:
        ax4 = fig.add_subplot(gs[2, 0])
        
        if sagemaker_pred is not None and len(sagemaker_pred) > 0:
            min_len = min(len(actual_future), len(sagemaker_pred))
            residuals_sm = actual_future[:min_len] - sagemaker_pred[:min_len]
            ax4.scatter(range(len(residuals_sm)), residuals_sm, 
                       label='SageMaker', alpha=0.6, s=50)
        
        if baseline_pred is not None and len(baseline_pred) > 0:
            min_len = min(len(actual_future), len(baseline_pred))
            residuals_bl = actual_future[:min_len] - baseline_pred[:min_len]
            ax4.scatter(range(len(residuals_bl)), residuals_bl, 
                       label='Baseline', alpha=0.6, s=50, marker='^')
        
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax4.set_title('Prediction Residuals', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Prediction Step')
        ax4.set_ylabel('Residual (Actual - Predicted)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Error distribution
        ax5 = fig.add_subplot(gs[2, 1])
        
        if sagemaker_pred is not None and len(sagemaker_pred) > 0:
            ax5.hist(residuals_sm, bins=20, alpha=0.6, label='SageMaker', edgecolor='black')
        
        if baseline_pred is not None and len(baseline_pred) > 0:
            ax5.hist(residuals_bl, bins=20, alpha=0.6, label='Baseline', edgecolor='black')
        
        ax5.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Residual Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.suptitle('SageMaker Fine-tuned vs Baseline Chronos - Performance Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Comparison plot saved: {output_path}")
    
    plt.show()


def main(args):
    print("\n" + "="*80)
    print("🔬 MODEL COMPARISON: SAGEMAKER vs BASELINE")
    print("="*80)
    
    # Load test data
    df, item_id_col, timestamp_col = load_test_data(
        args.test_data,
        args.item_id_column,
        args.timestamp_column
    )
    
    # Use first N points as context, rest for evaluation
    context_length = args.context_length
    prediction_length = args.prediction_length
    
    if len(df) < context_length + prediction_length:
        print(f"\n⚠️ Warning: Test data has only {len(df)} rows")
        print(f"   Requested: {context_length} context + {prediction_length} prediction")
        context_length = len(df) - prediction_length
        print(f"   Adjusted context_length to: {context_length}")
    
    # Prepare data
    target_col = args.target_column
    if target_col not in df.columns:
        print(f"\n❌ Target column '{target_col}' not found in data")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    historical_values = df[target_col].values[:context_length]
    actual_future = df[target_col].values[context_length:context_length + prediction_length]
    
    print(f"\n📊 Data split:")
    print(f"   Historical (context): {len(historical_values)} points")
    print(f"   Future (to predict):  {len(actual_future)} points")
    
    # Prepare SageMaker payload
    sagemaker_data = []
    # ✅ Get the actual item_id from the data (should be 'Turbine_1')
    item_id = str(df[item_id_col].iloc[0])
    print(f"   Using item_id: '{item_id}'")
    
    for idx in range(context_length):
        sagemaker_data.append({
            'item_id': item_id,  # Use the extracted item_id consistently
            'timestamp': df[timestamp_col].iloc[idx].strftime("%Y-%m-%d %H:%M:%S"),
            target_col: float(df[target_col].iloc[idx])
        })
    
    # Get predictions
    sagemaker_predictions = None
    baseline_predictions = None
    
    if not args.skip_sagemaker:
        sagemaker_result = predict_with_sagemaker(
            sagemaker_data,
            config["sagemaker"]["endpoint_name"],
            args.region,
            args.aws_profile  # ✅ Pass AWS profile
        )
        
        if sagemaker_result and 'predictions' in sagemaker_result:
            pred_data = sagemaker_result['predictions']['data']
            sagemaker_predictions = np.array([row[1] for row in pred_data])  # Extract 'mean' column
    
    if not args.skip_baseline:
        baseline_predictions = predict_with_baseline(
            historical_values,
            args.baseline_model_path,
            prediction_length
        )
    
    # Calculate metrics
    metrics_list = []
    
    if sagemaker_predictions is not None and len(sagemaker_predictions) > 0:
        sm_metrics = calculate_metrics(
            actual_future[:len(sagemaker_predictions)],
            sagemaker_predictions,
            'SageMaker (Fine-tuned)'
        )
        metrics_list.append(sm_metrics)
        
        print(f"\n📊 SageMaker Metrics:")
        for key, value in sm_metrics.items():
            if key != 'model':
                print(f"   {key}: {value:.4f}")
    
    if baseline_predictions is not None and len(baseline_predictions) > 0:
        bl_metrics = calculate_metrics(
            actual_future[:len(baseline_predictions)],
            baseline_predictions,
            'Baseline (Chronos)'
        )
        metrics_list.append(bl_metrics)
        
        print(f"\n📊 Baseline Metrics:")
        for key, value in bl_metrics.items():
            if key != 'model':
                print(f"   {key}: {value:.4f}")
    
    if len(metrics_list) >= 2:
        improvement = (
            (metrics_list[1]['RMSE'] - metrics_list[0]['RMSE']) 
            / metrics_list[1]['RMSE'] * 100
        )
        
        print(f"\n{'='*80}")
        if improvement > 0:
            print(f"✅ Fine-tuning IMPROVED RMSE by {improvement:.2f}%")
        else:
            print(f"⚠️ Fine-tuning DEGRADED RMSE by {abs(improvement):.2f}%")
        print(f"{'='*80}")
    
    # Create comparison plots
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        
        plot_comparison(
            historical_values,
            sagemaker_predictions,
            baseline_predictions,
            actual_future,
            metrics_df,
            args.output_plot
        )
        
        # Save metrics to JSON
        metrics_output = Path(args.output_plot).parent / "comparison_metrics.json"
        with open(metrics_output, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        print(f"💾 Metrics saved: {metrics_output}")
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SageMaker endpoint vs baseline Chronos model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--baseline-model-path",
        type=str,
        default="./models/base/chronos-bolt-tiny",
        help="Path to baseline Chronos model"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=672,
        help="Number of historical points to use as context (default: 672 = 14 days @ 30min)"
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=48,
        help="Number of future points to predict (default: 48 = 1 day @ 30min)"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="ActivePower",
        help="Target column name (default: ActivePower)"
    )
    parser.add_argument(
        "--item-id-column",
        type=str,
        default=None,
        help="Item ID column name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default=None,
        help="Timestamp column name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--skip-sagemaker",
        action="store_true",
        help="Skip SageMaker endpoint prediction"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline model prediction"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="eu-west-1",
        help="AWS region (default: eu-west-1)"
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile name (uses default credentials if not specified)"
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="model_comparison.png",
        help="Output plot filename (default: model_comparison.png)"
    )
    
    args = parser.parse_args()
    main(args)
