import os
import sys
import json
import requests
import traceback
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config

# Load configuration
config = Config("./config.yaml")

# Get configuration from config.yaml
BASE_URL = "http://localhost:8080"  # Local FastAPI/Docker
AWS_ENDPOINT_NAME = config["sagemaker"]["endpoint_name"]
AWS_S3_BUCKET = config["s3"]["bucket"]
REGION = "eu-west-1"

# Ask user for test mode
def ask_test_mode():
    """Ask user if they want to test local or SageMaker endpoint."""
    print("=" * 80)
    print("Chronos Forecasting API - Inference Test")
    print("=" * 80)
    print("\nSelect test mode:\n")
    print("  1. Local server (http://localhost:8080)")
    print("  2. SageMaker endpoint")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            print("Invalid choice. Please enter 1 or 2.")

USE_SAGEMAKER = ask_test_mode()
# ===================


def log(msg, level="info"):
    """Minimal logger with simple prefixes."""
    levels = {
        "info": "[INFO]",
        "warn": "[WARN]",
        "error": "[ERROR]",
        "ok": "[OK]",
    }
    print(f"{levels.get(level, '[INFO]')} {msg}")


def plot_predictions(predictions_data, historical_data, request_id, series_name="Series"):
    """
    Create a beautiful plot of predictions with confidence intervals using seaborn.
    
    Args:
        predictions_data: Predictions JSON from API response
        historical_data: Original input data used for prediction
        request_id: Request identifier for saving plot
        series_name: Name of the time series for the title
    """
    try:
        # Set seaborn style like in notebook
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Parse predictions
        pred = predictions_data["predictions"]
        columns = pred["columns"]
        index = pred["index"]
        data = pred["data"]
        
        # Convert to DataFrame
        df_pred = pd.DataFrame(data, columns=columns)
        df_pred["timestamp"] = [datetime.fromtimestamp(ts[1] / 1000) for ts in index]
        
        # Parse historical data
        df_hist = pd.DataFrame(historical_data)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        
        # Create indices for plotting
        n_hist = len(df_hist)
        n_pred = len(df_pred)
        x_hist = range(n_hist)
        x_pred = range(n_hist, n_hist + n_pred)
        
        # Plot using seaborn (similar to notebook)
        sns.lineplot(x=x_hist, y=df_hist["ActivePower"], ax=ax, 
                     label="Historical data", linewidth=2)
        sns.lineplot(x=x_pred, y=df_pred["mean"], ax=ax, 
                     label="Prediction (median)", linewidth=2, linestyle='--')
        
        # Fill confidence intervals
        ax.fill_between(x_pred, df_pred["0.1"], df_pred["0.9"], 
                       alpha=0.3, label="80% interval")
        
        # Formatting like in notebook
        ax.set_title(f"{series_name} - Chronos Prediction", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"forecast_{series_name.lower().replace(' ', '_')}_{request_id[:8]}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        log(f"Plot saved to: {plot_filename}", "ok")
        
        plt.close()
        
    except Exception as e:
        log(f"Failed to create plot: {e}", "error")
        traceback.print_exc()


def plot_all_series_comparison(all_results):
    """
    Create a comparison plot of all series predictions (like in notebook).
    
    Args:
        all_results: Dictionary with series names as keys and (historical, predictions, request_id) as values
    """
    try:
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        n_series = len(all_results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Comparison of Chronos Predictions on Different Time Series', 
                     fontsize=16, fontweight='bold', y=0.995)
        axes = axes.flatten()
        
        for idx, (series_name, (hist_data, pred_data, _)) in enumerate(all_results.items()):
            if idx >= 6:  # Maximum 6 subplots
                break
                
            ax = axes[idx]
            
            # Parse data
            df_hist = pd.DataFrame(hist_data)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            
            pred = pred_data["predictions"]
            df_pred = pd.DataFrame(pred["data"], columns=pred["columns"])
            
            # Create indices
            n_hist = len(df_hist)
            n_pred = len(df_pred)
            x_hist = range(n_hist)
            x_pred = range(n_hist, n_hist + n_pred)
            
            # Plot
            sns.lineplot(x=x_hist, y=df_hist["ActivePower"], ax=ax,
                        label="Historical", linewidth=1.5)
            sns.lineplot(x=x_pred, y=df_pred["mean"], ax=ax,
                        label="Prediction", linewidth=1.5, linestyle='--')
            ax.fill_between(x_pred, df_pred["0.1"], df_pred["0.9"], alpha=0.3)
            
            ax.set_title(series_name, fontweight='bold')
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(all_results), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_filename = "forecast_comparison_all_series.png"
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        log(f"Comparison plot saved to: {comparison_filename}", "ok")
        
        plt.show()
        
    except Exception as e:
        log(f"Failed to create comparison plot: {e}", "error")
        traceback.print_exc()


def generate_payload(values=None, series_name="series_1"):
    """Generate time series data formatted for Chronos inference.
    
    Args:
        values: Optional numpy array with values. If None, generates default series.
        series_name: Name/ID for the time series
    """
    if values is None:
        # Default simple series
        now = datetime.now()
        timestamps = [
            (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(48)
        ][::-1]
        data = [
            {"item_id": series_name, "timestamp": ts, "ActivePower": float(i) + 0.5}
            for i, ts in enumerate(timestamps)
        ]
    else:
        # Custom series from numpy array
        n_points = len(values)
        now = datetime.now()
        timestamps = [
            (now - timedelta(hours=n_points-i-1)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(n_points)
        ]
        data = [
            {"item_id": series_name, "timestamp": ts, "ActivePower": float(val)}
            for ts, val in zip(timestamps, values)
        ]
    
    return data


def generate_synthetic_series():
    """Generate synthetic time series matching those in test_chornos.ipynb."""
    import numpy as np
    
    n_points = 100  # historical points
    
    # 1. Linear series
    x_linear = np.arange(n_points)
    y_linear = 2 * x_linear + 10 + np.random.normal(0, 5, n_points)
    
    # 2. Sinusoidal series
    x_sin = np.arange(n_points)
    y_sin = 50 * np.sin(2 * np.pi * x_sin / 20) + 100 + np.random.normal(0, 5, n_points)
    
    # 3. Sawtooth series
    x_saw = np.arange(n_points)
    y_saw = 30 * (x_saw % 15) / 15 + 50 + np.random.normal(0, 3, n_points)
    
    # 4. Exponential series
    x_exp = np.arange(n_points)
    y_exp = 10 * np.exp(0.02 * x_exp) + np.random.normal(0, 5, n_points)
    
    # 5. Multiple seasonality
    x_multi = np.arange(n_points)
    trend = 0.5 * x_multi
    seasonal1 = 20 * np.sin(2 * np.pi * x_multi / 12)
    seasonal2 = 10 * np.sin(2 * np.pi * x_multi / 30)
    noise = np.random.normal(0, 5, n_points)
    y_multi = trend + seasonal1 + seasonal2 + 100 + noise
    
    return {
        'Linear': y_linear,
        'Sinusoidal': y_sin,
        'Sawtooth': y_saw,
        'Exponential': y_exp,
        'Multiple Seasonality': y_multi
    }


def test_ping():
    """Check health endpoint /ping."""
    url = (
        f"{BASE_URL}/ping"
        if not USE_SAGEMAKER
        else f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{AWS_ENDPOINT_NAME}/ping"
    )
    log(f"Testing health endpoint: {url}", "info")

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            log(f"Health OK → {response.text}", "ok")
        else:
            log(f"Unexpected status code {response.status_code}: {response.text}", "warn")
    except Exception as e:
        log(f"Health check failed: {e}", "error")


def test_model_info():
    """Test the /model/info endpoint (local only)."""
    if USE_SAGEMAKER:
        log("Skipping /model/info (not available on SageMaker)", "info")
        return
    
    url = f"{BASE_URL}/model/info"
    log(f"Testing model info endpoint: {url}", "info")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            info = response.json()
            log("Model Info Retrieved ✅", "ok")
            print(f"  • Prediction Length: {info.get('prediction_length')}")
            print(f"  • Target Variable: {info.get('target')}")
            print(f"  • Eval Metric: {info.get('eval_metric')}")
        else:
            log(f"Model info failed [{response.status_code}] → {response.text}", "error")
    except Exception as e:
        log(f"Model info request failed: {e}", "error")


def test_invocation():
    """Test the inference endpoint with multiple synthetic series."""
    print("\n" + "="*80)
    print("TESTING WITH SYNTHETIC TIME SERIES")
    print("="*80)
    print("\nGenerating synthetic time series (matching test_chornos.ipynb)...\n")
    
    # Generate all synthetic series
    synthetic_series = generate_synthetic_series()
    all_results = {}
    
    for series_name, series_values in synthetic_series.items():
        print(f"\n{'─'*80}")
        print(f"Testing {series_name} Time Series")
        print(f"{'─'*80}")
        
        # Generate payload for this series
        payload = generate_payload(values=series_values, series_name=series_name)
        
        try:
            if USE_SAGEMAKER:
                import boto3
                runtime = boto3.client("sagemaker-runtime", region_name=REGION)
                
                log(f"Invoking SageMaker endpoint: {AWS_ENDPOINT_NAME}", "info")
                response = runtime.invoke_endpoint(
                    EndpointName=AWS_ENDPOINT_NAME,
                    ContentType="application/json",
                    Body=json.dumps(payload),
                )
                status = response["ResponseMetadata"]["HTTPStatusCode"]
                body = response["Body"].read().decode("utf-8")
            
            else:
                url = f"{BASE_URL}/invocations"
                log(f"Invoking local endpoint: {url}", "info")
                headers = {"Content-Type": "application/json"}
                res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
                status, body = res.status_code, res.text
            
            if status == 200:
                log(f"✅ {series_name} prediction successful!", "ok")
                result = json.loads(body)
                
                # Print summary
                print(f"  • Predictions: {result['metadata']['num_predictions']} steps")
                print(f"  • Request ID: {result['metadata']['request_id'][:8]}...")
                
                # Store results for comparison plot
                all_results[series_name] = (payload, result, result['metadata']['request_id'])
                
                # Create individual visualization
                log(f"Creating {series_name} plot...", "info")
                plot_predictions(result, payload, result['metadata']['request_id'], series_name)
            else:
                log(f"Inference failed [{status}] → {body}", "error")
        
        except requests.exceptions.RequestException as e:
            log(f"Network or connection error for {series_name}: {e}", "error")
        except Exception as e:
            log(f"Unexpected error during {series_name} invocation: {e}", "error")
            traceback.print_exc(file=sys.stdout)
    
    # Create comparison plot of all series
    if all_results:
        print("\n" + "="*80)
        print("Creating comparison plot of all series...")
        print("="*80)
        plot_all_series_comparison(all_results)
        
        print("\n" + "="*80)
        print(f"✅ All tests completed! Generated {len(all_results)} predictions.")
        print("="*80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Endpoint: {AWS_ENDPOINT_NAME if USE_SAGEMAKER else BASE_URL}")
    print(f"  • Mode: {'SageMaker' if USE_SAGEMAKER else 'Local'}")
    print(f"  • Region: {REGION}")
    print(f"  • S3 Bucket: {AWS_S3_BUCKET}")
    print("=" * 80 + "\n")
    
    if not USE_SAGEMAKER:
        print("Warning: Make sure the local server is running:")
        print("   Option 1: cd src/deployment && python serve.py")
        print("   Option 2: docker run -p 8080:8080 chronos-api")
        print()
    
    test_ping()
    print("-" * 80)
    test_model_info()
    print("-" * 80)
    test_invocation()
    print("-" * 80)
