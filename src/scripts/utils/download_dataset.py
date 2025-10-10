import kagglehub
import shutil
import os

path = kagglehub.dataset_download("theforcecoder/wind-power-forecasting")

dest_dir = "data/wind-power-forecasting"

os.makedirs(dest_dir, exist_ok=True)

shutil.copytree(path, dest_dir, dirs_exist_ok=True)

print("Dataset copied in:", os.path.abspath(dest_dir))
