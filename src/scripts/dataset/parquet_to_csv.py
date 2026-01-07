"""
Convierte archivo Parquet a CSV para visualización.

Usage:
    python src/scripts/dataset/parquet_to_csv.py
"""

import pandas as pd
from pathlib import Path

# Rutas
parquet_file = Path("./data/wind-power-forecasting/processed/Turbine_data_processed.csv")
csv_file = Path("./data/wind-power-forecasting/processed/Turbine_data_processed_readable.csv")

print(f"📥 Leyendo archivo Parquet: {parquet_file}")
df = pd.read_parquet(parquet_file)

print(f"✅ Cargado: {len(df)} filas, {len(df.columns)} columnas")
print(f"   Columnas: {list(df.columns)}")
print(f"\n📊 Primeras filas:")
print(df.head())

print(f"\n💾 Guardando como CSV: {csv_file}")
df.to_csv(csv_file, encoding='utf-8')

print(f"✅ ¡Listo! Ahora puedes abrir: {csv_file}")
