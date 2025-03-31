import kagglehub
import os
import shutil
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("oktayrdeki/heart-disease")

print("Path to dataset files:", path)

# Ensure the destination directory exists
dest_dir = Path("data/raw")
dest_dir.mkdir(parents=True, exist_ok=True)

# Locate the CSV file in the downloaded directory
source_file = None
for file in Path(path).glob("**/*.csv"):
    source_file = file
    break

if source_file:
    dest_file = dest_dir / "heart_disease.csv"
    shutil.copy(source_file, dest_file)
    print(f"Dataset copied to: {dest_file}")
else:
    print("No CSV file found in the downloaded dataset.")
