import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocess import load_data, clean_data
from src.train_model import train_model

# Load and prepare data
print("📊 Loading data...")
df = load_data('data/survey.csv')

if df is None:
    print("❌ Failed to load data")
    exit(1)

print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Clean data
print("🧹 Cleaning data...")
df_cleaned = clean_data(df)

if df_cleaned is None:
    print("❌ Failed to clean data")
    exit(1)

print(f"✓ Data cleaned: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

# Train model
print("🤖 Training model...")
model, accuracy = train_model(df_cleaned)

if model is None:
    print("❌ Failed to train model")
    exit(1)

print(f"✅ Model trained successfully!")
print(f"📈 Model Accuracy: {accuracy:.4f}")
