import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path


def encode_data(df):
    """Encode categorical data using LabelEncoder"""
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    return df


def train_model(df):
    """Train the machine learning model"""
    try:
        df = encode_data(df)

        X = df.drop('treatment', axis=1)
        y = df['treatment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Model Accuracy:", accuracy)

        # Save model to correct path
        project_root = Path(__file__).parent.parent
        model_dir = project_root / 'output' / 'plots'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to {model_path}")

        return model, accuracy

    except Exception as e:
        print(f"Error training model: {e}")
        return None, None