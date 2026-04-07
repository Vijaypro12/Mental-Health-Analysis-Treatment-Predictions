import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def load_model():
    """Load the trained model from the output directory"""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        model_path = project_root / 'output' / 'plots' / 'model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def encode_input(df):
    """Encode categorical features using LabelEncoder (same as training)"""
    try:
        df_encoded = df.copy()
        df_encoded = df_encoded.astype(str)  # Convert all to string first
        
        # Define categorical features (all except Age)
        categorical_features = [
            'Gender', 'Country', 'state', 'self_employed', 'family_history',
            'work_interfere', 'no_employees', 'remote_work', 'tech_company',
            'benefits', 'care_options', 'wellness_program', 'seek_help', 
            'anonymity', 'leave', 'mental_health_consequence', 
            'phys_health_consequence', 'coworkers', 'supervisor', 
            'mental_health_interview', 'phys_health_interview',
            'mental_vs_physical', 'obs_consequence'
        ]
        
        # Encode each categorical feature
        le = LabelEncoder()
        for col in categorical_features:
            if col in df_encoded.columns:
                try:
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                except Exception as col_err:
                    print(f"Error encoding column {col}: {col_err}")
                    raise
        
        # Make sure Age is numeric
        if 'Age' in df_encoded.columns:
            df_encoded['Age'] = pd.to_numeric(df_encoded['Age'], errors='coerce')
        
        return df_encoded
    except Exception as e:
        print(f"Error encoding input: {e}")
        return None


def predict(input_data):
    """Make prediction based on input data"""
    try:
        model = load_model()
        
        if model is None:
            raise ValueError("Model failed to load")

        df = pd.DataFrame([input_data])
        
        # Encode the input data
        df_encoded = encode_input(df)
        
        if df_encoded is None:
            raise ValueError("Failed to encode input data")
        
        # Make prediction
        prediction = model.predict(df_encoded)

        return prediction[0]

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None