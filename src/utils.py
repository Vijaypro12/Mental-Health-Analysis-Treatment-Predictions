import pickle
import pandas as pd


def load_model():
    with open('../outputs/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def predict(input_data):
    try:
        model = load_model()

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)

        return prediction[0]

    except Exception as e:
        print("Prediction error:", e)
        return None