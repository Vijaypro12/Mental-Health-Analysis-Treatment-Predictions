import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


def clean_data(df):
    try:
        # Drop unnecessary columns
        df = df.drop(['Timestamp', 'comments'], axis=1, errors='ignore')

        # Handle missing values
        df.fillna('Unknown', inplace=True)

        # Clean Gender column
        df['Gender'] = df['Gender'].str.lower()

        def clean_gender(g):
            if 'male' in g:
                return 'Male'
            elif 'female' in g:
                return 'Female'
            else:
                return 'Other'

        df['Gender'] = df['Gender'].apply(clean_gender)

        # Remove unrealistic ages
        df = df[(df['Age'] > 15) & (df['Age'] < 70)]

        return df

    except Exception as e:
        print("Error cleaning data:", e)
        return df