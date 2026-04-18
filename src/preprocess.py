import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    print("Columns:", df.columns)

    # Convert date column
    df['dteday'] = pd.to_datetime(df['dteday'])

    # Extract features
    df['day'] = df['dteday'].dt.day
    df['month'] = df['dteday'].dt.month
    df['year'] = df['dteday'].dt.year

    # Drop unnecessary columns
    df = df.drop(['dteday', 'instant', 'casual', 'registered'], axis=1)

    return df


def scale_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler


def create_sequences(data, seq_length=10):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, -1])  # target = cnt

    return np.array(X), np.array(y)
    
          
  
    