import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocess import load_data, preprocess_data, scale_data, create_sequences
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = load_data('data/bike.csv')

# Preprocess
df = preprocess_data(df)

# Scale
scaled_data, scaler = scale_data(df)

# Create sequences
X, y = create_sequences(scaled_data, seq_length=10)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = build_model(input_shape=X_train.shape[1:])

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model.keras")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Model Loss")
plt.show()