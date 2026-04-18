import numpy as np
from tensorflow.keras.models import load_model

def predict(model_path, input_data):
    model = load_model(model_path)

    prediction = model.predict(input_data)
    return prediction


# Example usage
if __name__ == "__main__":
    sample = np.random.rand(1, 10, 12)  # adjust shape
    pred = predict("model.h5", sample)
    print("Prediction:", pred)