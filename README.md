#  Bike Demand Prediction (ANN)

##  Overview

This project predicts bike rental demand using historical time-series data to help optimize inventory management.

## Approach

 Preprocessed data and extracted date features
 Scaled data using MinMaxScaler
 Created time-series sequences (window = 10)
 Built an ANN model using TensorFlow/Keras

## Results

 Achieved low validation loss (~0.004)
 Model shows good performance with minimal overfitting

##  Key Insight

ANN works as a baseline, but models like LSTM can perform better for time-series data.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib

##  Conclusion

The model successfully predicts bike demand and can help improve operational efficiency.
