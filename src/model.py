from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input


def build_model(input_shape):
    model = Sequential()

    # Proper input layer (fixes warning)
    model.add(Input(shape=input_shape))

    # Flatten sequence data
    model.add(Flatten())

    # Hidden layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    # Output layer (regression)
    model.add(Dense(1))

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
