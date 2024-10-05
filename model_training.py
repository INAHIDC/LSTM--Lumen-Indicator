from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=64):
    model = create_model(input_shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    return model

def save_model(model, model_name, model_path='models/'):
    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save(os.path.join(model_path, f"{model_name}.keras"))
        print(f"Model saved as {model_path}{model_name}.keras")
    except Exception as e:
        print(f"Error saving model {model_name}: {e}")

def load_model_from_file(model_name, model_path='models/'):
    from tensorflow.keras.models import load_model
    try:
        model = load_model(os.path.join(model_path, f"{model_name}.keras"))
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None
