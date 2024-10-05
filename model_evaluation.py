import numpy as np
from sklearn.metrics import mean_squared_error


def predict_and_evaluate(model, X_test, y_test, scaler):
    try:
        predictions = model.predict(X_test)
        if y_test.ndim == 2 and y_test.shape[1] > 1:
            y_test = y_test[:, 0].reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        y_test_scaled = scaler.inverse_transform(y_test)
        if predictions.shape != y_test_scaled.shape:
            min_length = min(predictions.shape[0], y_test_scaled.shape[0])
            predictions = predictions[:min_length]
            y_test_scaled = y_test_scaled[:min_length]
        rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
        return predictions, rmse
    except Exception as e:
        print(f"pre or eval error: {e}")
        return None, None
