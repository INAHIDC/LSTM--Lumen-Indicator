import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def evaluate_predictions(predictions, y_test, target_scaler):
    try:
        predictions_rescaled = target_scaler.inverse_transform(predictions)
        y_test_rescaled = target_scaler.inverse_transform(y_test)
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
        return predictions_rescaled, y_test_rescaled, rmse
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None, None, None


def plot_predictions(y_test_scaled, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_scaled, label="Actual Prices", color="blue")
    plt.plot(predictions, label="Predicted Prices", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices")
    plt.show()
