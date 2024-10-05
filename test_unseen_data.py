import os
from dotenv import load_dotenv
from model_training import load_model_from_file
from data_preprocessing import (
    load_data,
    add_technical_indicators,
    normalize_data,
    create_sequences,
)
from prediction import evaluate_predictions, plot_predictions
import matplotlib.pyplot as plt


def main():

    load_dotenv()

    model_path = os.getenv("MODEL_PATH", "models/")
    unseen_data_path = os.getenv("UNSEEN_DATA_PATH", "data/unseen_stellar.csv")
    results_path = os.getenv("RESULTS_PATH", "results/")
    seq_length = int(os.getenv("SEQ_LENGTH", 60))

    best_model_file = os.path.join(results_path, "best_model.txt")
    if not os.path.exists(best_model_file):
        print("Best model file not found. Please run main.py first.")
        return

    with open(best_model_file, "r") as f:
        best_model_name = f.read().strip()

    print(f"best model: {best_model_name}")

    best_model = load_model_from_file(best_model_name, model_path)
    if best_model is None:
        print("cant load the best model.")
        return

    unseen_data = load_data(unseen_data_path)
    if unseen_data is None:
        return

    unseen_data = add_technical_indicators(unseen_data)
    unseen_data = unseen_data.dropna()

    scaled_features, scaled_target, feature_scaler, target_scaler = normalize_data(
        unseen_data
    )
    if scaled_features is None or scaled_target is None:
        return

    X_unseen, y_unseen = create_sequences(scaled_features, scaled_target, seq_length)

    predictions = best_model.predict(X_unseen)

    predictions_rescaled, y_unseen_rescaled, rmse = evaluate_predictions(
        predictions, y_unseen, target_scaler
    )

    if rmse is not None:
        print(f"Unseen Data RMSE: {rmse}")

        with open(os.path.join(results_path, f"unseen_data_rmse.txt"), "w") as f:
            f.write(f"Unseen Data RMSE: {rmse}\n")

        plot_predictions(y_unseen_rescaled, predictions_rescaled)
        plt.savefig(os.path.join(results_path, f"unseen_data_plot.png"))
        plt.close()
    else:
        print("RMSE could not be calculated for unseen data.")


if __name__ == "__main__":
    main()
