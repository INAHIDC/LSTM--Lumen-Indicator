import os
from dotenv import load_dotenv
from model_training import save_model
from ensemble import train_ensemble_models, ensemble_predictions
from data_preprocessing import (
    load_data,
    add_technical_indicators,
    normalize_data,
    create_sequences,
)
from walk_forward_validation import walk_forward_split
from prediction import evaluate_predictions, plot_predictions
import matplotlib.pyplot as plt


def main():

    load_dotenv()
    model_path = os.getenv("MODEL_PATH", "models/")
    data_path = os.getenv("DATA_PATH", "data/stellar.csv")
    results_path = os.getenv("RESULTS_PATH", "results/")
    epochs = int(os.getenv("EPOCHS", 50))
    batch_size = int(os.getenv("BATCH_SIZE", 64))
    n_splits = int(os.getenv("N_SPLITS", 5))
    n_models = int(os.getenv("N_MODELS", 3))
    seq_length = int(os.getenv("SEQ_LENGTH", 60))

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    data = load_data(data_path)
    if data is None:
        return

    data = add_technical_indicators(data)
    data = data.dropna()

    scaled_features, scaled_target, feature_scaler, target_scaler = normalize_data(data)
    if scaled_features is None or scaled_target is None:
        return

    X, y = create_sequences(scaled_features, scaled_target, seq_length)
    splits = walk_forward_split(X, y, n_splits=n_splits)
    all_rmse = []
    model_rmse = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        print(f"\nTraining and testing on fold {i + 1}...")
        input_shape = (X_train.shape[1], X_train.shape[2])

        models = train_ensemble_models(
            X_train,
            y_train,
            X_test,
            y_test,
            input_shape,
            n_models=n_models,
            epochs=epochs,
            batch_size=batch_size,
        )

        for idx, model in enumerate(models):
            model_name = f"ensemble_model_fold_{i + 1}_model_{idx + 1}"
            save_model(model, model_name, model_path)

            predictions = model.predict(X_test)
            predictions_rescaled, y_test_rescaled, rmse = evaluate_predictions(
                predictions, y_test, target_scaler
            )

            if rmse is not None:
                print(f"Fold {i + 1} Model {idx + 1} RMSE: {rmse}")
                all_rmse.append(rmse)
                model_rmse.append((rmse, model_name))

                with open(
                    os.path.join(
                        results_path, f"fold_{i + 1}_model_{idx + 1}_rmse.txt"
                    ),
                    "w",
                ) as f:
                    f.write(f"Fold {i + 1} Model {idx + 1} RMSE: {rmse}\n")

                plot_predictions(y_test_rescaled, predictions_rescaled)
                plt.savefig(
                    os.path.join(results_path, f"fold_{i + 1}_model_{idx + 1}_plot.png")
                )
                plt.close()
            else:
                print(f"Fold {i + 1} Model {idx + 1} RMSE could not be calculated.")

    if model_rmse:
        best_rmse, best_model_name = min(model_rmse, key=lambda x: x[0])
        print(f"\nBest Model: {best_model_name} with RMSE: {best_rmse}")

        with open(os.path.join(results_path, "best_model.txt"), "w") as f:
            f.write(f"{best_model_name}\n")
    else:
        print("No valid RMSE values were found.")


if __name__ == "__main__":
    main()
