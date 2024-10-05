import numpy as np
from model_training import create_model


def train_ensemble_models(
    X_train, y_train, X_val, y_val, input_shape, n_models=3, epochs=50, batch_size=64
):
   
    models = []
    for i in range(n_models):
        print(f"Training ensemble model {i + 1}/{n_models}...")
        model = create_model(input_shape)
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0,
        )
        models.append(model)
    return models


def ensemble_predictions(models, X_test):
    predictions = np.array([model.predict(X_test) for model in models])
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions
