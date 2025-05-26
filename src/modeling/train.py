import tensorflow as tf
from src.config import EPOCHS, BATCH_SIZE


def create_model(input_shape):
    """Create and compile the neural network model"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation="relu", input_shape=(input_shape,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        metrics=["mae"],
    )
    return model


def train_model(model, x_train, y_train):
    """Train the model"""
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return history


def evaluate_model(model, x_cv, y_cv, x_test, y_test):
    """Evaluate model on CV and test sets"""
    from sklearn.metrics import mean_absolute_error

    # CV evaluation
    y_cv_pred = model.predict(x_cv)
    cv_loss = tf.keras.losses.MeanSquaredLogarithmicError()(y_cv, y_cv_pred).numpy()
    cv_mae = mean_absolute_error(y_cv, y_cv_pred)

    # Test evaluation
    test_loss, test_mae = model.evaluate(x_test, y_test)

    return {
        "cv_loss": cv_loss,
        "cv_mae": cv_mae,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "y_cv_pred": y_cv_pred,
    }
