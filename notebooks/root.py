# main.py or in a Jupyter notebook
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import load_raw_data, split_data
from src.features import create_features, scale_features
from src.modeling.train import create_model, train_model, evaluate_model
from src.modeling.predict import predict_new_sample
from src.plots import plot_predictions_vs_actual, plot_residuals, plot_shap_summary


def main():
    # Load and split data
    df = load_raw_data()
    x_train, x_cv, x_test, y_train, y_cv, y_test = split_data(df)

    # Feature engineering
    x_train, x_cv, x_test = create_features(x_train, x_cv, x_test)
    x_train, x_cv, x_test, scaler = scale_features(x_train, x_cv, x_test)

    # Create and train model
    model = create_model(x_train.shape[1])
    history = train_model(model, x_train, y_train)

    # Evaluate model
    results = evaluate_model(model, x_cv, y_cv, x_test, y_test)

    # Save model
    model.save("models/my_model")

    # Create visualizations
    plot_predictions_vs_actual(y_cv, results["y_cv_pred"])
    plot_residuals(y_cv, results["y_cv_pred"])
    plot_shap_summary(model, x_train, x_test)

    # Make a prediction
    sample = {
        "Age": 25,
        "Height": 175,
        "Weight": 70,
        "Duration": 30,
        "Heart_Rate": 110,
        "Body_Temp": 37.0,
        "Sex_Male": 1,
    }
    prediction = predict_new_sample(model, scaler, sample)
    print(f"Predicted calories: {prediction:.2f}")


if __name__ == "__main__":
    main()
