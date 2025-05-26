import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import shap


df = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project/data/raw/train.csv"
)
df = df.drop(columns=["id"])
df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

Y = df["Calories"]
X = df.drop("Calories", axis=1)


# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(X, Y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

numeric_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI"]
categorical_cols = [col for col in X.columns if col not in numeric_cols]

x_train["BMI"] = x_train["Weight"] / (x_train["Height"] / 100) ** 2
x_test["BMI"] = x_test["Weight"] / (x_test["Height"] / 100) ** 2
x_cv["BMI"] = x_cv["Weight"] / (x_cv["Height"] / 100) ** 2


scaler = StandardScaler()
x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])
x_cv[numeric_cols] = scaler.transform(x_cv[numeric_cols])

train_shape = x_train.shape

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(32, activation="relu", input_shape=(train_shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),  # Single output for regression (calories)
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.MeanSquaredLogarithmicError(),
    metrics=["mae"],
)


history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
)

y_cv_pred = model.predict(x_cv)
cv_loss = tf.keras.losses.MeanSquaredLogarithmicError()(y_cv, y_cv_pred).numpy()
cv_mae = mean_absolute_error(y_cv, y_cv_pred)

print(f"Cross-validation MSLE Loss: {cv_loss}")
print(f"Cross-validation MAE: {cv_mae}")

y_test_pred = model.predict(x_test)
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test MAE: {test_mae}")
print(f"Test Loss: {test_loss}")

# Save the entire model
model.save("models/my_model")

# Save only the model's weights
model.save_weights("path/to/weights")


# Assuming you have a trained Keras model and input data X
explainer = shap.DeepExplainer(model, x_train)
shap_values = explainer.shap_values(x_test)

# Visualize feature importance for the first prediction
shap.summary_plot(shap_values, x_test)

# Calculate absolute errors using Pandas
errors = (y_cv - y_cv_pred.flatten()).abs()

# Make the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    y_cv,
    y_cv_pred,
    c=errors,
    cmap="coolwarm",  # Color map to show error size
    alpha=0.7,
    edgecolor="k",
)

# Reference line for perfect predictions
plt.plot(
    [y_cv.min(), y_cv.max()],
    [y_cv.min(), y_cv.max()],
    "r--",
    label="Perfect Prediction",
)

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label("Absolute Error (|True - Predicted|)")

# Labels and titles
plt.xlabel("True Calories")
plt.ylabel("Predicted Calories")
plt.title("Predicted vs True Calories with Error Color")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Step 1: Create a DataFrame
new_sample = pd.DataFrame(
    [
        {
            "Age": 25,
            "Height": 175,
            "Weight": 70,
            "Duration": 30,
            "Heart_Rate": 110,
            "Body_Temp": 37.0,
            "Sex_Male": 1,  # Because 'Sex_Female' was dropped by get_dummies
        }
    ]
)

# Step 2: Compute BMI
new_sample["BMI"] = new_sample["Weight"] / (new_sample["Height"] / 100) ** 2

# Step 3: Apply the same scaler to numeric columns
new_sample[numeric_cols] = scaler.transform(new_sample[numeric_cols])

# Step 4: Predict
new_prediction = model.predict(new_sample)
print(f"Predicted calories burned: {new_prediction[0][0]:.2f}")

residuals = y_cv - y_cv_pred.flatten()

plt.hist(residuals, bins=30, edgecolor="k")
plt.xlabel("Prediction Error (Calories)")
plt.title("Residuals Histogram (CV Set)")
plt.grid(True)
plt.show()
