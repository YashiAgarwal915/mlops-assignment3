import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

print("--- Training script started ---")

# --- 1. Loading the dataset ---
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# --- 2. Spliting the data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset loaded and split into training and testing sets.")

# --- 3. Train the Linear Regression model ---
print("Training the scikit-learn Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Saving the trained model ---
output_dir = 'model'
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'sklearn_model.joblib')

joblib.dump(model, model_path)

print(f"Scikit-learn model has been saved to: {model_path}")
print("--- Training script finished successfully ---")