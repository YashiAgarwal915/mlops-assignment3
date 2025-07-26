import joblib
import numpy as np
import os

print("--- Verification script started ---")

model_path = 'model/sklearn_model.joblib'

print(f"Looking for model at: {model_path}")
if not os.path.exists(model_path):
    print(f"FATAL: Model file not found at {model_path}. The Docker image is not correctly built.")
    exit(1)

try:
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Failed to load the model. Error: {e}")
    exit(1)

try:
    print("Performing a test prediction...")

    dummy_sample = np.random.rand(1, 8)
    prediction = model.predict(dummy_sample)
    print(f"Successfully made a test prediction: {prediction}")
except Exception as e:
    print(f"FATAL: Failed to make a prediction. Error: {e}")
    exit(1)

print("--- Verification script finished successfully ---")
