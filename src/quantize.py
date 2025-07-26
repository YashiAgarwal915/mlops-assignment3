import joblib
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("--- Quantization Script Started ---")


model_path = 'model/sklearn_model.joblib'
sklearn_model = joblib.load(model_path)

housing = fetch_california_housing()
_, X_test, _, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)


coef = sklearn_model.coef_
intercept = sklearn_model.intercept_
unquant_params = {'coef': coef, 'intercept': intercept}
joblib.dump(unquant_params, 'model/unquant_params.joblib')
print("Unquantized parameters saved.")


def quantize(params):

    scale = (np.max(params) - np.min(params)) / 255.0
    

    if scale == 0:
        scale = 1e-8 
    
    zero_point = -np.min(params) / scale
    quantized_params = np.round((params / scale) + zero_point).astype(np.uint8)
    return quantized_params, scale, zero_point

quantized_coef, scale_coef, zp_coef = quantize(coef)
quantized_intercept, scale_intercept, zp_intercept = quantize(np.array([intercept]))

quant_params = {
    'quantized_coef': quantized_coef, 'scale_coef': scale_coef, 'zp_coef': zp_coef,
    'quantized_intercept': quantized_intercept, 'scale_intercept': scale_intercept, 'zp_intercept': zp_intercept
}
joblib.dump(quant_params, 'model/quant_params.joblib')
print("Quantized parameters saved.")


def dequantize(quantized_params, scale, zero_point):
    return (quantized_params.astype(np.float32) - zero_point) * scale

dequantized_coef = dequantize(quantized_coef, scale_coef, zp_coef)
dequantized_intercept = dequantize(quantized_intercept, scale_intercept, zp_intercept)


class QuantizedLinearRegression(nn.Module):
    def __init__(self, input_features, output_features):
        super(QuantizedLinearRegression, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x)

pytorch_model = QuantizedLinearRegression(input_features=8, output_features=1)


pytorch_model.linear.weight.data = torch.from_numpy(dequantized_coef).float().unsqueeze(0)
pytorch_model.linear.bias.data = torch.from_numpy(dequantized_intercept).float()
print("PyTorch model created and weights set.")


sklearn_preds = sklearn_model.predict(X_test)
sklearn_r2 = r2_score(y_test, sklearn_preds)

pytorch_preds = pytorch_model(torch.from_numpy(X_test).float()).detach().numpy().flatten()
pytorch_r2 = r2_score(y_test, pytorch_preds)


unquant_size_kb = os.path.getsize('model/unquant_params.joblib') / 1024
quant_size_kb = os.path.getsize('model/quant_params.joblib') / 1024

print("\n--- Analysis Report ---")
print(f"| Metric         | Original Sklearn Model | Quantized Model |")
print(f"|----------------|------------------------|-----------------|")
print(f"| R2 Score       | {sklearn_r2:<22.4f} | {pytorch_r2:<15.4f} |")
print(f"| Model Size (KB)| {unquant_size_kb:<22.2f} | {quant_size_kb:<15.2f} |")
print("------------------------------------------------------------\n")
print("--- Quantization Script Finished ---")