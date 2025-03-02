import numpy as np
import random
import torch
from gmdh_nas.gmdh import GMDH
from sklearn.metrics import mean_squared_error

# Fix randomness i added this as rerunning the same give different mean squared errors from the cross-validation
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
X = np.random.rand(200, 2)
y = 3 * X[:, 0]**2 + 2 * X[:, 1] + np.random.randn(200) * 0.1  # Adds slight noise

# Train improved GMDH model
model = GMDH(max_layers=5, threshold=0.001)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
print("Improved MSE:", mse)
