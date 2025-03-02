import sys
import os

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics import mean_squared_error
from gmdh_main import GMDH
from nas import optimize_gmdh, generate_data

def main():
    print("🚀 Starting GMDH experiment...")

    # Generate dataset
    X, y = generate_data()
    print(f"✅ Dataset generated with shape: X={X.shape}, y={y.shape}")

    # Optimize GMDH hyperparameters
    print("🔍 Running Bayesian Optimization for hyperparameters...")
    best_params = optimize_gmdh()
    print(f"✅ Best Hyperparameters: {best_params}")

    # Train the model with optimized parameters
    print("🛠 Training GMDH model with optimized parameters...")
    model = GMDH(
        max_layers=int(best_params["max_layers"]),
        threshold=best_params["threshold"],
        validation_split=best_params["validation_split"]
    )

    model.fit(X, y)
    print("✅ Model training complete.")

    # Make predictions
    print("📊 Making predictions...")
    predictions = model.predict(X)

    # Evaluate performance
    mse = mean_squared_error(y, predictions)
    print(f"🎯 Final Model MSE: {mse}")

    # Save results (optional)
    np.save("predictions.npy", predictions)
    np.save("true_values.npy", y)
    print("💾 Predictions and true values saved.")

if __name__ == "__main__":
    main()
