import numpy as np
from bayes_opt import BayesianOptimization
from gmdh_main import GMDH
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
def generate_data():
    np.random.seed(42)
    X = np.random.rand(200, 2)
    y = 3 * X[:, 0]**2 + 2 * X[:, 1] + np.random.randn(200) * 0.1
    return X, y

# Objective function for Bayesian Optimization
def objective_function(max_layers, threshold, validation_split):
    max_layers = int(max_layers)
    X, y = generate_data()
    
    model = GMDH(max_layers=max_layers, threshold=threshold, validation_split=validation_split)
    model.fit(X, y)
    predictions = model.predict(X)
    
    if predictions.shape[0] != y.shape[0]:
        print("⚠️ Prediction shape mismatch. Returning high error.")
        return -np.inf
    
    mse = mean_squared_error(y, predictions)
    return -mse  # Negative because Bayesian Optimization maximizes

# Run Bayesian Optimization
def optimize_gmdh():
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds={
            "max_layers": (1, 5),
            "threshold": (0.001, 0.05),
            "validation_split": (0.1, 0.3)
        },
        random_state=42,
    )

    optimizer.maximize(init_points=5, n_iter=20)
    return optimizer.max["params"]

# Train the best GMDH model based on optimization results
def train_best_gmdh():
    best_params = optimize_gmdh()
    print(f"✅ Best Hyperparameters: {best_params}")

    X, y = generate_data()
    model = GMDH(
        max_layers=int(best_params["max_layers"]),
        threshold=best_params["threshold"],
        validation_split=best_params["validation_split"]
    )
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    mse = mean_squared_error(y, predictions)
    print(f"🚀 Optimized Model MSE: {mse}")
    return model