from auto_gmdh.nas import optimize_gmdh

best_params = optimize_gmdh()
print("Best GMDH Hyperparameters:", best_params)