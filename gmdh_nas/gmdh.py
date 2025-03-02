import numpy as np
from scipy.linalg import lstsq
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class GMDH:
    def __init__(self, max_layers=3, threshold=0.01, validation_split=0.2):
        self.max_layers = max_layers
        self.threshold = threshold
        self.validation_split = validation_split
        self.models = []

    def _generate_polynomials(self, X):
        """ Generate polynomial terms: x1, x2, x1*x2, x1^2, x2^2, x1^3, x2^3 """
        n_samples, n_features = X.shape
        terms = [X]

        for i in range(n_features):
            for j in range(i, n_features):
                terms.append(X[:, i] * X[:, j])   # x1 * x2
                terms.append(X[:, i] ** 2)        # x1^2
                terms.append(X[:, j] ** 2)        # x2^2
                terms.append(X[:, i] ** 3)        # x1^3
                terms.append(X[:, j] ** 3)        # x2^3

        return np.column_stack(terms)

    def _train_layer(self, X, y):
        """ Train models for a single layer with validation """
        X_poly = self._generate_polynomials(X)
        X_train, X_val, y_train, y_val = train_test_split(X_poly, y, test_size=self.validation_split)

        best_models = []
        for i in range(X_poly.shape[1]):
            try:
                coef, _, _, _ = lstsq(X_train[:, [i]], y_train)
                y_pred = X_val[:, [i]] @ coef
                error = mean_squared_error(y_val, y_pred)
                best_models.append((coef, i, error))
            except Exception as e:
                print(f"⚠️ Skipping invalid feature index {i}: {e}")

        best_models.sort(key=lambda x: x[2])
        return best_models[:5] if best_models else []

    def fit(self, X, y):
        """ Train GMDH model with cross-validation """
        X_current = X
        for layer in range(self.max_layers):
            models = self._train_layer(X_current, y)
            if not models:
                print(f"⚠️ Layer {layer} has no valid models. Stopping early.")
                break 

            self.models.append(models)
            selected_indices = [m[1] for m in models]
            X_poly = self._generate_polynomials(X_current)
            selected_indices = [idx for idx in selected_indices if idx < X_poly.shape[1]]
            X_current = X_poly[:, selected_indices] if selected_indices else np.zeros((X.shape[0], 1))

        if not self.models:
            raise RuntimeError("Training failed: No valid models were generated.")

    def predict(self, X):
        """ Predict using trained GMDH model """
        if not self.models:
            print("⚠️ No trained models available. Returning zeros.")
            return np.zeros(X.shape[0])

        X_current = X
        for layer in self.models:
            poly_features = self._generate_polynomials(X_current)
            selected_indices = [m[1] for m in layer]
            selected_indices = [idx for idx in selected_indices if idx < poly_features.shape[1]]

            if not selected_indices:
                print("⚠️ No valid features found. Returning zeros.")
                return np.zeros(X.shape[0])

            X_current = poly_features[:, selected_indices]

        final_preds = np.zeros(X_current.shape[0])
        for coef, feature_index, _ in self.models[-1]:
            if feature_index >= X_current.shape[1]:  
                print(f"⚠️ Skipping invalid feature index {feature_index}")
                continue

            feature_values = X_current[:, feature_index].reshape(-1, 1)
            final_preds += (feature_values @ coef.reshape(-1, 1)).flatten()

        return final_preds
